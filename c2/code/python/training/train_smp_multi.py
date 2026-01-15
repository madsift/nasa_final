
import os
import argparse
import warnings

# Suppress Pydantic warnings from dependencies (timm, smp)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import bitsandbytes as bnb
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
import random

# Import your modules using package paths (for running as python -m python.training.train_smp_multi)
from python.training.datasets import CraterDataset3, get_valid_transforms, get_train_transforms
from python.models.models_smp import (CraterSMP, CraterSMP_3Ch_RAM, CraterSMP_GroupNorm,
                        CraterSMP_LayerNorm, CraterSMP_LayerNorm_Deep, CraterSMP_LayerNorm_Deep_v2,
                        CraterSMPd)
from python.training.losses import AutomaticWeightedLoss_Boundary


# =============================================================================
# CHANNEL-WISE SOFT STANDARDIZATION (Winning Normalization from Benchmark)
# =============================================================================
class ChannelWiseSoftStd(nn.Module):
    """
    Apply soft standardization independently to each channel.
    
    This normalization strategy won the benchmark for handling domain shift
    (Dark/Low-Contrast Train → Bright/High-Contrast Test).
    
    Formula per channel: (x - mean) / max(std, clamp_val)
    
    Args:
        clamp_val: Minimum std value to prevent noise amplification (default: 0.1)
    """
    def __init__(self, clamp_val=0.1):
        super().__init__()
        self.clamp_val = clamp_val
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=(-1, -2), keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True)
        return (x - mean) / torch.clamp(std, min=self.clamp_val)


def setup(rank, world_size):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def save_checkpoint(state, filename="best_model.pth"):
    print(f"=> Saving checkpoint: {filename}")
    torch.save(state, filename)
    

def plot_training_history(history, save_dir):
    if not history['train_loss']:
        return

    epochs = range(1, len(history['train_loss']) + 1)
    
    try:
        plt.close('all')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        plt.subplots_adjust(hspace=0.3)

        ax1.plot(epochs, history['train_loss'], label='Train Total', color='tab:blue')
        ax1.plot(epochs, history['val_loss'], label='Val Total', color='tab:orange', linestyle='--')
        ax1.set_title('Total Weighted Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        ax2.plot(epochs, history['train_score'], label='Train Score', color='tab:green')
        ax2.plot(epochs, history['val_score'], label='Val Score', color='tab:red', linestyle='--')
        ax2.set_title('Metric')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Score')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        
        ax3.plot(epochs, history['loss_core'], label='Core', color='purple')
        ax3.plot(epochs, history['loss_global'], label='Global', color='brown')
        ax3.plot(epochs, history['loss_rim'], label='Rim', color='cyan')
        ax3.set_title('Component Losses')
        ax3.set_xlabel('Epochs')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend()

        ax4.plot(epochs, history['sigma_core'], label='Sigma Core', color='purple', linestyle='-.')
        ax4.plot(epochs, history['sigma_global'], label='Sigma Global', color='brown', linestyle='-.')
        ax4.plot(epochs, history['sigma_rim'], label='Sigma Rim', color='cyan', linestyle='-.')
        ax4.set_title('Learned Sigma')
        ax4.set_xlabel('Epochs')
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.legend()

        fig.savefig(os.path.join(save_dir, 'training_plot.png'), dpi=300, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error generating plot: {e}")
    finally:
        plt.close('all')


class GaussianAngleMetric:
    """Metric to compute Rim IoU."""
    def __init__(self, rim_idx=2):
        self.rim_idx = rim_idx
        self.reset()
        
    def reset(self):
        self.scores = []
        
    def update(self, pred, target):
        with torch.no_grad():
            rim_target = target[:, 2, :, :] > 0.5
            rim_pred = torch.sigmoid(pred[:, self.rim_idx, :, :]) > 0.5
            intersection = (rim_pred & rim_target).float().sum((1, 2))
            union = (rim_pred | rim_target).float().sum((1, 2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            self.scores.append(iou.mean().item())
            
    def get_avg(self):
        return np.mean(self.scores) if self.scores else 0.0


def get_datasets(data_dir, tile_size=None, use_instance_norm=False, use_channel_norm=False):
    """Create train and val datasets from pre-tiled .npz data."""
    seed = 42
    random.seed(seed)
    
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    # Get all tile IDs from .npz files
    all_files = [f.stem for f in images_dir.glob("*.npz")]
    
    # Group Split Logic (group by altitude + longitude)
    groups = {}
    for fname in all_files:
        parts = fname.split('_')
        group_id = f"{parts[0]}_{parts[1]}"
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(fname)
    
    # Fixed validation groups
    fixed_val_groups = ['altitude01_longitude10']
    
    val_group_ids = [g for g in fixed_val_groups if g in groups]
    train_group_ids = [g for g in groups.keys() if g not in val_group_ids]
    
    train_files = []
    for gid in train_group_ids:
        train_files.extend(groups[gid])
        
    val_files = []
    for gid in val_group_ids:
        val_files.extend(groups[gid])
    
    random.shuffle(train_files)
    
    # Create datasets with tile_size-aware transforms
    train_ds = CraterDataset3(images_dir, masks_dir, train_files, 
                              transform=get_train_transforms(tile_size=tile_size),
                              use_instance_norm=use_instance_norm,
                              use_channel_norm=use_channel_norm)
    val_ds = CraterDataset3(images_dir, masks_dir, val_files, 
                            transform=get_valid_transforms(),
                            use_instance_norm=use_instance_norm,
                            use_channel_norm=use_channel_norm)
    
    return train_ds, val_ds, len(train_files), len(val_files)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, metric, rank, is_main, accum_steps=1,
                    deep_supervision=False, aux_weights=None):
    model.train()
    
    if is_main:
        loop = tqdm(loader, desc="Training")
    else:
        loop = loader
    
    acc = {'total': [], 'core': [], 'global': [], 'rim': [],
           's_core': [], 's_global': [], 's_rim': []}

    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.cuda(rank, non_blocking=True)
        targets = targets.cuda(rank, non_blocking=True)

        with autocast('cuda', enabled=True, dtype=torch.float16):
            output = model(data)
            
            # Handle deep supervision
            if deep_supervision and isinstance(output, tuple):
                predictions, aux_outputs = output
                loss, loss_dict = loss_fn(predictions, targets)
                if aux_weights is None:
                    aux_weights = [0.4, 0.3, 0.2, 0.1]
                for i, aux_out in enumerate(aux_outputs):
                    if i < len(aux_weights):
                        aux_loss, _ = loss_fn(aux_out, targets)
                        loss = loss + aux_weights[i] * aux_loss
            else:
                predictions = output
                loss, loss_dict = loss_fn(predictions, targets)
            
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Periodic cache clearing to free fragmented memory
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

        acc['total'].append(loss.item() * accum_steps)
        acc['core'].append(loss_dict['loss_core'].item())
        acc['global'].append(loss_dict['loss_global'].item())
        acc['rim'].append(loss_dict['loss_rim_total'].item())
        acc['s_core'].append(loss_dict['sigma_core'].item())
        acc['s_global'].append(loss_dict['sigma_global'].item())
        acc['s_rim'].append(loss_dict['sigma_rim'].item())

        metric.update(predictions, targets)
        
        if is_main and hasattr(loop, 'set_postfix'):
            loop.set_postfix(loss=loss.item() * accum_steps)

    return {k: np.mean(v) for k, v in acc.items()}, metric.get_avg()


def validate(loader, model, loss_fn, metric, rank, is_main):
    model.eval()
    
    if is_main:
        loop = tqdm(loader, desc="Validation")
    else:
        loop = loader
    
    acc = {'total': [], 'core': [], 'global': [], 'rim': []}
    metric.reset()

    with torch.no_grad():
        for data, targets in loop:
            data = data.cuda(rank, non_blocking=True)
            targets = targets.cuda(rank, non_blocking=True)
           
            with autocast('cuda', enabled=True, dtype=torch.float16):
                predictions = model(data)
                loss, loss_dict = loss_fn(predictions, targets)
            
            acc['total'].append(loss.item())
            acc['core'].append(loss_dict['loss_core'].item())
            acc['global'].append(loss_dict['loss_global'].item())
            acc['rim'].append(loss_dict['loss_rim_total'].item())
            
            metric.update(predictions, targets)
            
    return {k: np.mean(v) for k, v in acc.items()}, metric.get_avg()


def train_worker(rank, world_size, args):
    """Worker function for each GPU."""
    
    # Setup distributed
    setup(rank, world_size)
    is_main = is_main_process(rank)
    
    if is_main:
        print(f"=" * 60)
        print(f"MULTI-GPU TRAINING with DDP + SyncBatchNorm")
        print(f"=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Effective Batch Size: {args.batch_size * world_size}")
        print(f"Epochs: {args.num_epochs}")
        if args.use_channel_norm:
            print(f"Normalization: ChannelWiseSoftStd(0.1) [RECOMMENDED]")
        elif args.use_instance_norm:
            print(f"Normalization: Instance Norm (x - mean)")
        else:
            print(f"Normalization: Fixed (x - 0.5) / 0.5")
        print(f"=" * 60)
    
    # Save directory
    if args.name_prefix:
        save_dir = f'./{args.name_prefix}_{args.model}_{args.backbone}'
    else:
        save_dir = f'./{args.model}_{args.backbone}'
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
    
    # Parse tile_size from string "W H" format
    tile_size = None
    if args.tile_size:
        parts = args.tile_size.strip().split()
        if len(parts) == 1:
            tile_size = (int(parts[0]), int(parts[0]))
        elif len(parts) >= 2:
            tile_size = (int(parts[0]), int(parts[1]))
        if is_main:
            print(f">>> Tile size: {tile_size} <<<")
    
    # Create datasets
    train_ds, val_ds, n_train, n_val = get_datasets(
        args.data_dir, 
        tile_size=tile_size, 
        use_instance_norm=args.use_instance_norm,
        use_channel_norm=args.use_channel_norm
    )
    
    if is_main:
        print(f"Training: {n_train} tiles | Validation: {n_val} tiles")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=2, pin_memory=True)
    
    # Parse decoder_channels
    decoder_channels = tuple(int(x) for x in args.decoder_channels.split(','))
    
    # Create model: Always 3 input channels, 3 output classes (Core, Global, Rim)
    if args.model == "CraterSMP_3Ch_RAM":
        model = CraterSMP_3Ch_RAM(backbone=args.backbone, num_classes=3)
    elif args.model == "CraterSMP_GroupNorm":
        model = CraterSMP_GroupNorm(backbone=args.backbone, num_classes=3)
        model.freeze_encoder_batchnorm()  # Freeze encoder BN to use pretrained stats
    elif args.model == "CraterSMP_LayerNorm":
        if is_main:
            print(f">>> Using CraterSMP_LayerNorm decoder_channels={decoder_channels} <<<")
        model = CraterSMP_LayerNorm(
            backbone=args.backbone,
            in_channels=3,
            num_classes=3,
            decoder_channels=decoder_channels
        )
    elif args.model == "CraterSMP_LayerNorm_Deep":
        if is_main:
            print(f">>> Using CraterSMP_LayerNorm_Deep (with Deep Supervision) decoder_channels={decoder_channels} <<<")
        model = CraterSMP_LayerNorm_Deep(
            backbone=args.backbone,
            in_channels=3,
            num_classes=3,
            decoder_channels=decoder_channels,
            deep_supervision=True
        )
    elif args.model == "CraterSMP_LayerNorm_Deep_v2":
        if is_main:
            print(f">>> Using CraterSMP_LayerNorm_Deep_v2 (Affine-less LN + LayerScale) decoder_channels={decoder_channels} <<<")
            print(f"    - Affine-less LayerNorm in decoder (scale-invariant)")
            print(f"    - LayerScale init={args.layer_scale_init} for gradient stability")
        model = CraterSMP_LayerNorm_Deep_v2(
            backbone=args.backbone,
            in_channels=3,
            num_classes=3,
            decoder_channels=decoder_channels,
            deep_supervision=True,
            layer_scale_init=args.layer_scale_init
        )
    elif args.model == "CraterSMP":
        model = CraterSMP(backbone=args.backbone, num_classes=3, decoder_channels=decoder_channels)
    elif args.model == "CraterSMPd":
        if is_main:
            print(f">>> Using CraterSMPd (Depthwise-Separable Decoder) decoder_channels={decoder_channels} <<<")
        model = CraterSMPd(
            backbone=args.backbone,
            in_channels=3,
            num_classes=3,
            decoder_channels=decoder_channels
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Convert to SyncBatchNorm BEFORE moving to GPU
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if is_main:
        print(f">>> SyncBatchNorm + DDP enabled <<<")
    
    # Parse sigma clamp range
    sigma_clamp_parts = args.sigma_clamp.split(',')
    sigma_clamp_min = float(sigma_clamp_parts[0])
    sigma_clamp_max = float(sigma_clamp_parts[1])
    if is_main:
        print(f">>> Sigma clamp range: [{sigma_clamp_min}, {sigma_clamp_max}] <<<")
    
    # Loss function: Always 3 losses (Core, Global, Rim), global always enabled
    loss_fn = AutomaticWeightedLoss_Boundary(
        num_losses=3, use_global=True,
        sigma_clamp_min=sigma_clamp_min, sigma_clamp_max=sigma_clamp_max
    ).cuda(rank)
    
    # Optimizer
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = bnb.optim.AdamW8bit(params, lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # History (only main process)
    history = {'train_loss': [], 'val_loss': [], 'train_score': [], 'val_score': [],
               'loss_core': [], 'loss_global': [], 'loss_rim': [],
               'sigma_core': [], 'sigma_global': [], 'sigma_rim': []}
    
    best_val_score = 0.0
    start_epoch = 0
    
    # Resume
    if args.resume and os.path.isfile(args.resume):
        if is_main:
            print(f"=> Loading checkpoint '{args.resume}'")
        map_location = {'cuda:0': f'cuda:{rank}'}
        ckpt = torch.load(args.resume, map_location=map_location, weights_only=False)
        
        state = ckpt['state_dict']
        if list(state.keys())[0].startswith('module.'):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        
        model.module.load_state_dict(state, strict=False)
        start_epoch = ckpt.get('epoch', 0)
        best_val_score = ckpt.get('best_val_score', 0.0)
        if is_main:
            print(f"=> Resumed from epoch {start_epoch}")
    
    # Rim is always at index 2 (Core=0, Global=1, Rim=2)
    train_metric = GaussianAngleMetric(rim_idx=2)
    val_metric = GaussianAngleMetric(rim_idx=2)
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        
        if is_main:
            print(f"\nEpoch {epoch+1}/{args.num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_metric.reset()
        
        # Parse aux_weights for deep supervision
        aux_weights = None
        if args.aux_weights:
            aux_weights = [float(x) for x in args.aux_weights.split(',')]
        
        use_deep_supervision = args.model in ['CraterSMP_LayerNorm_Deep', 'CraterSMP_LayerNorm_Deep_v2']
        
        train_avgs, train_score = train_one_epoch(
            train_loader, model, optimizer, loss_fn, scaler, train_metric, 
            rank, is_main, accum_steps=args.accum_steps,
            deep_supervision=use_deep_supervision, aux_weights=aux_weights
        )
        
        val_avgs, val_score = validate(val_loader, model, loss_fn, val_metric, rank, is_main)
        
        scheduler.step()
        
        # Logging and checkpointing (main process only)
        if is_main:
            history['train_loss'].append(train_avgs['total'])
            history['val_loss'].append(val_avgs['total'])
            history['train_score'].append(train_score)
            history['val_score'].append(val_score)
            history['loss_core'].append(train_avgs['core'])
            history['loss_global'].append(train_avgs['global'])
            history['loss_rim'].append(train_avgs['rim'])
            history['sigma_core'].append(train_avgs['s_core'])
            history['sigma_global'].append(train_avgs['s_global'])
            history['sigma_rim'].append(train_avgs['s_rim'])

            print(f"Train Loss: {train_avgs['total']:.4f} | Val Loss: {val_avgs['total']:.4f}")
            print(f"Val Score: {val_score:.4f} | Best: {best_val_score:.4f}")
            
            ckpt_state = {
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_score": best_val_score,
            }
            
            if val_score > best_val_score:
                best_val_score = val_score
                ckpt_state['best_val_score'] = best_val_score
                save_checkpoint(ckpt_state, filename=f"{save_dir}/best_model.pth")
            
            if epoch % 3 == 0 and epoch != 0:
                save_checkpoint(ckpt_state, filename=f"{save_dir}/checkpoint_{epoch}.pth")
            
            plot_training_history(history, save_dir)
        
        # Sync before next epoch
        dist.barrier()
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training with DDP + SyncBatchNorm")
    
    # GPU settings
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size PER GPU")
    parser.add_argument("--num_epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--model", type=str, default="CraterSMP_GroupNorm", help="Model to use")
    # Model settings
    parser.add_argument("--backbone", type=str, default="mobileone_s2")
    parser.add_argument("--data_dir", type=str, default="./train_tiles")
    parser.add_argument("--tile_size", type=str, default="672 544", help="Tile size as 'W H' for rectangular tile handling")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_instance_norm", action='store_true', 
                        help="Use mean-centering: x = x - mean(x) instead of fixed (x-0.5)/0.5")
    parser.add_argument("--use_channel_norm", action='store_true',
                        help="Use ChannelWiseSoftStd(0.1): per-channel soft standardization. "
                             "RECOMMENDED for domain shift robustness.")
    parser.add_argument("--decoder_channels", type=str, default='256,128,64,32,16',
                        help="Decoder channel sizes as comma-separated values (default: 256,128,64,32,16)")
    parser.add_argument("--aux_weights", type=str, default=None,
                        help="Comma-separated auxiliary loss weights for deep supervision (default: 0.4,0.3,0.2,0.1)")
    parser.add_argument("--layer_scale_init", type=float, default=1e-5,
                        help="Initial value for LayerScale gamma (default: 1e-5). Only used with CraterSMP_LayerNorm_Deep_v2.")
    parser.add_argument("--sigma_clamp", type=str, default="-5.0,10.0",
                        help="Min,max clamp range for learned sigma params (default: -5.0,10.0). "
                             "Tighter range like -2.0,2.0 can stabilize training for larger tiles.")
    parser.add_argument("--name_prefix", type=str, default=None,
                        help="Prefix for output directory name (e.g., 'exp1' -> 'exp1_CraterSMP_LayerNorm_mobileone_s2')")
    
    args = parser.parse_args()
    
    # Validate GPU count
    available_gpus = torch.cuda.device_count()
    if args.gpus > available_gpus:
        print(f"Warning: Requested {args.gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        args.gpus = available_gpus
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Devices: {available_gpus}")
    for i in range(available_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if args.gpus == 1:
        # Single GPU - run directly
        print("\n>>> Single GPU mode <<<")
        train_worker(0, 1, args)
    else:
        # Multi GPU - spawn workers
        print(f"\n>>> Multi-GPU mode: {args.gpus} GPUs <<<")
        mp.spawn(train_worker, args=(args.gpus, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
