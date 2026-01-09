import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class CraterSMP(nn.Module):
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=2):
        super(CraterSMP, self).__init__()
        
        # SMP automatically adjusts the first layer weights from RGB (3) to Grayscale (1)
        # by summing/averaging the pretrained ImageNet weights.
        self.model = smp.Unet(
            encoder_name=backbone,      # "resnet34", "efficientnet-b3", etc.
            encoder_weights="imagenet", # Use pretrained weights
            in_channels=in_channels,              # Grayscale input
            classes=num_classes,        # 2 channels: Rim, Core
            activation=None             # Return Logits (we apply Sigmoid in Loss/Eval)
        )

    def forward(self, x):
        return self.model(x)


def replace_bn_with_gn(module, num_groups=8, _count=None):
    """
    Recursively replace all BatchNorm2d layers with GroupNorm.
    
    GroupNorm normalizes per-sample (not per-batch), making it robust
    to distribution shift between training and test sets.
    
    Args:
        module: PyTorch module to modify
        num_groups: Number of groups for GroupNorm (8 is typical, must divide num_channels)
        _count: Internal counter dict (do not pass manually)
    
    Returns:
        Modified module with GroupNorm instead of BatchNorm
    """
    # Initialize counter on first call
    if _count is None:
        _count = {'replaced': 0}
        is_root = True
    else:
        is_root = False
    
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_groups divides num_channels evenly
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            
            # Create GroupNorm with same number of channels
            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine
            )
            
            # Copy affine parameters if available
            if child.affine and child.weight is not None:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            
            setattr(module, name, gn)
            _count['replaced'] += 1
        else:
            # Recurse into child modules
            replace_bn_with_gn(child, num_groups, _count)
    
    # Print count only on root call
    if is_root:
        print(f"[replace_bn_with_gn] Replaced {_count['replaced']} BatchNorm2d layers with GroupNorm")
    
    return module


class CraterSMP_GroupNorm(nn.Module):
    """
    CraterSMP with GroupNorm instead of BatchNorm for distribution-shift robustness.
    
    BatchNorm uses batch statistics during training but frozen running stats during
    inference. If test distribution differs from training, these stats are wrong.
    
    GroupNorm normalizes per-sample (independent of batch), so test-time behavior
    is identical regardless of input distribution. This is critical when:
    - Test images have different lighting conditions
    - Test has different crater size distributions
    - Test comes from different terrain/regions
    
    Usage:
        model = CraterSMP_GroupNorm(backbone="mobileone_s2", in_channels=3, num_classes=3)
        # Train from scratch - ImageNet weights will be converted
    
    Args:
        backbone: Encoder backbone name (default: mobileone_s2)
        in_channels: Number of input channels (default: 3 for Gray+DEM+Grad)
        num_classes: Number of output classes (default: 3 for Core/Global/Rim)
        num_groups: Number of groups for GroupNorm (default: 8)
    """
    
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=3, num_groups=8):
        super(CraterSMP_GroupNorm, self).__init__()
        
        # Create base SMP model with BatchNorm
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        # HYBRID APPROACH: Only replace BatchNorm in decoder and segmentation_head
        # Keep encoder BatchNorm intact for MobileOne reparameterization compatibility
        decoder_replaced = replace_bn_with_gn(self.model.decoder, num_groups=num_groups)
        head_replaced = replace_bn_with_gn(self.model.segmentation_head, num_groups=num_groups)
        
        self.num_groups = num_groups
        print(f"[CraterSMP_GroupNorm] Hybrid mode: GroupNorm in decoder/head only, encoder keeps BatchNorm for reparameterization")
        
        # Freeze encoder BatchNorm parameters (but keep in train mode for stability)
        self.freeze_encoder_batchnorm()

    def freeze_encoder_batchnorm(self):
        """
        Freeze BatchNorm parameters in the encoder but KEEP them in train mode.
        
        Why not eval() mode?
        - eval() uses pretrained ImageNet running_mean/running_var
        - Lunar images have very different distribution from ImageNet
        - This causes numerical instability (NaN gradients)
        
        This approach:
        - BN stays in train() mode -> uses current batch statistics (stable)
        - gamma/beta frozen -> no parameter updates
        - Running stats still update but that's fine since we're not using them
        """
        frozen_count = 0

        for m in self.model.encoder.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                # DON'T call m.eval() - keep using batch statistics
                # Just freeze the learnable parameters
                for p in m.parameters():
                    p.requires_grad = False

                frozen_count += 1

        print(f"[CraterSMP_GroupNorm] Frozen {frozen_count} encoder BatchNorm parameters (kept in train mode)")

    
    def train(self, mode=True):
        """
        Standard train mode - no special handling needed since we don't force eval() on BN.
        """
        super().train(mode)
        return self

    def _encoder_forward(self, x):
        """Wrapper for encoder forward to enable checkpointing."""
        return self.model.encoder(x)
    
    def _decoder_forward(self, features):
        """Wrapper for decoder forward to enable checkpointing."""
        decoder_output = self.model.decoder(features)
        return self.model.segmentation_head(decoder_output)

    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(features)
        masks = self.model.segmentation_head(decoder_output)
        
        return masks
    
    def load_from_batchnorm_checkpoint(self, checkpoint_path, strict=False):
        """
        Load weights from a BatchNorm-based checkpoint.
        
        BatchNorm running_mean/running_var will be ignored.
        Affine weights (gamma/beta) will be copied to GroupNorm.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, raise error on missing/unexpected keys
        
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out BatchNorm running stats (not needed for GroupNorm)
        filtered_state = {}
        for k, v in state_dict.items():
            if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                continue  # Skip BatchNorm running stats
            filtered_state[k] = v
        
        # Try to load, mapping keys if needed
        missing, unexpected = self.load_state_dict(filtered_state, strict=strict)
        
        # Filter out expected differences
        missing_filtered = [k for k in missing if 'running' not in k and 'num_batches' not in k]
        
        if missing_filtered:
            print(f"[Warning] Missing keys: {missing_filtered[:5]}{'...' if len(missing_filtered) > 5 else ''}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        
        print(f"[CraterSMP_GroupNorm] Loaded weights from {checkpoint_path}")
        return missing_filtered, unexpected

# --- 1. EFFICIENT RAM (Memory Optimized + AMP Safe + ONNX Exportable) ---
# All high-impact fixes applied:
# - Fix #1: Shadow mask as bias BEFORE softmax (proper attention distribution)
# - Fix #2: Project K/V first, then pool (preserves more information)
# - Fix #3: Gate uses both input AND output (attention-aware gating)
# - Fix #4: More heads (8) for diverse attention patterns
class EfficientRegionalAttention(nn.Module):
    def __init__(self, in_channels, num_regions=8, heads=8, dim_red=4):
        """
        Enhanced Regional Attention for crater detection.
        
        Args:
            in_channels: Input feature channels
            num_regions: Grid size for pooling K/V (e.g., 16 means 16x16 = 256 reference points)
            heads: Number of attention heads (8 for diverse rim patterns)
            dim_red: Dimension reduction factor for Q/K/V projections
        """
        super().__init__()
        self.num_regions = num_regions
        self.heads = heads
        self.d_k = max(8, in_channels // dim_red // heads)
        
        # Projections
        self.query = nn.Conv2d(in_channels, heads * self.d_k, 1)
        self.key = nn.Conv2d(in_channels, heads * self.d_k, 1)
        self.value = nn.Conv2d(in_channels, heads * self.d_k, 1)
        
        self.out_proj = nn.Conv2d(heads * self.d_k, in_channels, 1)
        
        # Fix #3: Gate uses both input AND attention output
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Shadow suppression strength (learnable, initialized to -4.0)
        self.shadow_bias_scale = nn.Parameter(torch.tensor(-4.0))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Shadow Detection (Stability Fix for AMP)
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            var_map = torch.var(x_fp32, dim=1, keepdim=True)
            mu = var_map.mean(dim=(2, 3), keepdim=True)
            std = var_map.std(dim=(2, 3), keepdim=True)
            shadow_mask = (var_map < (mu - std)).float()
        
        # 2. Queries at full resolution -> [B, Heads, HW, D]
        q = self.query(x).view(B, self.heads, self.d_k, -1).transpose(-2, -1)
        
        # 3. Fix #2: Project K/V first, THEN pool (preserves more info)
        k_full = self.key(x)    # [B, heads*d_k, H, W]
        v_full = self.value(x)  # [B, heads*d_k, H, W]
        
        # Pool after projection using adaptive avg pool (ONNX-safe via interpolate)
        k_pooled = F.interpolate(k_full, size=(self.num_regions, self.num_regions), 
                                  mode='bilinear', align_corners=False)
        v_pooled = F.interpolate(v_full, size=(self.num_regions, self.num_regions), 
                                  mode='bilinear', align_corners=False)
        
        k = k_pooled.view(B, self.heads, self.d_k, -1).transpose(-2, -1)  # [B, heads, N_regions, d_k]
        v = v_pooled.view(B, self.heads, self.d_k, -1).transpose(-2, -1)
        
        # 4. Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)
        
        # 5. Fix #1: Apply shadow mask as BIAS before softmax
        # Non-shadow regions get suppressed (large negative bias)
        m = F.interpolate(shadow_mask, size=(H, W), mode='nearest').view(B, 1, H*W, 1)
        shadow_bias = (1 - m) * self.shadow_bias_scale  # Non-shadow gets negative bias
        attn = attn + shadow_bias
        
        attn = F.softmax(attn, dim=-1)
        
        # 6. Aggregate
        out = torch.matmul(attn, v)  # [B, Heads, HW, D]
        out = out.transpose(-2, -1).reshape(B, -1, H, W)
        out = self.out_proj(out)
        
        # 7. Fix #3: Gate using both input AND output
        gate = self.gate(torch.cat([x, out], dim=1))
        
        return x + gate * out

# --- 2. MAIN MODEL (With Full Checkpointing) ---
class CraterSMP_3Ch_RAM(nn.Module):
    def __init__(self, backbone="mobileone_s2",in_channels=3, num_classes=3):
        super().__init__()
        
        # 1. Base Model
        self.base_model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet", 
            in_channels=3, # Texture, DEM, Grad
            classes=num_classes,
            activation=None
        )
        
        # 2. Dynamic Channel Detection
        encoder_channels = self.base_model.encoder.out_channels
        
        # 3. Integrate Efficient RAM (High Res Config for 1024px)
        # Stage 2 (Stride 4): Use 32 regions to stop "feature smearing"
        self.ram_stage2 = EfficientRegionalAttention(
            in_channels=encoder_channels[2], 
            num_regions=12 #16 # High Precision Grid
        )
        # Stage 3 (Stride 8): Use 16 regions
        self.ram_stage3 = EfficientRegionalAttention(
            in_channels=encoder_channels[3], 
            num_regions=6 #8
        )

    def _encoder_forward(self, x):
        """Wrapper for encoder forward to enable checkpointing."""
        return self.base_model.encoder(x)
    
    def _decoder_forward(self, features):
        """Wrapper for decoder forward to enable checkpointing."""
        decoder_output = self.base_model.decoder(features)
        return self.base_model.segmentation_head(decoder_output)

    def forward(self, x):
        # 1. Encoder Pass with GRADIENT CHECKPOINTING
        if self.training:
            # Checkpoint the entire encoder for memory savings
            # Note: checkpoint requires at least one tensor input with requires_grad=True
            features = checkpoint.checkpoint(self._encoder_forward, x, use_reentrant=False)
        else:
            features = self.base_model.encoder(x)
        
        # 2. Apply RAM with GRADIENT CHECKPOINTING
        if self.training:
            features[2] = checkpoint.checkpoint(self.ram_stage2, features[2], use_reentrant=False)
            features[3] = checkpoint.checkpoint(self.ram_stage3, features[3], use_reentrant=False)
        else:
            features[2] = self.ram_stage2(features[2])
            features[3] = self.ram_stage3(features[3])
        
        # 3. Decoder Pass with GRADIENT CHECKPOINTING
        if self.training:
            # Convert features list to tuple for checkpointing
            # Note: We need to be careful here as features is a list
            masks = checkpoint.checkpoint(self._decoder_forward, features, use_reentrant=False)
        else:
            decoder_output = self.base_model.decoder(features)
            masks = self.base_model.segmentation_head(decoder_output)
        
        return masks


if __name__ == "__main__":
    # Final Verification
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize
    model = CraterSMP_GroupNorm("mobileone_s2", num_classes=3).to(device)
    
    # Batch Size 6
    x = torch.randn(4, 3, 544, 416).to(device)
    
    print(f"Testing Model on {device} with Batch Size 6...")
    
    # Run in FP16 (Simulating Training)
    with torch.cuda.amp.autocast():
        y = model(x)
        
    print(f"Success! Output Shape: {y.shape}")
    # Expected: torch.Size([6, 3, 768, 768])
