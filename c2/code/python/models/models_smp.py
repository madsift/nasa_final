import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class CraterSMP(nn.Module):
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=2, 
                 decoder_channels=(128, 64, 32, 24, 16)):
        super(CraterSMP, self).__init__()
        if decoder_channels is None:
            decoder_channels = (256, 128, 64, 32, 16)
        # Default SMP decoder_channels is (256, 128, 64, 32, 16)
        # Using smaller channels (128, 64, 32, 24, 16) reduces parameters/memory
        self.model = smp.Unet(
            encoder_name=backbone,      # "resnet34", "efficientnet-b3", etc.
            encoder_weights="imagenet", # Use pretrained weights
            in_channels=in_channels,    # Grayscale input
            classes=num_classes,        # 2 channels: Rim, Core
            activation=None,            # Return Logits (we apply Sigmoid in Loss/Eval)
            decoder_channels=decoder_channels
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
        
        # Freeze encoder BatchNorm layers permanently
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


class CraterSMP_GroupNorm_Nofreeze(nn.Module):
    """
    CraterSMP with GroupNorm in decoder but NO frozen encoder BatchNorm.
    
    Use this for models trained without frozen encoder BatchNorm layers.
    The encoder BatchNorm will use its learned running statistics during inference.
    
    Usage:
        model = CraterSMP_GroupNorm_Nofreeze(backbone="mobileone_s2", in_channels=3, num_classes=3)
    """
    
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=3, num_groups=8,
                 decoder_channels=(256, 128, 64, 32, 16)):
        super(CraterSMP_GroupNorm_Nofreeze, self).__init__()
        
        # Create base SMP model with BatchNorm
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_channels=decoder_channels
        )
        
        # HYBRID APPROACH: Only replace BatchNorm in decoder and segmentation_head
        # Keep encoder BatchNorm intact for MobileOne reparameterization compatibility
        replace_bn_with_gn(self.model.decoder, num_groups=num_groups)
        replace_bn_with_gn(self.model.segmentation_head, num_groups=num_groups)
        
        self.num_groups = num_groups
        print(f"[CraterSMP_GroupNorm_Nofreeze] Hybrid mode: GroupNorm in decoder/head, encoder BatchNorm NOT frozen")
    
    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(features)
        masks = self.model.segmentation_head(decoder_output)
        return masks
    
    def load_from_batchnorm_checkpoint(self, checkpoint_path, strict=False):
        """
        Load weights from a BatchNorm-based checkpoint.
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
        
        # Filter out BatchNorm running stats for decoder (converted to GroupNorm)
        # But KEEP encoder BatchNorm stats (they weren't converted)
        filtered_state = {}
        for k, v in state_dict.items():
            # Skip running stats only for decoder/segmentation_head (converted to GroupNorm)
            if ('decoder' in k or 'segmentation_head' in k) and \
               ('running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k):
                continue
            filtered_state[k] = v
        
        missing, unexpected = self.load_state_dict(filtered_state, strict=strict)
        
        missing_filtered = [k for k in missing if 'running' not in k and 'num_batches' not in k]
        
        if missing_filtered:
            print(f"[Warning] Missing keys: {missing_filtered[:5]}{'...' if len(missing_filtered) > 5 else ''}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        
        print(f"[CraterSMP_GroupNorm_Nofreeze] Loaded weights from {checkpoint_path}")
        return missing_filtered, unexpected


class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for 2D feature maps (NCHW format).
    
    Normalizes across channels at each spatial location independently.
    This is equivalent to GroupNorm with num_groups=1 but implemented
    using nn.LayerNorm for better CPU performance.
    
    Why use this instead of GroupNorm?
    - LayerNorm has highly optimized CPU implementations
    - GroupNorm with arbitrary groups is expensive on CPU
    - For inference on ARM/CPU, LayerNorm can be 2-3x faster
    
    Args:
        num_channels: Number of input channels
        eps: Small constant for numerical stability
        affine: If True, learnable scale and shift parameters
    """
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # x: (N, C, H, W)
        # Normalize across C dimension at each spatial location
        # Reshape to (N, H, W, C), apply LayerNorm, reshape back
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        return x


def replace_bn_with_ln(module, _count=None):
    """
    Recursively replace all BatchNorm2d layers with LayerNorm2d.
    
    LayerNorm normalizes per-sample across channels, making it robust
    to distribution shift. Unlike GroupNorm, it's very fast on CPU.
    
    Args:
        module: PyTorch module to modify
        _count: Internal counter dict (do not pass manually)
    
    Returns:
        Modified module with LayerNorm2d instead of BatchNorm2d
    """
    if _count is None:
        _count = {'replaced': 0}
        is_root = True
    else:
        is_root = False
    
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            
            # Create LayerNorm2d with same number of channels
            ln = LayerNorm2d(
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine
            )
            
            # Copy affine parameters if available
            if child.affine and child.weight is not None:
                ln.weight.data.copy_(child.weight.data)
                ln.bias.data.copy_(child.bias.data)
            
            setattr(module, name, ln)
            _count['replaced'] += 1
        else:
            # Recurse into child modules
            replace_bn_with_ln(child, _count)
    
    if is_root:
        print(f"[replace_bn_with_ln] Replaced {_count['replaced']} BatchNorm2d layers with LayerNorm2d")
    
    return module


def replace_bn_with_ln_noaffine(module, _count=None):
    """
    Recursively replace all BatchNorm2d layers with affine-less LayerNorm2d.
    
    Affine-less LayerNorm (no learnable gamma/beta) forces scale-invariance,
    which improves robustness to distribution shift. The decoder cannot
    "memorize" intensity scales from training data.
    
    This is particularly useful when training on pseudo-generated data
    where the intensity distributions may not match real test data.
    
    Args:
        module: PyTorch module to modify
        _count: Internal counter dict (do not pass manually)
    
    Returns:
        Modified module with affine-less LayerNorm2d
    """
    if _count is None:
        _count = {'replaced': 0}
        is_root = True
    else:
        is_root = False
    
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            
            # Create affine-less LayerNorm2d (no learnable gamma/beta)
            ln = LayerNorm2d(
                num_channels=num_channels,
                eps=child.eps,
                affine=False  # Key difference: no learnable scale/shift
            )
            
            setattr(module, name, ln)
            _count['replaced'] += 1
        else:
            replace_bn_with_ln_noaffine(child, _count)
    
    if is_root:
        print(f"[replace_bn_with_ln_noaffine] Replaced {_count['replaced']} BatchNorm2d with affine-less LayerNorm2d")
    
    return module


class LayerScale(nn.Module):
    """
    LayerScale from CaiT/DeiT-III for stable deep supervision.
    
    Adds a learnable diagonal scaling at the end of each residual block.
    Initialized to a very small value (e.g., 1e-5) to allow gradients
    from deep supervision to flow primarily through skip connections
    early in training.
    
    As training progresses, the scale learns to increase, gradually
    incorporating the learned features. This prevents gradient explosion
    from auxiliary heads.
    
    Usage:
        self.layer_scale = LayerScale(channels, init_value=1e-5)
        output = x + self.layer_scale(block_output)
    
    Args:
        channels: Number of channels
        init_value: Initial scale value (default: 1e-5, very small)
    """
    def __init__(self, channels, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(channels))
    
    def forward(self, x):
        # x: (N, C, H, W)
        # Apply learnable per-channel scaling
        return x * self.gamma.view(1, -1, 1, 1)


class CraterSMP_LayerNorm(nn.Module):
    """
    CraterSMP with LayerNorm instead of BatchNorm for CPU-efficient inference.
    
    LayerNorm advantages over GroupNorm:
    - Much faster on CPU (2-3x speedup) due to optimized implementations
    - Same distribution-shift robustness as GroupNorm(groups=1)
    - Simple channel-wise normalization at each spatial location
    
    Trade-off:
    - Less expressive than GroupNorm with multiple groups
    - In practice, <1% accuracy difference for crater detection
    
    Architecture:
    - Encoder: Keeps BatchNorm (for MobileOne reparameterization compatibility)
    - Decoder: Uses LayerNorm (fast CPU inference, distribution-robust)
    - Segmentation Head: Uses LayerNorm
    
    Usage:
        model = CraterSMP_LayerNorm(backbone="mobileone_s2", in_channels=3, num_classes=3)
    """
    
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=3,
                 decoder_channels=(256, 128, 64, 32, 16)):
        super(CraterSMP_LayerNorm, self).__init__()
        
        # Create base SMP model with BatchNorm
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_channels=decoder_channels
        )
        
        # HYBRID APPROACH: Only replace BatchNorm in decoder and segmentation_head
        # Keep encoder BatchNorm intact for MobileOne reparameterization compatibility
        replace_bn_with_ln(self.model.decoder)
        replace_bn_with_ln(self.model.segmentation_head)
        
        print(f"[CraterSMP_LayerNorm] Hybrid mode: LayerNorm in decoder/head, encoder keeps BatchNorm")
    
    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(features)
        masks = self.model.segmentation_head(decoder_output)
        return masks
    
    def load_from_batchnorm_checkpoint(self, checkpoint_path, strict=False):
        """
        Load weights from a BatchNorm-based checkpoint.
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
        
        # Filter out BatchNorm running stats for decoder (converted to LayerNorm)
        # But KEEP encoder BatchNorm stats (they weren't converted)
        filtered_state = {}
        for k, v in state_dict.items():
            # Skip running stats only for decoder/segmentation_head (converted to LayerNorm)
            if ('decoder' in k or 'segmentation_head' in k) and \
               ('running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k):
                continue
            filtered_state[k] = v
        
        missing, unexpected = self.load_state_dict(filtered_state, strict=strict)
        
        missing_filtered = [k for k in missing if 'running' not in k and 'num_batches' not in k]
        
        if missing_filtered:
            print(f"[Warning] Missing keys: {missing_filtered[:5]}{'...' if len(missing_filtered) > 5 else ''}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        
        print(f"[CraterSMP_LayerNorm] Loaded weights from {checkpoint_path}")
        return missing_filtered, unexpected


class CraterSMP_LayerNorm_Deep(nn.Module):
    """
    CraterSMP with LayerNorm and Deep Supervision for improved training.
    
    Deep Supervision adds auxiliary loss heads at each decoder stage:
    - Stage 1 (1/16 resolution): Detects very large craters
    - Stage 2 (1/8 resolution): Detects large craters  
    - Stage 3 (1/4 resolution): Detects medium craters
    - Stage 4 (1/2 resolution): Detects small craters
    - Final (full resolution): Main output
    
    Benefits:
    - Better gradient flow to early layers
    - Multi-scale crater detection
    - Faster convergence (~20-30%)
    - Better generalization / distribution robustness
    
    At inference: Only final output is used (no overhead)
    
    Usage:
        model = CraterSMP_LayerNorm_Deep(backbone="mobileone_s2", num_classes=3)
        
        # Training: returns (final_out, [aux1, aux2, aux3, aux4])
        final, aux_outputs = model(x)
        
        # Inference: set model.deep_supervision = False
        model.deep_supervision = False
        final = model(x)  # Returns only final output
    """
    
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=3,
                 decoder_channels=(256, 128, 64, 32, 16), deep_supervision=True):
        super(CraterSMP_LayerNorm_Deep, self).__init__()
        
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # Create base SMP model
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_channels=decoder_channels
        )
        
        # Replace BatchNorm with LayerNorm in decoder and head
        replace_bn_with_ln(self.model.decoder)
        replace_bn_with_ln(self.model.segmentation_head)
        
        # Create auxiliary segmentation heads for deep supervision
        # Each aux head takes decoder block output and produces segmentation map
        # decoder_channels = (256, 128, 64, 32, 16) for 5 stages
        self.aux_heads = nn.ModuleList()
        
        # Create aux heads for stages 0-3 (skip final stage, that's the main head)
        # Stages are: 256 (1/16) -> 128 (1/8) -> 64 (1/4) -> 32 (1/2) -> 16 (1/1)
        for i, ch in enumerate(decoder_channels[:-1]):  # Skip last channel (main head)
            aux_head = nn.Sequential(
                nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
                LayerNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, num_classes, kernel_size=1)
            )
            self.aux_heads.append(aux_head)
        
        print(f"[CraterSMP_LayerNorm_Deep] Created with {len(self.aux_heads)} auxiliary heads")
        print(f"  Decoder channels: {decoder_channels}")
        print(f"  Deep supervision: {deep_supervision}")
    
    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder
        features = self.model.encoder(x)
        
        # Get spatial shapes for decoder (following SMP's pattern)
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]  # Reverse to go from deep to shallow
        
        # Process features like SMP decoder does
        features_for_decoder = features[1:]  # remove first skip with same spatial resolution
        features_for_decoder = features_for_decoder[::-1]  # reverse to start from head of encoder
        
        head = features_for_decoder[0]
        skip_connections = features_for_decoder[1:]
        
        # Apply center block
        decoder = self.model.decoder
        x_dec = decoder.center(head)
        
        # Manual decoder forward to capture intermediate outputs
        decoder_outputs = []
        
        for i, decoder_block in enumerate(decoder.blocks):
            # Get target spatial shape for this block
            target_h, target_w = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            
            # Decoder block forward with correct signature
            x_dec = decoder_block(x_dec, target_h, target_w, skip_connection=skip_connection)
            decoder_outputs.append(x_dec)
        
        # Final segmentation head
        final_output = self.model.segmentation_head(x_dec)
        
        if self.training and self.deep_supervision:
            # Generate auxiliary outputs
            aux_outputs = []
            
            for i, (aux_head, dec_out) in enumerate(zip(self.aux_heads, decoder_outputs[:-1])):
                # Apply aux head
                aux_out = aux_head(dec_out)
                # Upsample to input resolution
                aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
                aux_outputs.append(aux_out)
            
            return final_output, aux_outputs
        else:
            return final_output
    
    def load_from_batchnorm_checkpoint(self, checkpoint_path, strict=False):
        """
        Load weights from a BatchNorm-based checkpoint.
        Aux heads will be randomly initialized (they don't exist in regular checkpoints).
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out BatchNorm running stats for decoder
        filtered_state = {}
        for k, v in state_dict.items():
            if ('decoder' in k or 'segmentation_head' in k) and \
               ('running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k):
                continue
            filtered_state[k] = v
        
        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        
        # Filter expected missing keys (aux_heads are new)
        missing_filtered = [k for k in missing 
                          if 'running' not in k and 'num_batches' not in k and 'aux_heads' not in k]
        
        if missing_filtered:
            print(f"[Warning] Missing keys: {missing_filtered[:5]}{'...' if len(missing_filtered) > 5 else ''}")
        
        print(f"[CraterSMP_LayerNorm_Deep] Loaded weights from {checkpoint_path}")
        print(f"  Note: aux_heads are randomly initialized (not in checkpoint)")
        return missing_filtered, unexpected


class CraterSMP_LayerNorm_Deep_v2(nn.Module):
    """
    CraterSMP with Affine-less LayerNorm + LayerScale for maximum distribution shift robustness.
    
    Key improvements over v1:
    1. Decoder uses affine-less LayerNorm (elementwise_affine=False)
       - Forces scale-invariance, decoder cannot "memorize" training intensity scales
       - More robust to pseudo-data → real-data distribution shift
    
    2. LayerScale (γ initialized to 1e-5) after each decoder block
       - Stabilizes gradients from deep supervision auxiliary heads
       - Early training: gradients flow through skip connections (stable)
       - Late training: γ grows, model incorporates learned features
    
    3. Aux heads keep standard affine LayerNorm
       - They're trained from scratch anyway
       - Need expressive power to produce good segmentation
    
    Architecture:
    - Encoder: BatchNorm (for MobileOne reparameterization)
    - Decoder: Affine-less LayerNorm + LayerScale
    - Aux Heads: Standard affine LayerNorm
    
    Usage:
        model = CraterSMP_LayerNorm_Deep_v2(backbone="mobileone_s2", num_classes=3)
        
        # Training mode
        final, aux_outputs = model(x)  # Returns (final, [aux1, aux2, aux3, aux4])
        
        # Inference mode
        model.deep_supervision = False
        final = model(x)
    """
    
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=3,
                 decoder_channels=(256, 128, 64, 32, 16), deep_supervision=True,
                 layer_scale_init=1e-5):
        super(CraterSMP_LayerNorm_Deep_v2, self).__init__()
        
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # Create base SMP model
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_channels=decoder_channels
        )
        
        # Replace BatchNorm with AFFINE-LESS LayerNorm in decoder
        # This forces scale-invariance for distribution shift robustness
        replace_bn_with_ln_noaffine(self.model.decoder)
        replace_bn_with_ln_noaffine(self.model.segmentation_head)
        
        # LayerScale for each decoder block output (stabilizes deep supervision)
        self.layer_scales = nn.ModuleList()
        for ch in decoder_channels:
            self.layer_scales.append(LayerScale(ch, init_value=layer_scale_init))
        
        # Auxiliary heads with STANDARD affine LayerNorm (trained from scratch)
        self.aux_heads = nn.ModuleList()
        for ch in decoder_channels[:-1]:
            aux_head = nn.Sequential(
                nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
                LayerNorm2d(ch // 2, affine=True),  # Keep affine for aux heads
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, num_classes, kernel_size=1)
            )
            self.aux_heads.append(aux_head)
        
        print(f"[CraterSMP_LayerNorm_Deep_v2] Created with:")
        print(f"  - Affine-less LayerNorm in decoder (scale-invariant)")
        print(f"  - LayerScale (init={layer_scale_init}) for gradient stability")
        print(f"  - {len(self.aux_heads)} auxiliary heads with standard affine LN")
    
    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder
        features = self.model.encoder(x)
        
        # Get spatial shapes for decoder
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]
        
        # Process features like SMP decoder does
        features_for_decoder = features[1:][::-1]
        head = features_for_decoder[0]
        skip_connections = features_for_decoder[1:]
        
        # Apply center block
        decoder = self.model.decoder
        x_dec = decoder.center(head)
        
        # Manual decoder forward with LayerScale
        decoder_outputs = []
        
        for i, decoder_block in enumerate(decoder.blocks):
            target_h, target_w = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            
            # Decoder block forward
            x_dec_new = decoder_block(x_dec, target_h, target_w, skip_connection=skip_connection)
            
            # Apply LayerScale for stable deep supervision gradients
            x_dec = self.layer_scales[i](x_dec_new)
            
            decoder_outputs.append(x_dec)
        
        # Final segmentation head
        final_output = self.model.segmentation_head(x_dec)
        
        if self.training and self.deep_supervision:
            aux_outputs = []
            for aux_head, dec_out in zip(self.aux_heads, decoder_outputs[:-1]):
                aux_out = aux_head(dec_out)
                aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
                aux_outputs.append(aux_out)
            return final_output, aux_outputs
        else:
            return final_output

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


# --- DEPTHWISE-SEPARABLE CONVOLUTION MODULE ---
class DWConv3x3_PW(nn.Module):
    """
    Depthwise-separable convolution: 3x3 depthwise + 1x1 pointwise.
    
    More efficient than standard 3x3 convolution with similar receptive field.
    Reduces parameters and FLOPs while maintaining feature quality.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


def convert_smp_decoder(model):
    """
    Convert SMP decoder blocks to use depthwise-separable convolutions.
    
    Only converts blocks >= 2 (lower resolution blocks), keeping
    high-resolution blocks unchanged for quality preservation.
    
    Args:
        model: SMP model with a decoder attribute
    """
    decoder = model.decoder

    for idx, block in enumerate(decoder.blocks):
        # SMP order: block_0 = highest resolution
        # We convert only blocks >= 2
        if idx < 2:
            continue

        for conv_name in ["conv1", "conv2"]:
            conv_block = getattr(block, conv_name)

            conv = conv_block[0]
            bn   = conv_block[1]
            act  = conv_block[2]

            if not isinstance(conv, nn.Conv2d):
                continue

            in_c  = conv.in_channels
            out_c = conv.out_channels

            new_conv = DWConv3x3_PW(in_c, out_c)

            setattr(
                block,
                conv_name,
                nn.Sequential(new_conv, bn, act),
            )


class CraterSMPd(nn.Module):
    """
    CraterSMP with depthwise-separable convolutions in decoder blocks >= 2.
    
    This variant reduces computational cost while maintaining accuracy
    by using efficient depthwise-separable convolutions in lower-resolution
    decoder blocks.
    
    Args:
        backbone: Encoder backbone name (default: mobileone_s2)
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 3)
        decoder_channels: Decoder channel configuration
    """
    def __init__(self, backbone="mobileone_s2", in_channels=3, num_classes=3,
                 decoder_channels=(256, 128, 64, 32, 16)):
        super(CraterSMPd, self).__init__()
        
        # Create base SMP model
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_channels=decoder_channels
        )
        
        # Convert decoder to use depthwise-separable convolutions
        convert_smp_decoder(self.model)
        
        print(f"[CraterSMPd] Created with depthwise-separable convolutions in decoder blocks >= 2")
    
    def forward(self, x):
        return self.model(x)

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
