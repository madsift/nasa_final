import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from scipy.ndimage import distance_transform_edt


def compute_distance_map(mask):
    """
    Computes the distance map for a binary mask.
    mask: numpy array of shape (H, W)
    Returns: distance map of shape (H, W)
    """
    mask = mask.astype(np.uint8)
    
    # Distance to the nearest foreground pixel (for background pixels)
    if np.sum(mask) == 0:
        dist_to_fg = np.ones_like(mask, dtype=np.float32) * 50.0 
    else:
        dist_to_fg = distance_transform_edt(1 - mask)
    
    # Distance to the nearest background pixel (for foreground pixels)
    if np.sum(mask) == mask.size:
        dist_to_bg = np.ones_like(mask, dtype=np.float32) * 50.0
    else:
        dist_to_bg = distance_transform_edt(mask)
    
    # Combine: negative inside, positive outside
    phi = dist_to_fg - dist_to_bg
    
    # Scale-Aware Normalization
    h, w = mask.shape
    norm = np.sqrt(h*h + w*w)
    
    phi = np.clip(phi, -norm, norm) / norm
    phi = np.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return phi.astype(np.float32)


class CraterDataset3(Dataset):
    """
    Dataset for 3-channel input loaded from pre-computed .npz files.
    
    Data format (created by prepare_data.py):
    - .npz files contain: img (uint8), dem (uint16), grad (uint16)
    - .npy mask files contain: (3, H, W) uint8 [core, global, rim]
    
    The dem and grad channels are stored as uint16 (0-65535) representing [0, 1] range.
    """
    
    def __init__(self, images_dir, masks_dir, image_ids, transform=None, pad_to_multiple=32):
        """
        Args:
            images_dir: Path to images directory containing .npz files
            masks_dir: Path to masks directory containing .npy files
            image_ids: List of image IDs to load (without extension)
            transform: Albumentations transform
            pad_to_multiple: Pad image to multiple of this value
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_ids = image_ids 
        self.transform = transform
        self.pad_to_multiple = pad_to_multiple

    def __len__(self):
        return len(self.image_ids)

    def _pad_to_multiple(self, arr, multiple):
        """Pad array to nearest multiple with zeros."""
        if multiple <= 0:
            return arr
            
        h, w = arr.shape[:2]
        
        # Pad dimensions INDEPENDENTLY (support rectangular images)
        target_h = ((h + multiple - 1) // multiple) * multiple
        target_w = ((w + multiple - 1) // multiple) * multiple
        
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h == 0 and pad_w == 0:
            return arr
        
        # Pad at bottom and right
        if arr.ndim == 2:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        else:  # (H, W, C)
            padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        
        return padded

    def process_item(self, image_np, mask_np):
        """Convert numpy arrays to tensors with proper normalization."""
        # Compute distance map from rim mask (channel 2)
        rim_mask = mask_np[:, :, 2]
        rim_mask_bin = (rim_mask > 0.5).astype(np.uint8)
        dist_map = compute_distance_map(rim_mask_bin)
        
        # Append distance map -> (H, W, 4)
        mask_final = np.dstack([mask_np, dist_map])
            
        # Convert image to tensor: (H, W, 3) -> (3, H, W)
        img_t = np.transpose(image_np, (2, 0, 1))
        image_tensor = torch.from_numpy(img_t).float()
        
        # Normalize: [0, 1] -> [-1, 1]
        image_tensor = (image_tensor - 0.5) / 0.5

        # Convert mask to tensor: (H, W, 4) -> (4, H, W)
        mask_t = np.transpose(mask_final, (2, 0, 1))
        mask_tensor = torch.from_numpy(mask_t).float()
            
        return image_tensor, mask_tensor

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load pre-computed 3-channel data from .npz
        npz_path = self.images_dir / f"{img_id}.npz"
        data = np.load(str(npz_path))
        
        # Extract channels with proper normalization
        # img: uint8 [0, 255] -> float32 [0, 1]
        img = data['img'].astype(np.float32) / 255.0
        
        # dem: uint16 [0, 65535] -> float32 [0, 1]
        dem = data['dem'].astype(np.float32) / 65535.0
        
        # grad: uint16 [0, 65535] -> float32 [0, 1]
        grad = data['grad'].astype(np.float32) / 65535.0
        
        # Stack: (H, W, 3) [gray, dem, grad]
        image_3ch = np.dstack([img, dem, grad])
        
        # Load mask: (3, H, W) uint8 [0, 255]
        mask_path = self.masks_dir / f"{img_id}.npy"
        mask = np.load(str(mask_path))
        
        # Convert to float [0, 1] and transpose to (H, W, 3) for Albumentations
        mask = mask.astype(np.float32) / 255.0
        mask = np.transpose(mask, (1, 2, 0))
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image_3ch, mask=mask)
            image_3ch = augmented['image']
            mask = augmented['mask']
        
        # Pad to multiple
        image_3ch = self._pad_to_multiple(image_3ch, self.pad_to_multiple)
        mask = self._pad_to_multiple(mask, self.pad_to_multiple)
        
        return self.process_item(image_3ch, mask)


class GrayscaleOnlyGamma(ImageOnlyTransform):
    """
    Applies gamma correction ONLY to channel 0 (grayscale texture).
    Preserves channel 1 (DEM) and channel 2 (gradient) which have physical meanings.
    """
    def __init__(self, gamma_limit=(80, 120), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.gamma_limit = gamma_limit
    
    def apply(self, img, **params):
        if img.ndim != 3 or img.shape[2] < 1:
            return img
        
        # Random gamma value (normalize from percentage to actual gamma)
        gamma = random.uniform(self.gamma_limit[0] / 100.0, self.gamma_limit[1] / 100.0)
        
        img_out = img.copy()
        
        # Only apply gamma to channel 0 (grayscale texture)
        ch0 = np.clip(img[:, :, 0], 0, 1)
        img_out[:, :, 0] = np.power(ch0, gamma)
        
        return img_out

    def get_transform_init_args_names(self):
        return ("gamma_limit",)


class GrayscaleOnlyBrightnessContrast(ImageOnlyTransform):
    """
    Applies brightness/contrast adjustment ONLY to channel 0 (grayscale texture).
    Preserves channel 1 (DEM) and channel 2 (gradient).
    """
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
    
    def apply(self, img, **params):
        if img.ndim != 3 or img.shape[2] < 1:
            return img
        
        brightness = random.uniform(-self.brightness_limit, self.brightness_limit)
        contrast = random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)
        
        img_out = img.copy()
        
        # Only apply to channel 0
        ch0 = img[:, :, 0] * contrast + brightness
        img_out[:, :, 0] = np.clip(ch0, 0, 1)
        
        return img_out

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit")


def get_train_transforms(tile_size=None):
    """
    Training augmentations for 3-channel (gray, dem, grad) data.
    
    Args:
        tile_size: Tuple (width, height) or None. If rectangular (w != h),
                   RandomRotate90 is skipped to preserve aspect ratio.
    """
    # Determine if tiles are rectangular
    is_rect = False
    if tile_size is not None:
        if isinstance(tile_size, (tuple, list)) and len(tile_size) >= 2:
            is_rect = tile_size[0] != tile_size[1]
        # If single int, it's square (not rect)
    
    transforms = [
        # Geometric (Applies to Image + Mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    
    # Only add Rotate90 if tiles are square (preserves batch dimension consistency)
    if not is_rect:
        transforms.append(A.RandomRotate90(p=0.5))
    
    transforms.extend([
        # Affine - Conservative settings (replaces deprecated ShiftScaleRotate)
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5
        ),
        
        # Pixel Level (IMAGE ONLY - only affects channel 0)
        GrayscaleOnlyBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        GrayscaleOnlyGamma(gamma_limit=(80, 120), p=0.2),
    ])
    return A.Compose(transforms)


def get_valid_transforms():
    """Validation transforms (no augmentation, just passthrough)."""
    return A.Compose([])


def get_loaders(data_root, batch_size=8, split_ratio=0.8, seed=42, tile_size=None):
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Path to dataset root (contains images/ and masks/)
        batch_size: Batch size for dataloaders
        split_ratio: Train/val split ratio (not used, fixed val groups)
        seed: Random seed for shuffling
        tile_size: Tuple (width, height) of tiles. If rectangular, RandomRotate90 is skipped.
    
    Returns:
        train_loader, val_loader
    """
    random.seed(seed)
    data_path = Path(data_root)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    # Get all image IDs from .npz files
    all_files = [f.stem for f in images_dir.glob("*.npz")]
    
    # Group Split Logic (group by altitude + longitude)
    groups = {}
    for fname in all_files:
        parts = fname.split('_')
        # Group ID = altitude + longitude (first 2 parts)
        group_id = f"{parts[0]}_{parts[1]}"
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(fname)
    
    # Fixed validation groups
    '''
    fixed_val_groups =  [
        'altitude02_longitude07', 'altitude03_longitude14', 'altitude07_longitude11', 
        'altitude04_longitude02', 'altitude01_longitude14', 'altitude05_longitude15', 
        'altitude04_longitude08', 'altitude09_longitude14', 'altitude04_longitude14', 
        'altitude01_longitude12', 'altitude01_longitude10', 'altitude06_longitude05', 
        'altitude05_longitude16', 'altitude06_longitude13', 'altitude02_longitude10', 
        'altitude04_longitude13', 'altitude09_longitude08'
    ]
    '''
    fixed_val_groups = ['altitude01_longitude10', 'altitude01_longitude14',
        'altitude04_longitude13', 'altitude06_longitude13',
        'altitude09_longitude08']
    val_group_ids = [g for g in fixed_val_groups if g in groups]
    train_group_ids = [g for g in groups.keys() if g not in val_group_ids]
    
    print(f"Validation groups: {val_group_ids}")
    
    train_files = []
    for gid in train_group_ids:
        train_files.extend(groups[gid])
        
    val_files = []
    for gid in val_group_ids:
        val_files.extend(groups[gid])
    
    # Shuffle training files
    random.shuffle(train_files)

    print(f"Total Groups: {len(groups)}")
    print(f"Training:   {len(train_group_ids)} groups -> {len(train_files)} samples")
    print(f"Validation: {len(val_group_ids)} groups -> {len(val_files)} samples")
    
    # Create datasets with tile_size-aware transforms
    train_ds = CraterDataset3(images_dir, masks_dir, train_files, transform=get_train_transforms(tile_size=tile_size))
    val_ds = CraterDataset3(images_dir, masks_dir, val_files, transform=get_valid_transforms())

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test execution
    train_dl, val_dl = get_loaders("./train_tiles", batch_size=4, tile_size=(544, 416))
    
    # Verify shape
    img, mask = next(iter(train_dl))
    print(f"\nSUCCESS: Loader working with Albumentations v{A.__version__}")
    print(f"Img Shape: {img.shape} | Mask Shape: {mask.shape}")
    print(f"Img range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"Mask range: [{mask.min():.2f}, {mask.max():.2f}]")
