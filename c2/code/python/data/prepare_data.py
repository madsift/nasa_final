import os
import cv2
import numpy as np
import pandas as pd
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

# Import sfs_fast using relative import for package structure
from python.features.sfs_fast import compute_dem_and_gradient

OUTPUT_MIN_THICKNESS = 4  # Minimum rim thickness in OUTPUT image (after resize)
RIM_THICKNESS_RATIO = 0.05
TARGET_SIZE = 1024  # Applied to the larger dimension


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gt_csv', type=str, required=True)
    parser.add_argument('--target_size', type=int, default=TARGET_SIZE,
                        help='Target size for the larger dimension (smaller computed automatically)')
    parser.add_argument('--tilesize', type=int, nargs=2, default=[544, 416], metavar=('W', 'H'),
                        help='Tile size as (width, height). Default: 544 416. Must be multiples of 32.')
    return parser.parse_args()


def validate_tilesize(tile_w, tile_h):
    """Warn if tile dimensions are not multiples of 32."""
    issues = []
    if tile_w % 32 != 0:
        issues.append(f"tile width {tile_w} is not a multiple of 32")
    if tile_h % 32 != 0:
        issues.append(f"tile height {tile_h} is not a multiple of 32")
    
    if issues:
        warnings.warn(
            f"Tile size validation: {', '.join(issues)}. "
            "This may cause issues with neural network architectures that require 32-pixel alignment.",
            UserWarning
        )
        return False
    return True


def compute_resize_dims(orig_w, orig_h, target_size):
    """
    Compute output dimensions that preserve aspect ratio.
    The larger original dimension is scaled to target_size,
    and the smaller dimension is computed to preserve aspect ratio.
    
    Args:
        orig_w: original width
        orig_h: original height  
        target_size: size for the larger dimension
    
    Returns:
        (new_w, new_h): dimensions that preserve aspect ratio
        scale_factor: the scaling factor applied
    """
    if orig_w >= orig_h:
        # Width is larger -> scale width to target_size
        new_w = target_size
        scale_factor = target_size / orig_w
        new_h = int(round(orig_h * scale_factor))
    else:
        # Height is larger -> scale height to target_size
        new_h = target_size
        scale_factor = target_size / orig_h
        new_w = int(round(orig_w * scale_factor))
    
    return new_w, new_h, scale_factor


def pad_for_tiling(img, target_size):
    """
    Pad image to ensure both dimensions are divisible for perfect tiling after resize.
    
    For target_size=1440 with 2592x2048 images:
    - Pad height from 2048 to 2052 (add 4 pixels to bottom)
    - 2592/1.8=1440, 2052/1.8=1140 -> perfect 720x570 tiles (2x2 grid)
    
    Args:
        img: Input image (grayscale or color)
        target_size: Target size for larger dimension
        
    Returns:
        Padded image, (pad_bottom, pad_right) tuple
    """
    h, w = img.shape[:2]
    pad_bottom = 0
    pad_right = 0
    
    if target_size == 1440:
        # For 2592x2048 images: pad height to 2052 for perfect 1/1.8 scaling
        # 2052/1.8 = 1140, 2592/1.8 = 1440 -> tiles 720x570
        if w == 2592 and h == 2048:
            pad_bottom = 4  # 2048 + 4 = 2052
    
    if pad_bottom > 0 or pad_right > 0:
        if len(img.shape) == 2:
            img = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='edge')
        else:
            img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='edge')
    
    return img, (pad_bottom, pad_right)


def generate_masks(image_shape, craters, scale_factor):
    """Generate masks at original resolution.
    
    Args:
        image_shape: Original image shape
        craters: DataFrame of crater annotations
        scale_factor: The resize scale factor (used to compute min thickness)
    """
    h, w = image_shape[:2]
    mask_core = np.zeros((h, w), dtype=np.uint8)
    mask_global = np.zeros((h, w), dtype=np.uint8)
    mask_rim = np.zeros((h, w), dtype=np.uint8)

    # Compute min thickness at original resolution to achieve OUTPUT_MIN_THICKNESS after resize
    min_thickness = max(1, int(OUTPUT_MIN_THICKNESS / scale_factor))

    craters = craters.sort_values(by='ellipseSemimajor(px)', ascending=True)

    for _, crater in craters.iterrows():
        try:
            cx = float(crater['ellipseCenterX(px)'])
            cy = float(crater['ellipseCenterY(px)'])
            major = float(crater['ellipseSemimajor(px)'])
            minor = float(crater['ellipseSemiminor(px)'])
            angle = float(crater['ellipseRotation(deg)'])

            if np.isnan(cx) or np.isnan(cy) or major <= 0 or minor <= 0:
                continue

            center = (int(round(cx)), int(round(cy)))

            core_a, core_b = major * 0.3, minor * 0.3
            max_dim = max(core_a, core_b)
            if max_dim > 40:
                sf = 40.0 / max_dim
                core_a *= sf
                core_b *= sf

            axes_core = (max(1, int(round(core_a))), max(1, int(round(core_b))))
            cv2.ellipse(mask_core, center, axes_core, angle, 0, 360, 255, -1)

            # Rim Mask: Centered on the true rim (100%)
            r_a = max(1, int(round(major)))
            r_b = max(1, int(round(minor)))
            avg_d = major + minor
            thick = max(int(avg_d * RIM_THICKNESS_RATIO), min_thickness)
            half_thick = thick / 2.0
            cv2.ellipse(mask_rim, center, (r_a, r_b), angle, 0, 360, 255, thick)
            
            # Global Mask: Extends to inner edge of Rim (no dead space, no overlap)
            # Inner edge of rim = major - half_thickness
            g_a = max(1, int(round(major - half_thick)))
            g_b = max(1, int(round(minor - half_thick)))
            cv2.ellipse(mask_global, center, (g_a, g_b), angle, 0, 360, 255, -1)

        except:
            continue

    return mask_core, mask_global, mask_rim


def create_tiles(img, tile_w, tile_h):
    """
    Create 4 tiles (patches) from an image with 2x2 grid layout.
    
    For images where dimensions are not exactly 2*tile_size, tiles will overlap
    to cover the entire image. Tiles are extracted from corners.
    
    Args:
        img: Image array of shape (H, W) or (H, W, C)
        tile_w: Width of each tile
        tile_h: Height of each tile
    
    Returns:
        List of 4 tiles [top-left, top-right, bottom-left, bottom-right]
    """
    h, w = img.shape[:2]
    
    # If image is smaller than tile size, this will fail - caller should ensure proper sizing
    if h < tile_h or w < tile_w:
        raise ValueError(f"Image size ({w}x{h}) is smaller than tile size ({tile_w}x{tile_h})")
    
    # Calculate starting positions for the 4 tiles
    # Top-left: (0, 0)
    # Top-right: (w - tile_w, 0)
    # Bottom-left: (0, h - tile_h)
    # Bottom-right: (w - tile_w, h - tile_h)
    
    tiles = []
    positions = [
        (0, 0),              # p1: top-left
        (w - tile_w, 0),     # p2: top-right
        (0, h - tile_h),     # p3: bottom-left
        (w - tile_w, h - tile_h)  # p4: bottom-right
    ]
    
    for x, y in positions:
        if len(img.shape) == 2:
            tile = img[y:y+tile_h, x:x+tile_w]
        else:
            tile = img[y:y+tile_h, x:x+tile_w, :]
        tiles.append(tile)
    
    return tiles


def float32_to_uint16(arr):
    """
    Convert float32 array (normalized to [0, 1]) to uint16.
    This provides 65535 levels of precision, which is lossless for practical purposes.
    """
    return (np.clip(arr, 0, 1) * 65535).astype(np.uint16)


def save_compressed_npz(filepath, img_uint8, dem_uint16, grad_uint16):
    """
    Save the 3-channel data as a compressed .npz file.
    
    Storage format:
    - img: uint8 grayscale image (1 byte per pixel)
    - dem: uint16 DEM (2 bytes per pixel, represents [0,1] as [0,65535])
    - grad: uint16 gradient magnitude (2 bytes per pixel, represents [0,1] as [0,65535])
    
    This uses zlib compression (level 6 by default) which is lossless.
    Expected compression ratio: ~40-60% depending on content.
    """
    np.savez_compressed(filepath, img=img_uint8, dem=dem_uint16, grad=grad_uint16)


def process_dataset(args):
    input_root = Path(args.input_dir)
    out_img_dir = Path(args.output_dir) / "images"
    out_mask_dir = Path(args.output_dir) / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    target_size = args.target_size
    tile_w, tile_h = args.tilesize
    
    # Validate tile size
    validate_tilesize(tile_w, tile_h)
    
    print(f"Target size: {target_size}")
    print(f"Tile size: {tile_w}x{tile_h}")

    df = pd.read_csv(args.gt_csv)
    df = df.dropna(subset=['ellipseCenterX(px)', 'ellipseSemimajor(px)'])
    grouped = df.groupby('inputImage')

    for rel_raw, group in tqdm(grouped, desc="Processing images"):
        rel = str(rel_raw)
        if not rel.endswith('.png'):
            rel += ".png"

        img_path = input_root / "train" / rel
        if not img_path.exists():
            img_path = input_root / rel
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        
        # Pad image if needed for perfect tiling (e.g., 2048->2052 for 1440 target)
        img, (pad_bottom, pad_right) = pad_for_tiling(img, target_size)
        padded_h, padded_w = img.shape[:2]
        
        # Compute aspect-ratio-preserving dimensions from PADDED size
        # Larger dimension -> target_size, smaller computed automatically
        new_w, new_h, scale = compute_resize_dims(padded_w, padded_h, target_size)
        
        # Generate masks at original resolution (before padding)
        m_core, m_global, m_rim = generate_masks((orig_h, orig_w), group, scale)
        
        # Pad masks to match padded image
        if pad_bottom > 0 or pad_right > 0:
            m_core = np.pad(m_core, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
            m_global = np.pad(m_global, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
            m_rim = np.pad(m_rim, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)

        # Resize preserving aspect ratio
        img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        core_r = cv2.resize(m_core, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        global_r = cv2.resize(m_global, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        rim_r = cv2.resize(m_rim, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Compute DEM and gradient using sfs_fast (simultaneous computation)
        dem_f32, grad_f32 = compute_dem_and_gradient(img_r)
        
        # Convert to uint16 for efficient storage (lossless for [0,1] range)
        dem_u16 = float32_to_uint16(dem_f32)
        grad_u16 = float32_to_uint16(grad_f32)

        # Combine masks
        combined_mask = np.stack([core_r, global_r, rim_r], axis=0)

        save_id = rel.replace('/', '_').replace('.png', '')
        
        # Check if image is large enough for tiling
        if new_h >= tile_h and new_w >= tile_w:
            # Create tiles for image data (img, dem, grad) and masks
            img_tiles = create_tiles(img_r, tile_w, tile_h)
            dem_tiles = create_tiles(dem_u16, tile_w, tile_h)
            grad_tiles = create_tiles(grad_u16, tile_w, tile_h)
            
            # For masks, we need to transpose to (H, W, C) for tiling, then back
            mask_hwc = np.transpose(combined_mask, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            mask_tiles = create_tiles(mask_hwc, tile_w, tile_h)
            
            # Save each tile
            for i, (img_tile, dem_tile, grad_tile, mask_tile) in enumerate(
                zip(img_tiles, dem_tiles, grad_tiles, mask_tiles), start=1
            ):
                tile_id = f"{save_id}_p{i}"
                
                # Save compressed image data (img uint8, dem uint16, grad uint16)
                save_compressed_npz(
                    out_img_dir / f"{tile_id}.npz",
                    img_tile,
                    dem_tile,
                    grad_tile
                )
                
                # Save mask (transpose back to (C, H, W))
                mask_tile_chw = np.transpose(mask_tile, (2, 0, 1))
                np.save(out_mask_dir / f"{tile_id}.npy", mask_tile_chw)
        else:
            # Image is too small for tiling - save as single file
            warnings.warn(
                f"Image {save_id} ({new_w}x{new_h}) is smaller than tile size ({tile_w}x{tile_h}). "
                "Saving without tiling.",
                UserWarning
            )
            
            # Save compressed image data
            save_compressed_npz(
                out_img_dir / f"{save_id}.npz",
                img_r,
                dem_u16,
                grad_u16
            )
            
            # Save mask
            np.save(out_mask_dir / f"{save_id}.npy", combined_mask)


if __name__ == "__main__":
    process_dataset(parse_args())