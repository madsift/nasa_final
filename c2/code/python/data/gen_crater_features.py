#!/usr/bin/env python3
"""
Generate crater feature data for training ranking and classification models.

This script:
1. Runs inference on images using a trained model
2. Extracts crater candidates using ellipse fitting (loose thresholds to get many candidates)
3. Computes features for each candidate (geometry, rim_prob, polar, morph, stability, context, mahanti)
4. Computes GT labels (ranking score, xi2, class) for each candidate
5. Saves to CSV for training ranker/classifier

OUTPUT COLUMNS:
- Features: geometry_*, rim_prob_*, polar_*, morph_*, stab_*, ctx_*, mahanti_*, meta_*
- Labels: target_score (0-1), target_xi2 (dGA match quality), target_class (1-4 or -1)
- Meta: image_id, pred_x, pred_y

This data is suitable for BOTH ranking and classification tasks.

Usage:
    python gen_crater_features.py \\
        --checkpoint path/to/model.pth \\
        --raw_data_dir /path/to/images \\
        --processed_dir /path/to/npz \\
        --output_csv features.csv \\
        --target_size 768 \\
        --workers 8
"""

import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import project modules
from models_smp import CraterSMP, CraterSMP_3Ch_RAM
from datasets import get_loaders
from extractions import extract_craters_cv2_adaptive_selection_rescue
from eval3rect import compute_resize_dims, pad_to_square, strip_padding, tiled_inference
import sfs_fast
import ranking_features_multires as rf  # Single source for all feature/label functions

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_crater_snr(img: np.ndarray, ellipse: tuple, illumination_level: float) -> dict:
    """
    Compute local SNR-based confidence features for a crater candidate.
    
    For dark images, all predictions degrade together, so the model has no anchor.
    These features help the ranker learn:
    "In dark scenes, only very high SNR rims are trustworthy"
    
    Args:
        img: Grayscale image (uint8 or float)
        ellipse: (cx, cy, a, b, angle) - center, semi-axes, rotation angle
        illumination_level: Global mean brightness of the image [0-255]
        
    Returns:
        Dict with SNR features:
        - snr_raw: (mean_rim - mean_interior) / std_background
        - snr_x_illumination: snr_raw * (illumination_level / 255.0)
        - snr_illumination_level: Global image brightness normalized
    """
    cx, cy, a, b, angle = ellipse
    h, w = img.shape[:2]
    
    # Normalize image to float
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0
    
    # Create ellipse mask for interior (shrink by 20%)
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    shrink_factor = 0.8
    cv2.ellipse(interior_mask, 
                (int(cx), int(cy)), 
                (int(a * shrink_factor), int(b * shrink_factor)), 
                angle, 0, 360, 255, -1)
    
    # Create rim annulus mask (between 80% and 120% of ellipse)
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    expand_factor = 1.2
    cv2.ellipse(outer_mask, 
                (int(cx), int(cy)), 
                (int(a * expand_factor), int(b * expand_factor)), 
                angle, 0, 360, 255, -1)
    
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(inner_mask, 
                (int(cx), int(cy)), 
                (int(a * shrink_factor), int(b * shrink_factor)), 
                angle, 0, 360, 255, -1)
    
    rim_mask = outer_mask.astype(np.int16) - inner_mask.astype(np.int16)
    rim_mask = (rim_mask > 0).astype(np.uint8) * 255
    
    # Create background mask (region around crater but not inside it)
    bg_outer_mask = np.zeros((h, w), dtype=np.uint8)
    bg_factor = 1.5
    cv2.ellipse(bg_outer_mask, 
                (int(cx), int(cy)), 
                (int(a * bg_factor), int(b * bg_factor)), 
                angle, 0, 360, 255, -1)
    
    background_mask = bg_outer_mask.astype(np.int16) - outer_mask.astype(np.int16)
    background_mask = (background_mask > 0).astype(np.uint8) * 255
    
    # Compute statistics
    interior_pixels = img_f[interior_mask > 0]
    rim_pixels = img_f[rim_mask > 0]
    bg_pixels = img_f[background_mask > 0]
    
    # Handle edge cases
    if len(interior_pixels) < 10 or len(rim_pixels) < 10 or len(bg_pixels) < 10:
        return {
            'snr_raw': 0.0,
            'snr_x_illumination': 0.0,
            'snr_illumination_level': illumination_level / 255.0
        }
    
    mean_rim = float(np.mean(rim_pixels))
    mean_interior = float(np.mean(interior_pixels))
    std_bg = float(np.std(bg_pixels))
    
    # Avoid division by zero
    if std_bg < 1e-6:
        std_bg = 1e-6
    
    # SNR = contrast / noise
    snr_raw = (mean_rim - mean_interior) / std_bg
    
    # Illumination-scaled SNR
    illumination_norm = illumination_level / 255.0
    snr_x_illumination = snr_raw * illumination_norm
    
    return {
        'snr_raw': snr_raw,
        'snr_x_illumination': snr_x_illumination,
        'snr_illumination_level': illumination_norm
    }


def load_gt(df, img_id):
    """Load ground truth craters for an image, including class labels."""
    parts = img_id.split('_')
    path = f"{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}" if len(parts) >= 3 else img_id
    
    rows = df[df['inputImage'] == path]
    if rows.empty: 
        rows = df[df['inputImage'] == path + ".png"]
    
    return [{
        'x': r['ellipseCenterX(px)'], 
        'y': r['ellipseCenterY(px)'],
        'a': r['ellipseSemimajor(px)'], 
        'b': r['ellipseSemiminor(px)'],
        'angle': r['ellipseRotation(deg)'],
        'class': r['crater_classification'] if 'crater_classification' in r else 4
    } for _, r in rows.iterrows() if r['ellipseSemiminor(px)'] >= 5]


def process_candidate(cand, rim_prob_map, img_orig, gt_craters, img_id, h_orig, w_orig, 
                       illumination_level=128.0, fast_mode=False):
    """
    Process a single candidate crater - designed for threading.
    
    Always computes:
    - geometry features (with prop if available)
    - rim_prob features
    - polar features (cheap, high value)
    - morphology features
    - context features
    - SNR features (local signal-to-noise ratio)
    
    Optionally computes (fast_mode=False):
    - stability features (bootstrap, ~5 iterations)
    - mahanti features (gradient-based)
    
    Returns:
        dict: Feature dict with all features + labels, or None if failed
    """
    try:
        from skimage import measure
        
        # Create mask for regionprops
        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.ellipse(mask, (int(cand['x']), int(cand['y'])), 
                    (int(cand['a']), int(cand['b'])), cand['angle'], 0, 360, 1, -1)
        
        if cv2.countNonZero(mask) == 0:
            return None
        
        # RegionProps
        props = measure.regionprops(mask)
        if not props:
            return None
        prop = props[0]
        
        # Rim points using fast parametric sampling
        ellipse = (cand['x'], cand['y'], cand['a'], cand['b'], cand['angle'])
        rim_pts = rf.get_rim_points_for_candidate_fast(
            rim_prob_map, ellipse, n_samples=48, band_width=4, prob_thresh=0.2
        )
        
        if len(rim_pts) < 5:
            return None
        
        # Features using full extractor (includes polar features)
        feats = rf.extract_crater_features(
            rim_pts, ellipse, prop,
            rim_prob_map, img_orig, mask,
            fast_mode=fast_mode
        )
        
        # SNR features (local signal-to-noise ratio)
        snr_feats = compute_crater_snr(img_orig, ellipse, illumination_level)
        feats.update(snr_feats)
        
        # GT Labels (ranking score, xi2, class)
        score_label, xi2, cls_label = rf.compute_gt_ranking_label(cand, gt_craters)
        
        feats['target_score'] = score_label
        feats['target_xi2'] = xi2
        feats['target_class'] = cls_label
        feats['image_id'] = img_id
        feats['pred_x'] = cand['x']
        feats['pred_y'] = cand['y']
        
        return feats
        
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate crater features for ranking/classification training')
    
    # Model
    parser.add_argument('--backbone', type=str, default='mobileone_s2')
    parser.add_argument('--model_type', type=str, default='CraterSMP', 
                        choices=['CraterSMP', 'CraterSMP_3Ch_RAM'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--im_dim', type=int, default=3)
    
    # Data
    parser.add_argument('--raw_data_dir', type=str, required=True, help='Directory with original images')
    parser.add_argument('--processed_dir', type=str, required=True, help='Directory with preprocessed .npz files')
    parser.add_argument('--gt_csv', type=str, default='train-gt.csv', help='Ground truth CSV')
    parser.add_argument('--subset', type=str, default='val', choices=['val', 'train', 'all'])
    parser.add_argument('--limit_images', type=int, default=0, help='Limit number of images (0 = all)')
    
    # Output
    parser.add_argument('--output_csv', type=str, default='crater_features.csv')
    
    # Inference
    parser.add_argument('--target_size', type=int, default=768, 
                        help='Target size for resizing (longest dimension). Image is then padded to multiple of 32.')
    parser.add_argument('--tile_size', type=str, default='0',
                        help='Tile size (WxH or single value). Default "0" = disabled. E.g., "672x544" for 2x2 grid on 1296x1024.')
    
    # Feature extraction
    parser.add_argument('--fast_mode', action='store_true', default=False,
                        help='Skip expensive features (stability, mahanti). Use for ranking only.')
    parser.add_argument('--workers', type=int, default=8, 
                        help='Number of threads for parallel feature extraction')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility (stability features use np.random)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size = args.target_size
    
    # Fixed extraction parameters (loose to get many candidates for training)
    THRESHOLD = 0.3
    RIM_THRESH = 0.2
    MIN_CONFIDENCE = 0.1
    MIN_CONFIDENCE_MIN = 0.1
    SCALE_STEPS = 30
    
    print(f"=== Crater Feature Generator ===")
    print(f"Target size: {target_size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fast mode: {args.fast_mode} (skip stability/mahanti)")
    print(f"Workers: {args.workers}")
    print(f"Extraction: threshold={THRESHOLD}, rim_thresh={RIM_THRESH}, min_confidence={MIN_CONFIDENCE}")
    
    # Load Model
    print(f"\nLoading model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Detect output channels
    head_key = [k for k in state_dict.keys() if 'segmentation_head.0.weight' in k]
    out_channels = state_dict[head_key[0]].shape[0] if head_key else 3
    print(f"Detected {out_channels} output channels")
    
    if args.model_type == 'CraterSMP_3Ch_RAM':
        model = CraterSMP_3Ch_RAM(backbone=args.backbone, num_classes=out_channels).to(device)
    else:
        model = CraterSMP(backbone=args.backbone, in_channels=args.im_dim, num_classes=out_channels).to(device)
    
    model.load_state_dict(state_dict)
    
    # Reparameterize if available
    encoder = model.model.encoder
    if hasattr(encoder, 'reparameterize'):
        encoder.reparameterize()
        print("✅ Reparameterized encoder")
    else:
        for module in model.modules():
            if hasattr(module, 'reparameterize'):
                module.reparameterize()
    
    model.eval()
    
    # Load Data
    print(f"\nLoading dataset from {args.processed_dir}...")
    train_loader, val_loader = get_loaders(args.processed_dir, im_dim=args.im_dim, batch_size=1)
    
    if args.subset == 'val':
        dataset = val_loader.dataset
    elif args.subset == 'train':
        dataset = train_loader.dataset
    else:
        # Combine train + val
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    
    # Load GT
    gt_df = pd.read_csv(args.gt_csv)
    
    # Process images
    limit = len(dataset) if args.limit_images == 0 else min(args.limit_images, len(dataset))
    print(f"\nProcessing {limit} images...")
    
    all_results = []
    thread_pool = ThreadPoolExecutor(max_workers=args.workers)
    
    try:
        for idx in tqdm(range(limit)):
            # Get image ID
            if hasattr(dataset, 'image_ids'):
                img_id = dataset.image_ids[idx]
            else:
                # ConcatDataset fallback
                img_id = dataset.datasets[0].image_ids[idx] if idx < len(dataset.datasets[0]) else \
                         dataset.datasets[1].image_ids[idx - len(dataset.datasets[0])]
            
            # Load original image
            # raw_data_dir should point to the base folder containing altitude/longitude structure
            parts = img_id.split('_')
            raw_path = Path(args.raw_data_dir) / parts[0] / parts[1] / ("_".join(parts[2:]) + ".png")
            
            if not raw_path.exists():
                raw_path = Path(args.raw_data_dir) / (img_id + ".png")
                if not raw_path.exists():
                    continue
            
            gt_craters = load_gt(gt_df, img_id)
            
            img_orig = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            if img_orig is None:
                continue
            
            h_orig, w_orig = img_orig.shape
            
            # Resize (aspect-preserving)
            resized_w, resized_h, scale = compute_resize_dims(w_orig, h_orig, target_size)
            img_resized = cv2.resize(img_orig, (resized_w, resized_h))
            
            # Prepare input (3-channel)
            dem, grad = sfs_fast.compute_dem_and_gradient(img_resized)
            img_norm = img_resized.astype(np.float32) / 255.0
            image_3ch = np.dstack([img_norm, dem, grad])
            
            # Determine if using tiled inference
            use_tiled = False
            tile_size_hw = None
            overlap_hw = None
            
            if args.tile_size and str(args.tile_size) != '0':
                use_tiled = True
                # Parse tile size "W,H" or "WxH" (Standard image notation)
                t_str = str(args.tile_size)
                if ',' in t_str or 'x' in t_str:
                    sep = ',' if ',' in t_str else 'x'
                    parts = t_str.split(sep)
                    tile_w = int(parts[0].strip())
                    tile_h = int(parts[1].strip()) if len(parts) > 1 else tile_w
                else:
                    tile_w = int(t_str)
                    tile_h = tile_w
                tile_size_hw = (tile_h, tile_w)  # (H, W)
                
                # Default overlap: heuristic 1/4, or specific values for known tile sizes
                ov_h = tile_h // 4
                ov_w = tile_w // 4
                
                # Optimized overlap for common tile sizes
                if tile_w == 672 and tile_h == 544:
                    ov_w, ov_h = 24, 32
                elif tile_w == 544 and tile_h == 544:
                    ov_w, ov_h = 32, 32
                elif tile_w == 544 and tile_h == 416:
                    ov_w, ov_h = 32, 11
                overlap_hw = (ov_h, ov_w)
            
            if use_tiled:
                # Tiled inference: no padding needed, tiler handles it
                img_norm_3ch = np.transpose(image_3ch, (2, 0, 1))
                input_tensor = torch.from_numpy(img_norm_3ch).float().unsqueeze(0).to(device)
                input_tensor = (input_tensor - 0.5) / 0.5
                
                # Tiled inference
                preds_padded = tiled_inference(
                    model, input_tensor,
                    tile_size=tile_size_hw,
                    overlap=overlap_hw,
                    device=device,
                    use_onnx=False,
                    use_tukey=True
                )
            else:
                # Full image inference: pad to square
                image_padded, _, _ = pad_to_square(image_3ch, target_size)
                
                # Build input tensor
                input_tensor = torch.from_numpy(np.transpose(image_padded, (2,0,1))).float().unsqueeze(0).to(device)
                input_tensor = (input_tensor - 0.5) / 0.5
                
                # Inference
                with torch.no_grad():
                    preds_padded = torch.sigmoid(model(input_tensor))[0].cpu().numpy()
            
            # Strip padding to get back to resized dimensions
            preds_resized = strip_padding(preds_padded, resized_h, resized_w)
            
            # Resize predictions to original
            preds_full = np.zeros((out_channels, h_orig, w_orig), dtype=np.float32)
            for c in range(out_channels):
                preds_full[c] = cv2.resize(preds_resized[c], (w_orig, h_orig))
            
            # Extract candidates (loose parameters to get many candidates)
            extracted = extract_craters_cv2_adaptive_selection_rescue(
                preds_full,
                threshold=THRESHOLD,
                rim_thresh=RIM_THRESH,
                min_confidence=MIN_CONFIDENCE,
                min_confidence_min=MIN_CONFIDENCE_MIN,
                scale_steps=SCALE_STEPS,
                image_shape=(h_orig, w_orig),
                fit_method='cv2',
                downsample_factor=1.0,
                skip_prop_image=False,
                min_semi_axis=5,                    # Looser: was 15
                max_size_ratio=0.75,
                return_all_candidates=True,
                n_samples_scale=48,
                n_samples_conf=72,
                angular_completeness_thresh=0.20    # Looser: was 0.35
            )
            
            if not extracted:
                continue
            
            rim_prob_map = preds_full[2] if preds_full.shape[0] > 2 else preds_full[0]
            
            # Compute illumination level for SNR features
            illumination_level = float(np.mean(img_orig))
            
            # Parallel feature extraction
            futures = [
                thread_pool.submit(
                    process_candidate,
                    cand, rim_prob_map, img_orig, gt_craters, img_id, h_orig, w_orig,
                    illumination_level=illumination_level,
                    fast_mode=args.fast_mode
                )
                for cand in extracted
            ]
            
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    if result is not None:
                        all_results.append(result)
                except Exception:
                    pass
                    
    finally:
        thread_pool.shutdown(wait=False)
    
    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    
    # Summary
    n_total = len(df)
    n_matched = (df['target_score'] > 0).sum() if n_total > 0 else 0
    n_unmatched = n_total - n_matched
    n_images = df['image_id'].nunique() if n_total > 0 else 0
    
    print(f"\n=== Summary ===")
    print(f"Total candidates: {n_total}")
    print(f"Images processed: {n_images}")
    print(f"Candidates per image: {n_total/max(1,n_images):.1f}")
    
    # Match distribution
    print(f"\n=== Match Distribution ===")
    print(f"Matched (TPs):   {n_matched:5d} ({100*n_matched/max(1,n_total):.1f}%)")
    print(f"Unmatched (FPs): {n_unmatched:5d} ({100*n_unmatched/max(1,n_total):.1f}%)")
    
    # Match quality breakdown
    if n_matched > 0:
        high_quality = (df['target_score'] >= 0.9).sum()
        medium_quality = ((df['target_score'] >= 0.7) & (df['target_score'] < 0.9)).sum()
        low_quality = ((df['target_score'] > 0) & (df['target_score'] < 0.7)).sum()
        print(f"\nMatch quality:")
        print(f"  High (>=0.9):   {high_quality:5d} ({100*high_quality/max(1,n_matched):.1f}% of matches)")
        print(f"  Medium (0.7-0.9): {medium_quality:3d} ({100*medium_quality/max(1,n_matched):.1f}% of matches)")
        print(f"  Low (<0.7):     {low_quality:5d} ({100*low_quality/max(1,n_matched):.1f}% of matches)")
    
    print(f"\nSaved to: {args.output_csv}")
    
    # Expected vs Actual GT comparison
    if n_total > 0 and n_images > 0:
        print(f"\n=== Expected vs Actual ===")
        total_expected = 0
        low_recall_images = []
        
        for img_id in df['image_id'].unique():
            # Count expected GT craters
            parts = img_id.split('_')
            path = f"{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}" if len(parts) >= 3 else img_id
            gt_rows = gt_df[gt_df['inputImage'] == path]
            if gt_rows.empty:
                gt_rows = gt_df[gt_df['inputImage'] == path + ".png"]
            expected = len(gt_rows[gt_rows['ellipseSemiminor(px)'] >= 5])
            
            # Count actual
            img_df = df[df['image_id'] == img_id]
            matched = (img_df['target_score'] > 0).sum()
            
            total_expected += expected
            recall = matched / max(1, expected)
            
            if recall < 0.3:  # Track low recall images
                low_recall_images.append((img_id, expected, matched, recall))
        
        overall_recall = n_matched / max(1, total_expected)
        avg_expected = total_expected / n_images
        avg_candidates = n_total / n_images
        
        print(f"Expected GT craters: {total_expected} ({avg_expected:.1f}/img)")
        print(f"Detected candidates: {n_total} ({avg_candidates:.1f}/img)")
        print(f"Matched craters: {n_matched}")
        print(f"Overall recall: {100*overall_recall:.1f}%")
        
        if overall_recall < 0.5:
            print(f"\n⚠️ WARNING: Low recall ({100*overall_recall:.1f}%) suggests:")
            print(f"   - Model predictions may be poor at target_size={target_size}")
            print(f"   - Try a model trained at this resolution")
            print(f"   - Or increase target_size to match training resolution")
        
        if low_recall_images and len(low_recall_images) <= 5:
            print(f"\nLow recall images (<30%):")
            for img_id, exp, matched, recall in low_recall_images:
                print(f"  {img_id}: {matched}/{exp} = {100*recall:.1f}%")
    
    # Feature info
    if n_total > 0:
        feature_cols = [c for c in df.columns if not c.startswith('target_') and c not in ['image_id', 'pred_x', 'pred_y']]
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"  - Geometry: {len([c for c in feature_cols if c.startswith('geometry_')])}")
        print(f"  - Rim prob: {len([c for c in feature_cols if c.startswith('rim_prob_')])}")
        print(f"  - Polar: {len([c for c in feature_cols if c.startswith('polar_')])}")
        print(f"  - Morphology: {len([c for c in feature_cols if c.startswith('morph_')])}")
        print(f"  - Stability: {len([c for c in feature_cols if c.startswith('stab_')])}")
        print(f"  - Context: {len([c for c in feature_cols if c.startswith('ctx_')])}")
        print(f"  - Mahanti: {len([c for c in feature_cols if c.startswith('mahanti_')])}")
        print(f"  - SNR: {len([c for c in feature_cols if c.startswith('snr_')])}")
        print(f"  - Meta: {len([c for c in feature_cols if c.startswith('meta_')])}")


if __name__ == '__main__':
    main()

