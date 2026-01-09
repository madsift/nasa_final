import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import math
import time  # Added for timing
import psutil  # Memory tracking (process RSS)
from tqdm import tqdm
from pathlib import Path
from python.features import sfs_fast
import yaml
from collections import defaultdict

# Set random seeds and deterministic mode for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# TIMING INSTRUMENTATION
# ==========================================
class TimingStats:
    """Track timing statistics for each processing stage."""
    def __init__(self):
        self.timings = defaultdict(list)
        self.stage_start = {}
    
    def start(self, stage_name):
        """Start timing a stage."""
        self.stage_start[stage_name] = time.perf_counter()
    
    def stop(self, stage_name):
        """Stop timing a stage and record duration."""
        if stage_name in self.stage_start:
            duration = time.perf_counter() - self.stage_start[stage_name]
            self.timings[stage_name].append(duration)
            del self.stage_start[stage_name]
            return duration
        return 0.0
    
    def add(self, stage_name, duration):
        """Directly add a timing measurement."""
        self.timings[stage_name].append(duration)
    
    def report(self):
        """Generate a detailed timing report."""
        print("\n" + "="*70)
        print("                    DETAILED TIMING REPORT")
        print("="*70)
        
        # Separate one-time and per-image timings
        one_time_stages = ['config_loading', 'model_weights_loading', 'ranker_loading', 'onnx_session_init']
        # Exclude 'total_per_image' from breakdown - it's the wrapper, not a stage
        exclude_stages = one_time_stages + ['total_per_image']
        per_image_stages = [k for k in self.timings.keys() if k not in exclude_stages]
        
        # One-time costs
        print("\n[ONE-TIME INITIALIZATION COSTS]")
        print("-" * 50)
        total_init = 0.0
        for stage in one_time_stages:
            if stage in self.timings:
                t = sum(self.timings[stage])
                total_init += t
                print(f"  {stage:30s}: {t*1000:10.2f} ms")
        print(f"  {'TOTAL INIT':30s}: {total_init*1000:10.2f} ms")
        
        # Get actual total time from wrapper (the real end-to-end time)
        actual_total_times = self.timings.get('total_per_image', [])
        actual_total = sum(actual_total_times) if actual_total_times else 0.0
        n_images = len(actual_total_times) if actual_total_times else 1
        
        # Per-image costs
        print("\n[PER-IMAGE PROCESSING COSTS]")
        print("-" * 70)
        print(f"  {'Stage':30s} | {'Total(s)':>10s} | {'Mean(ms)':>10s} | {'Max(ms)':>10s} | {'%':>6s}")
        print("-" * 70)
        
        stage_totals = {}
        for stage in per_image_stages:
            times = self.timings[stage]
            if times:
                total = sum(times)
                stage_totals[stage] = total
        
        # Sort by total time (descending)
        sorted_stages = sorted(stage_totals.items(), key=lambda x: x[1], reverse=True)
        
        tracked_total = sum(stage_totals.values())
        
        for stage, total in sorted_stages:
            times = self.timings[stage]
            mean_ms = (total / len(times)) * 1000
            max_ms = max(times) * 1000
            # Percentage of actual end-to-end time
            pct = (total / actual_total * 100) if actual_total > 0 else 0
            print(f"  {stage:30s} | {total:10.2f} | {mean_ms:10.2f} | {max_ms:10.2f} | {pct:5.1f}%")
        
        # Show untracked time (overhead not captured by stage timings)
        untracked = actual_total - tracked_total
        if untracked > 0.1:  # Only show if significant (>100ms total)
            untracked_pct = (untracked / actual_total * 100) if actual_total > 0 else 0
            print(f"  {'(untracked overhead)':30s} | {untracked:10.2f} | {(untracked/n_images)*1000:10.2f} | {'':>10s} | {untracked_pct:5.1f}%")
        
        print("-" * 70)
        mean_per_image = (actual_total / n_images) * 1000 if n_images > 0 else 0
        max_per_image = max(actual_total_times) * 1000 if actual_total_times else 0
        print(f"  {'TOTAL (end-to-end)':30s} | {actual_total:10.2f} | {mean_per_image:10.2f} | {max_per_image:10.2f} | 100.0%")
        print(f"\n  Images processed: {n_images}")
        print(f"  Average time per image: {mean_per_image:.2f} ms ({mean_per_image/1000:.2f} s)")
        print("="*70 + "\n")
        
        return stage_totals

# Optional ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# --- Import Project Modules ---
from python.models.models_smp import CraterSMP, CraterSMP_3Ch_RAM, CraterSMP_GroupNorm
from skimage import measure, segmentation, morphology
from skimage.measure import CircleModel
from scipy import ndimage as ndi
from python.evaluation.extractions import extract_craters_cv2_adaptive_selection_rescue, extract_craters_cv2_adaptive_selection_polar




# ==========================================
# CONSTANTS (FIXED)
# ==========================================
FIXED_SCALE = 1.2 #1.25
FIXED_THRESH = 0.80 #0.8 #0.65 #RAM
XI_2_THRESH = 13.277
NN_PIX_ERR_RATIO = 0.07 
FORCE_CIRCULAR = False
TARGET_SIZE = 1024

def get_filtered_files(raw_data_dir):
    fixed_val_groups_2 = ['altitude01_longitude10', 'altitude01_longitude14',
           'altitude04_longitude13', 'altitude06_longitude13',
           'altitude09_longitude08']
    
    # Recursive glob to find all pngs
    test_path = Path(raw_data_dir)
    all_files = sorted(list(test_path.rglob("*.png")))
    
    filtered_path = []
    # Filter based on groups
    for item in fixed_val_groups_2:
         # Check if both 'altitudeXX' and 'longitudeYY' appear in the path
         parts = item.split('_')
         filtered_path.extend([p for p in all_files if parts[0] in str(p) and parts[1] in str(p) and 'truth' not in str(p)])
    
    return sorted(list(set(filtered_path))) # Ensure unique


# ==========================================
# TILED INFERENCE (Memory Efficient)
# ==========================================
def create_gaussian_weight(tile_size, sigma_ratio=0.25):
    """Create Gaussian weight map for smooth tile blending."""
    if isinstance(tile_size, (tuple, list)):
        th, tw = tile_size
    else:
        th, tw = tile_size, tile_size
        
    center_y, center_x = th // 2, tw // 2
    sigma_y, sigma_x = th * sigma_ratio, tw * sigma_ratio
    
    y, x = np.ogrid[:th, :tw]
    # Elliptical gaussian
    weight = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))
    return weight.astype(np.float32)


def create_tukey_weight(tile_size, overlap):
    """
    Create Tukey window for seamless tile blending.
    
    Tukey window has:
    - Flat center region at weight=1.0 (no signal loss)
    - Smooth cosine taper only in the overlap regions
    
    This eliminates visible seams while preserving full signal in tile centers.
    """
    from scipy.signal.windows import tukey
    
    if isinstance(tile_size, (tuple, list)):
        th, tw = tile_size
    else:
        th, tw = tile_size, tile_size
        
    if isinstance(overlap, (tuple, list)):
        oh, ow = overlap
    else:
        oh, ow = overlap, overlap
    
    # Alpha = fraction of window that is tapered (both ends combined)
    # 2*overlap/size because overlap exists on BOTH sides
    alpha_h = min(2.0 * oh / th, 1.0) if th > 0 else 0.0
    alpha_w = min(2.0 * ow / tw, 1.0) if tw > 0 else 0.0
    
    tukey_y = tukey(th, alpha=alpha_h)
    tukey_x = tukey(tw, alpha=alpha_w)
    weight = np.outer(tukey_y, tukey_x)
    
    return weight.astype(np.float32)


def tiled_inference(model_or_session, x_full, tile_size=256, overlap=64, device='cpu', use_onnx=False, use_tukey=False):
    """
    Perform inference on large image using overlapping tiles.
    
    Args:
        tile_size: int or tuple (H, W)
        overlap: int or tuple (H_overlap, W_overlap)
    """
    if isinstance(x_full, torch.Tensor):
        x_full = x_full.cpu().numpy()
    
    # Parse tile dimensions
    if isinstance(tile_size, (tuple, list)):
        th, tw = tile_size
    else:
        th, tw = tile_size, tile_size
        
    # Parse overlap dimensions
    if isinstance(overlap, (tuple, list)):
        oh, ow = overlap
    else:
        oh, ow = overlap, overlap
        
    _, C_in, H, W = x_full.shape
    
    stride_h = th - oh
    stride_w = tw - ow
    
    # Determine output channels by running a small test
    test_tile = x_full[:, :, :th, :tw]
    # Pad test tile if image is smaller than tile (shouldn't happen with proper logic but safety)
    if test_tile.shape[2] < th or test_tile.shape[3] < tw:
        # Just create a dummy
        test_tile = np.zeros((1, C_in, th, tw), dtype=x_full.dtype)
        
    if use_onnx:
        input_name = model_or_session.get_inputs()[0].name
        test_out = model_or_session.run(None, {input_name: test_tile})[0]
    else:
        with torch.no_grad():
            test_out = model_or_session(torch.from_numpy(test_tile).to(device)).cpu().numpy()
    C_out = test_out.shape[1]
    
    # Initialize output and weight accumulator
    output = np.zeros((C_out, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    # Weight map for blending (Tukey or Gaussian)
    if use_tukey:
        blend_weight = create_tukey_weight((th, tw), (oh, ow))
        print(f"[DEBUG] Using Tukey window for tile blending (alpha_h={2*oh/th:.2f}, alpha_w={2*ow/tw:.2f})")
    else:
        blend_weight = create_gaussian_weight((th, tw))
    
    # Calculate grid (Optimized to avoid redundant pass at end if alignment is imperfect but sufficient)
    # Use ceil division based on valid starting positions
    if stride_h > 0:
        n_rows = (H - th + stride_h - 1) // stride_h + 1
    else:
        n_rows = 1
        
    if stride_w > 0:
        n_cols = (W - tw + stride_w - 1) // stride_w + 1
    else:
        n_cols = 1
        
    print(f"[DEBUG] Tiling Image {W}x{H} | Tile H={th}, W={tw} | Overlap H={oh}, W={ow} | Grid: {n_rows} rows x {n_cols} cols = {n_rows*n_cols} tiles")

    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate tile coordinates
            y1 = min(row * stride_h, H - th)
            x1 = min(col * stride_w, W - tw)
            
            # Ensure valid (if image smaller than tile or negative iter)
            y1 = max(0, y1)
            x1 = max(0, x1)
            
            y2 = y1 + th
            x2 = x1 + tw
            
            # Extract tile
            tile = x_full[:, :, y1:y2, x1:x2]
            
            # Additional safety: verify tile shape matches expectation (e.g. edge cases)
            if tile.shape[2] != th or tile.shape[3] != tw:
                # Pad to required size (bottom-right)
                pad_h = th - tile.shape[2]
                pad_w = tw - tile.shape[3]
                tile = np.pad(tile, ((0,0), (0,0), (0, pad_h), (0, pad_w)))
            
            # Inference
            if use_onnx:
                tile_out = model_or_session.run(None, {input_name: tile})[0]
                tile_pred = 1.0 / (1.0 + np.exp(-tile_out[0]))  # Sigmoid
            else:
                with torch.no_grad():
                    tile_tensor = torch.from_numpy(tile).to(device)
                    tile_out = torch.sigmoid(model_or_session(tile_tensor))
                    tile_pred = tile_out[0].cpu().numpy()
            
            # Crop back if we padded
            real_h = y2 - y1
            real_w = x2 - x1
            pred_crop = tile_pred[:, :real_h, :real_w]
            weight_crop = blend_weight[:real_h, :real_w]
            
            # Accumulate
            for c in range(C_out):
                output[c, y1:y2, x1:x2] += pred_crop[c] * weight_crop
            weight_sum[y1:y2, x1:x2] += weight_crop
    
    # Normalize by weight sum
    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(C_out):
        output[c] /= weight_sum
    
    return output


def get_overlap_mask(target_h, target_w, tile_size, overlap):
    """
    Generate a mask showing overlap regions in the target inference space.
    Returns: numpy array (H, W) where 1=overlap, 0=single coverage
    """
    if isinstance(tile_size, (tuple, list)):
        th, tw = tile_size
    else:
        th, tw = tile_size, tile_size
        
    if isinstance(overlap, (tuple, list)):
        oh, ow = overlap
    else:
        oh, ow = overlap, overlap
        
    stride_h = th - oh
    stride_w = tw - ow
    
    # Simple accumulation mask
    coverage = np.zeros((target_h, target_w), dtype=np.int32)
    
    # Logic matching tiled_inference
    if stride_h > 0:
        n_rows = (target_h - th + stride_h - 1) // stride_h + 1
    else:
        n_rows = 1
        
    if stride_w > 0:
        n_cols = (target_w - tw + stride_w - 1) // stride_w + 1
    else:
        n_cols = 1
        
    for row in range(n_rows):
        for col in range(n_cols):
             y1 = min(row * stride_h, target_h - th)
             x1 = min(col * stride_w, target_w - tw)
             y1 = max(0, y1); x1 = max(0, x1)
             coverage[y1:y1+th, x1:x1+tw] += 1
             
    # Overlap is where count > 1
    return (coverage > 1).astype(np.uint8)

# ==========================================
# RECTANGULAR IMAGE UTILS
# ==========================================
def compute_resize_dims(orig_w, orig_h, target_size):
    """
    Compute output dimensions that preserve aspect ratio.
    The larger original dimension is scaled to target_size,
    and the smaller dimension is computed to preserve aspect ratio.
    
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

def pad_to_square(img, target_size):
    """
    Pad image to target_size x target_size with zeros at bottom/right.
    
    Returns:
        padded_img: padded image
        pad_h: padding added to height
        pad_w: padding added to width
    """
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w
    
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
    
    if img.ndim == 2:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    return padded, pad_h, pad_w

def strip_padding(pred, resized_h, resized_w):
    """
    Remove padding from prediction, keeping only the valid region.
    
    Args:
        pred: prediction array of shape (C, H, W)
        resized_h: height before padding
        resized_w: width before padding
    
    Returns:
        Stripped prediction of shape (C, resized_h, resized_w)
    """
    return pred[:, :resized_h, :resized_w]

# ==========================================
# 2. SCORING UTILS
# ==========================================
def calcYmat(a, b, phi):
    unit_1 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    unit_2 = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
    unit_3 = np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])
    return unit_1 @ unit_2 @ unit_3

def dGA_calc(crater_A, crater_B):
    phi_A = crater_A['angle'] * math.pi / 180.0
    phi_B = crater_B['angle'] * math.pi / 180.0
    
    Yi = calcYmat(crater_A['a'], crater_A['b'], phi_A)
    Yj = calcYmat(crater_B['a'], crater_B['b'], phi_B)
    
    yi = np.array([[crater_A['x']], [crater_A['y']]])
    yj = np.array([[crater_B['x']], [crater_B['y']]])
    
    det_Sum = np.linalg.det(Yi + Yj)
    if det_Sum < 1e-9: return math.pi, 9999.0
    
    multiplicand = 4 * np.sqrt(np.linalg.det(Yi) * np.linalg.det(Yj)) / det_Sum
    diff = yi - yj
    inv_Sum = np.linalg.inv(Yi + Yj)
    exponent = (-0.5 * diff.T @ Yi @ inv_Sum @ Yj @ diff)[0,0]
    
    val = multiplicand * np.exp(exponent)
    dGA = np.arccos(min(1.0, max(-1.0, val)))
    
    min_axis = min(crater_A['a'], crater_A['b'])
    ref_sig = 0.85 / np.sqrt(crater_A['a'] * crater_A['b']) * (NN_PIX_ERR_RATIO * min_axis)
    if ref_sig < 1e-9: ref_sig = 1e-6
    
    xi_2 = (dGA ** 2) / (ref_sig ** 2)
    return dGA, xi_2

def compute_full_scores(gt_craters, pred_craters):
    if len(gt_craters) == 0: return 0.0, 0.0
    if len(pred_craters) == 0: return 0.0, 0.0
    
    matches_dga = []
    for p in pred_craters: p['matched'] = False
    
    for t in gt_craters:
        best_p = None
        best_xi = float('inf')
        best_dGA = math.pi
        
        for p in pred_craters:
            if p['matched']: continue
            # Scorer.py matching logic
            rA = min(t['a'], t['b'])
            rB = min(p['a'], p['b'])
            
            # Size check
            if rA > 1.5 * rB or rB > 1.5 * rA: continue
            
            # Distance check (Stricter)
            r = min(rA, rB)
            if abs(t['x'] - p['x']) > r: continue
            if abs(t['y'] - p['y']) > r: continue
            
            d, xi = dGA_calc(t, p)
            if xi < best_xi:
                best_xi = xi
                best_dGA = d
                best_p = p
        
        if best_xi < XI_2_THRESH and best_p is not None:
            best_p['matched'] = True
            matches_dga.append(1.0 - (best_dGA / math.pi))

    if len(matches_dga) == 0:
        nasa_score = 0.0
    else:
        avg_quality = sum(matches_dga) / len(pred_craters) 
        recall_factor = len(matches_dga) / min(10, len(gt_craters))
        if recall_factor > 1.0: recall_factor = 1.0
        nasa_score = avg_quality * recall_factor

    hits = 0
    for t in gt_craters:
        for p in pred_craters:
            r = min(t['a'], t['b'])
            if math.sqrt((t['x']-p['x'])**2 + (t['y']-p['y'])**2) < r:
                hits += 1; break
    iou_recall = hits / len(gt_craters)
    
    return nasa_score, iou_recall


def compute_full_scores_with_details(gt_craters, pred_craters):
    """
    Same as compute_full_scores but also returns per-crater match status and individual scores.
    
    Returns:
        nasa_score: Overall NASA score
        iou_recall: IoU-based recall
        per_crater_details: List of dicts with 'matched' (0/1) and 'score' (dGA-based) for each pred
    """
    if len(gt_craters) == 0 or len(pred_craters) == 0:
        # Return zeros for all predictions
        per_crater = [{'matched': 0, 'score': 0.0} for _ in pred_craters]
        return 0.0, 0.0, per_crater
    
    # Initialize per-crater tracking
    per_crater = [{'matched': 0, 'score': 0.0} for _ in pred_craters]
    matches_dga = []
    
    for p in pred_craters:
        p['_matched'] = False
        p['_match_score'] = 0.0
    
    for t in gt_craters:
        best_p = None
        best_p_idx = -1
        best_xi = float('inf')
        best_dGA = math.pi
        
        for idx, p in enumerate(pred_craters):
            if p['_matched']:
                continue
            # Scorer.py matching logic
            rA = min(t['a'], t['b'])
            rB = min(p['a'], p['b'])
            
            # Size check
            if rA > 1.5 * rB or rB > 1.5 * rA:
                continue
            
            # Distance check (Stricter)
            r = min(rA, rB)
            if abs(t['x'] - p['x']) > r:
                continue
            if abs(t['y'] - p['y']) > r:
                continue
            
            d, xi = dGA_calc(t, p)
            if xi < best_xi:
                best_xi = xi
                best_dGA = d
                best_p = p
                best_p_idx = idx
        
        if best_xi < XI_2_THRESH and best_p is not None:
            best_p['_matched'] = True
            match_score = 1.0 - (best_dGA / math.pi)
            best_p['_match_score'] = match_score
            matches_dga.append(match_score)
            per_crater[best_p_idx]['matched'] = 1
            per_crater[best_p_idx]['score'] = match_score

    if len(matches_dga) == 0:
        nasa_score = 0.0
    else:
        avg_quality = sum(matches_dga) / len(pred_craters)
        recall_factor = len(matches_dga) / min(10, len(gt_craters))
        if recall_factor > 1.0:
            recall_factor = 1.0
        nasa_score = avg_quality * recall_factor

    hits = 0
    for t in gt_craters:
        for p in pred_craters:
            r = min(t['a'], t['b'])
            if math.sqrt((t['x'] - p['x'])**2 + (t['y'] - p['y'])**2) < r:
                hits += 1
                break
    iou_recall = hits / len(gt_craters)
    
    # Clean up temporary keys
    for p in pred_craters:
        p.pop('_matched', None)
        p.pop('_match_score', None)
    
    return nasa_score, iou_recall, per_crater

def load_gt(df, img_id):
    parts = img_id.split('_')
    path = f"{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}" if len(parts) >= 3 else img_id
    rows = df[df['inputImage'] == path]
    if rows.empty: rows = df[df['inputImage'] == path + ".png"]
    
    return [{
        'x': r['ellipseCenterX(px)'], 'y': r['ellipseCenterY(px)'],
        'a': r['ellipseSemimajor(px)'], 'b': r['ellipseSemiminor(px)'],
        'angle': r['ellipseRotation(deg)']
    } for _, r in rows.iterrows() if r['ellipseSemiminor(px)'] >= 5]


def compute_gold_scores_for_candidates(candidates, gt_craters):
    """
    ORACLE RANKING: Compute actual GT match quality for each candidate.
    
    For each prediction, find the best matching GT and compute its dGA score.
    This gives the theoretical upper bound - what a perfect ranker would achieve.
    
    Args:
        candidates: List of prediction dicts with x, y, a, b, angle
        gt_craters: List of GT crater dicts
        
    Returns:
        List of (candidate, gold_score) tuples, where gold_score is:
        - 1.0 - (dGA / pi) for valid matches (higher = better match)
        - 0.0 for non-matches
    """
    results = []
    
    for p in candidates:
        best_score = 0.0
        
        for t in gt_craters:
            # Scorer.py matching logic (same as compute_full_scores)
            rA = min(t['a'], t['b'])
            rB = min(p['a'], p['b'])
            
            # Size check
            if rA > 1.5 * rB or rB > 1.5 * rA:
                continue
            
            # Distance check
            r = min(rA, rB)
            if abs(t['x'] - p['x']) > r:
                continue
            if abs(t['y'] - p['y']) > r:
                continue
            
            dGA, xi2 = dGA_calc(t, p)
            
            if xi2 < XI_2_THRESH:
                score = 1.0 - (dGA / math.pi)
                best_score = max(best_score, score)
        
        results.append((p, best_score))
    
    return results

# ==========================================
# 3. VISUALIZATION UTILS
# ==========================================
def visualize_prediction(img_orig, preds_full, extracted, gt_craters=None, save_path=None, tile_config=None, score=None, save_binary=False):
    """
    Visualization of Ellipses and 3-Channel Masks (Core, Global, Rim).
    Layout: 2x2 Grid
    [Ellipses] [Core]
    [Global]   [Rim]
    """
    # 1. Main Canvas with Ellipses
    viz_main = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    
    # Feature: Overlay Inference Overlaps (if provided)
    if tile_config:
        try:
            # 1. Generate overlap mask in Target Domain
            ov_mask_target = get_overlap_mask(
                tile_config['target_h'], tile_config['target_w'], 
                tile_config['tile_size'], tile_config['overlap']
            )
            
            # 2. Crop to valid region (removing padding)
            # The inference was done on padded image, but real content is only at top-left
            # We need to map this mask to original image size
            # Step A: Crop to resized dims
            rh, rw = tile_config['resized_h'], tile_config['resized_w']
            ov_mask_cropped = ov_mask_target[:rh, :rw]
            
            # Step B: Resize to Original Dims (Nearest Neighbor to keep sharp boundaries)
            h_orig, w_orig = img_orig.shape[:2]
            ov_mask_orig = cv2.resize(ov_mask_cropped, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # 3. Create Overlay (Solid Red)
            # Make a red overlay where mask is 1
            overlay = viz_main.copy()
            # Set overlap regions to pure red
            overlay[ov_mask_orig > 0] = [0, 0, 255]
            
            # 4. Blend with alpha 0.3 (30% Red, 70% Original)
            cv2.addWeighted(overlay, 0.3, viz_main, 0.7, 0, viz_main)
            
            # Debug print
            print(f"Viz: Generated overlap mask with Max={ov_mask_orig.max()}, Pixels={np.count_nonzero(ov_mask_orig)}")
            cv2.putText(viz_main, "Overlap Zones", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        except Exception as e:
            print(f"Viz Error: {e}")
    else:
        print("Viz: No tile_config provided.")

    # Score display
    score_str = f"Score: {score:.4f}" if score is not None else "Score: N/A"
    cv2.putText(viz_main, score_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 0), 4)

    # Draw GT if available (RED)
    if gt_craters:
        for c in gt_craters: 
            cv2.ellipse(viz_main, (int(c['x']), int(c['y'])), (int(c['a']), int(c['b'])), c['angle'], 0, 360, (0,0,255), 2)
    
    # Draw Predictions
    for c in extracted:
        # Determine Color
        if 'matched' in c:
            col = (0,255,0) if c['matched'] else (255,0,0) # Green if matched, Blue if FP
        else:
            col = (255,255,0) # Cyan for Test Mode
            
        cv2.ellipse(viz_main, (int(c['x']), int(c['y'])), (int(c['a']), int(c['b'])), c['angle'], 0, 360, col, 2)

        if c.get('_source') == 'rim_large':
             cv2.putText(viz_main, "L", (int(c['x']), int(c['y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
    cv2.putText(viz_main, "Overlay", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # 2. Visualize Channels from preds_full (3, H, W)
    def to_heatmap(plane, label, cmap=cv2.COLORMAP_JET, is_distance=False):
        # Normalize to 0-255 for visualization regardless of magnitude (Auto-gain)
        p_min, p_max = plane.min(), plane.max()
        if p_max - p_min < 1e-6:
            norm = np.zeros_like(plane)
            norm_uint8 = np.zeros_like(plane, dtype=np.uint8)
        else:
            norm = (plane - p_min) / (p_max - p_min)
            norm_uint8 = (norm * 255).astype(np.uint8)
            
        colored = cv2.applyColorMap(norm_uint8, cmap)
        
        # Feature: Add Isocontours for Distance Maps to visualize gradient/topology
        if is_distance and p_max > 0.05: # Only if there is some signal
            # Draw lines at 25%, 50%, 75% of the range
            for level in [64, 128, 192]: 
                # Find contours at this level
                _, thresh = cv2.threshold(norm_uint8, level, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # Draw white contours, thin
                cv2.drawContours(colored, contours, -1, (255, 255, 255), 1)
        
        # Add Label + Scale Info to check if it's truly blank or just weak
        stats = f"{label} (Max: {p_max:.2f})"
        cv2.putText(colored, stats, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return colored
        
    if preds_full.shape[0] == 3:
        # Channel 0: Core (Inferno)
        viz_core = to_heatmap(preds_full[0], "Core", cv2.COLORMAP_INFERNO)
        # Channel 1: Global/Dist
        viz_global = to_heatmap(preds_full[1], "Global/Dist", cv2.COLORMAP_VIRIDIS, is_distance=True)
        # Channel 2: Rim
        viz_rim = to_heatmap(preds_full[2], "Rim", cv2.COLORMAP_HOT)
    else:
        # 2 Channels: Global, Rim
        # Blank Core
        viz_core = to_heatmap(np.zeros_like(preds_full[0]), "No Core", cv2.COLORMAP_INFERNO)
        # Channel 0 is Global, 1 is Rim
        viz_global = to_heatmap(preds_full[0], "Global/Dist", cv2.COLORMAP_VIRIDIS, is_distance=True)
        viz_rim = to_heatmap(preds_full[1], "Rim", cv2.COLORMAP_HOT)
    
    # Grid Assembly
    top_row = np.hstack([viz_main, viz_core])
    bot_row = np.hstack([viz_global, viz_rim])
    full_viz = np.vstack([top_row, bot_row])
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), full_viz)
        
        if save_binary:
           # Save the raw prediction channels (Core, Global, Rim)
           # preds_full is (C, H, W) in [0,1] range
           
           def save_channel(data, name):
               # Scale to 0-255
               d_u8 = (data * 255).astype(np.uint8)
               path = p.parent / (p.stem + f"_{name}.png")
               cv2.imwrite(str(path), d_u8)
               
           if preds_full.shape[0] == 3:
               save_channel(preds_full[0], "core")
               save_channel(preds_full[1], "global")
               save_channel(preds_full[2], "rim")
           elif preds_full.shape[0] == 2:
               # Assuming Global, Rim
               save_channel(preds_full[0], "global")
               save_channel(preds_full[1], "rim")


# ==========================================
# 4. LOCAL SNR FEATURES (for dark images)
# ==========================================
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
        - illumination_level: Global image brightness
    """
    cx, cy, a, b, angle = ellipse
    h, w = img.shape[:2]
    
    # Normalize image to float
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0
    
    # Create ellipse mask for interior
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    # Shrink by 20% for interior (avoid rim)
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
            'illumination_level': illumination_level / 255.0
        }
    
    mean_rim = float(np.mean(rim_pixels))
    mean_interior = float(np.mean(interior_pixels))
    std_bg = float(np.std(bg_pixels))
    
    # Avoid division by zero
    if std_bg < 1e-6:
        std_bg = 1e-6
    
    # SNR = contrast / noise
    snr_raw = (mean_rim - mean_interior) / std_bg
    
    # Illumination-scaled SNR (helps ranker weight SNR by lighting quality)
    illumination_norm = illumination_level / 255.0
    snr_x_illumination = snr_raw * illumination_norm
    
    return {
        'snr_raw': snr_raw,
        'snr_x_illumination': snr_x_illumination,
        'illumination_level': illumination_norm
    }


# ==========================================
# 5. MAIN EVALUATION
# ==========================================
def evaluate(args):
    device = torch.device(args.device)
    print(f"--- 3-CHANNEL EVALUATION (Rectangular, Aspect-Preserving) ---")
    
    # Start memory tracking (using psutil for accurate process memory)
    process = psutil.Process()
    memory_samples = []  # Track memory after each inference (MB)
    baseline_mem = process.memory_info().rss / 1e6  # Baseline before inference

    # Initialize timing tracker
    timing = TimingStats()
    
    # Load Config (Overrides globals and defaults)
    timing.start('config_loading')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}: {config}")
    timing.stop('config_loading')
    
    fixed_thresh = config.get('FIXED_THRESH', FIXED_THRESH)
    target_size = config.get('TARGET_SIZE', TARGET_SIZE)
    rim_thresh_val = config.get('rim_thresh', 0.25)
    min_confidence_val = config.get('min_confidence', 0.97)
    min_confidence_min_val = config.get('min_confidence_min', 0.90)
    scale_steps_val = config.get('scale_steps', 15)  # Reduced from 30 for speed
    downsample_factor_val = config.get('downsample_factor', 1.0)  # 1.0=full res (default), 1.5/2.0=faster
    skip_prop_image_val = config.get('skip_prop_image', False)  # True=faster, avoids prop.image
    angular_completeness_thresh_val = config.get('angular_completeness_thresh', 0.35)  # Lower for shadow/terminator
    use_grad_conf = config.get('use_grad_conf', False)  # Gradient-weighted confidence
    

    
    # ONNX Runtime session (if specified)
    ort_session = None
    model = None
    
    if args.onnx_model:
        # Use ONNX Runtime - no PyTorch model needed
        if not HAS_ONNX:
            print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
            return
        print(f"Loading ONNX model: {args.onnx_model}")
        
        timing.start('onnx_session_init')
        # Memory optimization settings (30-50% RSS reduction)
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False      # Disable arena over-allocation
        sess_options.enable_mem_pattern = False        # Disable large pre-allocations
        sess_options.intra_op_num_threads = 1          # Single-threaded
        sess_options.inter_op_num_threads = 1
        
        ort_session = ort.InferenceSession(
            args.onnx_model, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        timing.stop('onnx_session_init')
        print(f"ONNX Runtime initialized (memory optimized): {ort_session.get_providers()}")
    else:
        # Use PyTorch model
        if not args.checkpoint:
            print("ERROR: Either --checkpoint or --onnx_model must be specified")
            return
        
        timing.start('model_weights_loading')
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt['state_dict']
        
        # Check output classes from weights
        head_weight_key = [k for k in state_dict.keys() if 'segmentation_head.0.weight' in k]
        if head_weight_key:
            out_channels = state_dict[head_weight_key[0]].shape[0]
            print(f"Detected {out_channels} output channels from checkpoint.")
        else:
            print("Could not detect channels from weights, defaulting to 3")
            out_channels = 3

        if args.model_type == 'CraterSMP_3Ch_RAM':
            print(f"Instantiating CraterSMP_3Ch_RAM with backbone={args.backbone}, out_channels={out_channels}")
            model = CraterSMP_3Ch_RAM(backbone=args.backbone, num_classes=out_channels).to(device)
            model.load_state_dict(ckpt['state_dict'])
        elif args.model_type == 'CraterSMP_GroupNorm':
            print(f"Instantiating CraterSMP_GroupNorm with backbone={args.backbone}, in_channels={args.im_dim}, num_classes={out_channels}")
            model = CraterSMP_GroupNorm(backbone=args.backbone, in_channels=args.im_dim, num_classes=out_channels).to(device)
            model.load_from_batchnorm_checkpoint(args.checkpoint)
            encoder = model.model.encoder
            if hasattr(encoder, 'reparameterize'):
                encoder.reparameterize()
                print("✅ Called encoder.reparameterize() on CraterSMP_GroupNorm")
        else:
            print(f"Instantiating CraterSMP with in_channels={args.im_dim}, num_classes={out_channels}")
            model = CraterSMP(backbone=args.backbone, in_channels=args.im_dim, num_classes=out_channels).to(device)          
            model.load_state_dict(ckpt['state_dict'])
            encoder = model.model.encoder
            if hasattr(encoder, 'reparameterize'):
                encoder.reparameterize()
                print("✅ Called encoder.reparameterize()")
            else:
                # Fallback traversal
                print("⚠️ Direct encoder method not found, searching modules...")
                found = False
                for module in model.modules():
                    if hasattr(module, 'reparameterize'):
                        module.reparameterize()
                        found = True
                if found:
                    print("✅ Called reparameterize() on sub-modules.")
        
        model.eval()
        timing.stop('model_weights_loading')
        
        # Note: convert_fx is only for QAT training workflow
        # For FP32 inference, just use the model directly
        
        if 'epoch' in ckpt.keys():
            print('model loaded at-> ', ckpt['epoch'])
    
    # 2. Setup Data
    df = pd.read_csv(args.gt_csv) if not args.is_test else None
    val_loader = None  # Will be loaded only if needed (validation mode)
    
    # List to store results
    detailed_results = []
    solution_rows = []
    tested_image_paths = set()
    num_per_images = []
    
    # List to track starving images
    low_count_records = []
    
    # Determine file list based on mode
    all_files = None
    use_csv_mode = False
    

    
    if args.is_test and not use_csv_mode:
        print(f"--- TEST MODE (No GT, Recursive Glob) ---")
        # Recursive glob for test mode
        test_path = Path(args.raw_data_dir)
        all_files = sorted([p for p in test_path.rglob("*.png") if 'truth' not in str(p)])
        print(f"Found {len(all_files)} images in {args.raw_data_dir}")
    elif not use_csv_mode:
        # Filtered validation mode using raw_data_dir
        all_files = get_filtered_files(args.raw_data_dir)
        if len(all_files) == 0:
            # Fallback: grab all files (excluding 'truth')
            print(f"WARNING: Filter returned 0 images, falling back to all files")
            test_path = Path(args.raw_data_dir)
            all_files = sorted([p for p in test_path.rglob("*.png") if 'truth' not in str(p)])
        print(f"Filtered to {len(all_files)} images from fixed validation groups in {args.raw_data_dir}")
    
    # Apply --limit_images (always)
    if args.limit_images > 0 and all_files is not None:
        all_files = all_files[:args.limit_images]
        print(f"Limited to {len(all_files)} images (--limit_images={args.limit_images})")
    
    # Determine iteration limit
    if all_files is not None:
        limit = len(all_files)
    else:
        limit = len(all_files)
    

    
    # List to track starving images
    low_count_records = []

    for idx in tqdm(range(limit)):
        timing.start('total_per_image')
        start_time = time.time() # START TIME
        
        if all_files is not None:
            # CSV or Test Mode: Load directly from file path list
            p = all_files[idx]
            # Generate img_id from path
            # Try to extract relative path structure for ID
            try:
                # Assume structure: .../altitude##/longitude##/filename.png
                parts_list = p.parts
                # Find altitude## and longitude## parts
                altitude_part = None
                longitude_part = None
                for i, part in enumerate(parts_list):
                    if part.startswith('altitude'):
                        altitude_part = part
                        if i + 1 < len(parts_list) and parts_list[i+1].startswith('longitude'):
                            longitude_part = parts_list[i+1]
                        break
                
                if altitude_part and longitude_part:
                    img_id = f"{altitude_part}_{longitude_part}_{p.stem}"
                else:
                    img_id = p.stem
            except:
                img_id = p.stem
        
        if not p.exists(): continue
        
        timing.start('image_loading')
        img_orig = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        timing.stop('image_loading')
        #img_orig = apply_gamma(img_orig, gamma=1.5)
        #img_orig = apply_clahe(img_orig, clip_limit=1.0, tile_grid_size=(12, 12))

        h_orig, w_orig = img_orig.shape
        
        # --- Preprocessing (Aspect-Preserving Resize + Pad) ---
        # Step 1: Resize with aspect ratio preserved (longest dim = TARGET_SIZE)
        timing.start('resizing')
        resized_w, resized_h, scale = compute_resize_dims(w_orig, h_orig, target_size)
        img_resized = cv2.resize(img_orig, (resized_w, resized_h))
        timing.stop('resizing')
        
        timing.stop('resizing')
        
        use_tiled = False
        if args.tile_size and str(args.tile_size) != '0':
            use_tiled = True
            
        # Note: padding is done inside the if-else branches below
        grad = None  # Initialize for gradient-weighted confidence (may not be computed for im_dim==1)
        if args.im_dim == 1:
            # Pad then normalize for 1-channel
            if not use_tiled:
                
                img_padded, pad_h, pad_w = pad_to_square(img_resized, target_size)
                x = torch.from_numpy(img_padded.astype(np.float32)/255.0).float()[None,None].to(device)
                if args.use_instance_norm:
                    mean = x.mean()
                    std = x.std()
                    x = (x - mean) / torch.clamp(std, min=0.1)
                else:
                    x = (x - 0.5) / 0.5
            else:
                # Tiled: No padding needed (tiler handles it), enables clean 2x2 grid
                img_norm = img_resized.astype(np.float32)/255.0
                x = torch.from_numpy(img_norm).float()[None,None].to(device)
                if args.use_instance_norm:
                    mean = x.mean()
                    std = x.std()
                    x = (x - mean) / torch.clamp(std, min=0.1)
                else:
                    x = (x - 0.5) / 0.5
                pad_h, pad_w = 0, 0
                
        elif args.im_dim == 2:
            # Image + Gradient (No DEM)
            timing.start('gradient_creation')
            grad = sfs_fast.compute_gradient_map(img_resized)
            timing.stop('gradient_creation')
            
            # Stack 2 channels
            img_norm = img_resized.astype(np.float32) / 255.0
            image_2ch = np.dstack([img_norm, grad])
            
            # Pad the 2-channel image
            if not use_tiled:
                image_2ch_padded, pad_h, pad_w = pad_to_square(image_2ch, target_size)
                image_2ch_padded = np.transpose(image_2ch_padded, (2, 0, 1))
                image_tensor = torch.from_numpy(image_2ch_padded).float().unsqueeze(0).to(device)
            else:
                image_2ch_t = np.transpose(image_2ch, (2, 0, 1))
                image_tensor = torch.from_numpy(image_2ch_t).float().unsqueeze(0).to(device)
                pad_h, pad_w = 0, 0
                
            if args.use_instance_norm:
                mean = image_tensor.mean()
                std = image_tensor.std()
                x = (image_tensor - mean) / torch.clamp(std, min=0.1)
            else:
                x = (image_tensor - 0.5) / 0.5
        else:
            # Compute SFS on UNPADDED image (matching training pipeline)
            timing.start('dem_gradient_creation')
            dem, grad = sfs_fast.compute_dem_and_gradient(img_resized)
            timing.stop('dem_gradient_creation')
            
            # Stack 3 channels
            img_norm = img_resized.astype(np.float32) / 255.0
            image_3ch = np.dstack([img_norm, dem, grad])
            
            # Pad the 3-channel image (matching training: SFS first, then pad)
            if not use_tiled:
                image_3ch_padded, pad_h, pad_w = pad_to_square(image_3ch, target_size)
                image_3ch_padded = np.transpose(image_3ch_padded, (2, 0, 1))
                image_tensor = torch.from_numpy(image_3ch_padded).float().unsqueeze(0).to(device)
            else:
                image_3ch_t = np.transpose(image_3ch, (2, 0, 1))
                image_tensor = torch.from_numpy(image_3ch_t).float().unsqueeze(0).to(device)
                pad_h, pad_w = 0, 0
                
            if args.use_instance_norm:
                mean = image_tensor.mean()
                std = image_tensor.std()
                x = (image_tensor - mean) / torch.clamp(std, min=0.1)
            else:
                x = (image_tensor - 0.5) / 0.5

        # --- Inference ---
        # --- Inference ---
        timing.start('model_inference')
        
        # Parse tile_size from args or config
        # Parse tile_size from args or config
        # use_tiled is already determining preprocessing, but we need variables map
        tile_size_hw = None
        overlap_hw = None
        
        if args.tile_size and str(args.tile_size) != '0':
            use_tiled = True
            # Parse tile size "W,H" or "WxH" (Standard image notation)
            t_str = str(args.tile_size)
            if ',' in t_str or 'x' in t_str:
                sep = ',' if ',' in t_str else 'x'
                parts = t_str.split(sep)
                tile_w = int(parts[0].strip()) # Width first
                tile_h = int(parts[1].strip()) if len(parts) > 1 else tile_w # Height second
            else:
                tile_w = int(t_str)
                tile_h = tile_w
            tile_size_hw = (tile_h, tile_w) # Pass as (H, W) to inference
            
            # Default overlap: optimized for 1296x1024 with 672x544 tiles (approx 2x2 grid)
            # Width: 1296, Tile W: 672. Stride W needed = 1296 - 672 = 624. Overlap W = 672 - 624 = 48.
            # Height: 1024, Tile H: 544. Stride H needed = 1024 - 544 = 480. Overlap H = 544 - 480 = 64.
            
            # Heuristic: 1/8 overlap or specific defaults if matching standard sizes
            ov_h = tile_h // 4
            ov_w = tile_w // 4
            
            # Adjust overlap to try and fit edges nicely if possible
            if tile_w == 672 and tile_h == 544:
                 ov_w = 24
                 ov_h = 32
            if tile_w == 544 and tile_h == 544:
                 ov_w = 32
                 ov_h = 32
            if tile_w == 544 and tile_h == 416:
                 ov_w = 32
                 ov_h = 11
            overlap_hw = (ov_h, ov_w)
            
        if use_tiled:
            # TILED INFERENCE (memory efficient)
            if ort_session is not None:
                preds_padded = tiled_inference(ort_session, x.numpy(), 
                                               tile_size=tile_size_hw, 
                                               overlap=overlap_hw,
                                               use_onnx=True,
                                               use_tukey=True)
            else:
                preds_padded = tiled_inference(model, x, 
                                               tile_size=tile_size_hw,
                                               overlap=overlap_hw,
                                               device=device,
                                               use_onnx=False,
                                               use_tukey=True)
        elif ort_session is not None:
            # ONNX Runtime inference (full image)
            ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
            ort_out = ort_session.run(None, ort_inputs)[0]
            # Apply sigmoid (ONNX model output is raw logits)
            preds_padded = 1.0 / (1.0 + np.exp(-ort_out[0]))  # Sigmoid
        else:
            # PyTorch inference (full image)
            with torch.no_grad():
                preds_padded = torch.sigmoid(model(x))[0].cpu().numpy()
        timing.stop('model_inference')
        
        # Track memory after inference (using psutil for accurate RSS)
        current_rss = process.memory_info().rss / 1e6  # MB
        memory_samples.append(current_rss)
 
        # --- Strip Padding from Predictions ---
        preds_resized = strip_padding(preds_padded, resized_h, resized_w)
            
        # --- Resize Back to Original Resolution ---
        n_ch = preds_resized.shape[0]
        preds_full = np.zeros((n_ch, h_orig, w_orig), dtype=np.float32)
        for c in range(n_ch):
            preds_full[c] = cv2.resize(preds_resized[c], (w_orig, h_orig))
        
        # Resize gradient to original resolution for gradient-weighted confidence
        if use_grad_conf and args.im_dim >= 2 and grad is not None:
            grad_full = cv2.resize(grad, (w_orig, h_orig))
        else:
            grad_full = None
        
        # 3. Extract Craters (Rim Only Logic - Hough + Refine)
        timing.start('ellipse_extraction')
        '''
        extracted_small = extract_craters_cv2_adaptive_selection_polar(preds_full, threshold=fixed_thresh, rim_thresh=rim_thresh_val,
                                  ecc_floor=0.15,
                                  scale_min=1.0,
                                  scale_max=1.4,
                                  scale_steps=scale_steps_val,
                                  min_confidence=min_confidence_val, #97 90  best
                                  min_confidence_min=min_confidence_min_val  , nms_filter=False,
                                  enable_rescue=False, enable_completeness=False, 
                                  fit_method='polar',
                                  min_semi_axis=40,
                                  max_size_ratio=0.6,
                                  require_fully_visible=True,
                                  image_shape=(h_orig, w_orig),
                                  downsample_factor=downsample_factor_val,
                                  skip_prop_image=skip_prop_image_val, polar_num_bins=180,
                                  polar_min_support=0.35, polar_refine_center=True,
                                  polar_max_center_shift=15.0, no_fallback=False)
        '''
        extracted_small = extract_craters_cv2_adaptive_selection_rescue(preds_full, threshold=fixed_thresh, rim_thresh=rim_thresh_val,
                                  ecc_floor=0.15,
                                  scale_min=1.0,
                                  scale_max=1.4,
                                  scale_steps=scale_steps_val,
                                  min_confidence=min_confidence_val,
                                  min_confidence_min=min_confidence_min_val,
                                  fit_method='cv2',
                                  min_semi_axis=40,
                                  max_size_ratio=0.6,
                                  require_fully_visible=True,
                                  image_shape=(h_orig, w_orig),
                                  downsample_factor=downsample_factor_val,
                                  skip_prop_image=skip_prop_image_val,
                                  angular_completeness_thresh=angular_completeness_thresh_val,
                                  use_grad_conf=use_grad_conf,
                                  grad_mag=grad_full
                                  )
        
        timing.stop('ellipse_extraction')
        extracted = extracted_small

        if args.limit_craters > 0:
            extracted = extracted[:args.limit_craters]
       
        num_per_images.append(len(extracted))

        # --- LOG LOW COUNT IMAGES ---
        if len(extracted) < 10:
             low_count_records.append({
                 "ImageID": img_id,
                 "Count": len(extracted)
             })

        # Collect for CSVs
        parts = img_id.split('_')
        path_str = f"{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}" if len(parts) >= 3 else img_id
        tested_image_paths.add(path_str)
        
        end_time = time.time() # END TIME
        duration = end_time - start_time
        timing.stop('total_per_image')
        
        # --- Scoring (Only if NOT test) ---
        per_crater_details = None
        if not args.is_test:
            gt = load_gt(df, img_id)
            
            s, iou_r = compute_full_scores(gt, extracted)
            

            
            # Store Result
            result_entry = {
                "ImageID": img_id,
                "NASAScore": s,
                "Recall": iou_r,
                "CratersFound": len(extracted),
                "GTCraters": len(gt),
                "TimeSec": duration
            }

            detailed_results.append(result_entry)
        
        # --- Build Solution Rows (after scoring so we have match info) ---
        if len(extracted) == 0:
            # No craters detected - add a -1 row
            row = {
                'ellipseCenterX(px)': -1,
                'ellipseCenterY(px)': -1,
                'ellipseSemimajor(px)': -1,
                'ellipseSemiminor(px)': -1,
                'ellipseRotation(deg)': -1,
                'inputImage': path_str,
                'crater_classification': -1
            }

            solution_rows.append(row)
        else:
            for idx, c in enumerate(extracted):
                row = {
                    'ellipseCenterX(px)': c['x'],
                    'ellipseCenterY(px)': c['y'],
                    'ellipseSemimajor(px)': c['a'],
                    'ellipseSemiminor(px)': c['b'],
                    'ellipseRotation(deg)': c['angle'],
                    'inputImage': path_str,
                    'crater_classification': 4
                }

                solution_rows.append(row)
            

    
    # Export CSV (OUTSIDE THE LOOP)
    if not args.is_test:
        results_df = pd.DataFrame(detailed_results)
        csv_filename = "evaluation_results.csv"
        results_df.to_csv(csv_filename, index=False)
        
        print(f"\n==========================================")
        print(f"Results saved to {csv_filename}")
        print(f"Average NASA Score: {results_df['NASAScore'].mean():.4f}")
        

        
        print(f"Average Time per Image: {results_df['TimeSec'].mean():.4f} s")
        print(f"==========================================")
        
        # Save test-sub.csv
        test_sub_df = df[df['inputImage'].isin(tested_image_paths)]
        test_sub_df.to_csv("test-sub.csv", index=False)
        print(f"Saved test-sub.csv with {len(test_sub_df)} rows.")
    
    # Save solution-test.csv (Always)
    sol_df = pd.DataFrame(solution_rows)
    # Ensure column order matches inference.py/scorer.py expectations
    cols = [
        'ellipseCenterX(px)', 'ellipseCenterY(px)', 
        'ellipseSemimajor(px)', 'ellipseSemiminor(px)', 
        'ellipseRotation(deg)', 'inputImage', 'crater_classification'
    ]

    if not sol_df.empty:
        sol_df = sol_df[cols]
    else:
        sol_df = pd.DataFrame(columns=cols)
    if not args.is_test:
        sol_df.to_csv("solution-val.csv", index=False)
    else:
        sol_df.to_csv("solution-test.csv", index=False)
    print (sol_df.inputImage.unique().shape[0])
    print(f"Saved solution with {len(sol_df)} rows.")
    
    # Save Low Count Records
    low_count_filename = "low_count_test1024.csv" if args.is_test else "low_count_val1024.csv"
    if low_count_records:
        pd.DataFrame(low_count_records).to_csv(low_count_filename, index=False)
        print(f"Saved {len(low_count_records)} low-count image records to {low_count_filename}")
    else:
        print(f"No images found with < 10 craters!")

    print ('avg num per image', np.mean(num_per_images))
    
    # Report memory usage (RSS = Resident Set Size = actual process memory)
    final_mem = process.memory_info().rss / 1e6
    
    print(f"\n--- MEMORY USAGE (RSS) ---")
    print(f"Baseline:       {baseline_mem:.1f} MB")
    print(f"Final:          {final_mem:.1f} MB")
    if memory_samples:
        print(f"Mean during:    {np.mean(memory_samples):.1f} MB")
        print(f"Peak during:    {np.max(memory_samples):.1f} MB")
        print(f"Min during:     {np.min(memory_samples):.1f} MB")
    print(f"---------------------------")
    
    # Generate detailed timing report
    timing.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='mobileone_s2')
    parser.add_argument('--checkpoint', type=str, default=None, help="PyTorch checkpoint (not needed if using --onnx_model)")
    parser.add_argument('--processed_dir', type=str, default=None, help="Deprecated/Unused")
    parser.add_argument('--im_dim', type=int, default=3)
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--gt_csv', type=str, default='train-gt.csv')

    parser.add_argument('--is_test', action='store_true', default=False, help="Run in test mode (no GT, recursive glob)")

    parser.add_argument('--limit_craters', type=int, default=0, help="Limit number of craters per image (0 = no limit)")

    parser.add_argument('--config', type=str, default='config.yaml', help="Path to YAML config file")
    parser.add_argument('--model_type', type=str, default='CraterSMP', choices=['CraterSMP', 'CraterSMP_3Ch_RAM', 'CraterSMP_GroupNorm'], help="Model class to use")
    parser.add_argument('--onnx_model', type=str, default=None, help="Path to ONNX model (uses ONNX Runtime instead of PyTorch)")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on (cpu or cuda)")
    parser.add_argument('--tile_size', type=str, default="672x544", help="Tile size (WxH). Default '672x544' for 2x2 on 1296x1024. Set 0 to disable.")
    parser.add_argument('--limit_images', type=int, default=50, help="Limit number of images to process (used as fallback if filter fails)")


    parser.add_argument('--use_instance_norm', action='store_true', default=False,
                        help="Use instance-wise normalization: x = (x - mean(x)) / clamp(std(x), 0.1) "
                             "instead of fixed (x - 0.5) / 0.5. Must match training mode.")
    args = parser.parse_args()
    evaluate(args)


'''
Training will be executed on g5.12xlarge EC2 instances running Debian OS. GPU acceleration is allowed.
Testing will be executed on m8g.xlarge EC2 instances running Debian OS.
'''
'''
python train_ranker.py --input_data  ranking_data/ranking_train_data_768_ckpt20.csv  --ultra_fast  --use_binary
python eval3rect.py   --checkpoint ./kaggle/ram_1024_t544x416/best_model.pth   --processed_dir ./train_rect1024_5px   --raw_data_dir /media/saket/6f312db4-bbec-406a-b9d1-987b656a35c8/home/saket/nasa/   --gt_csv ./train-gt.csv   --config config_544x416.yaml --limit_craters 10 --model_type CraterSMP_3Ch_RAM   --fast  --tile_size 544x416 --ranker lightgbm_ranker_ultrafast.txt --ranker_fast --ranker_thres 0.1
'''
