from skimage.morphology import opening, disk
from skimage.feature import peak_local_max
import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import math
import time  # Added for timing
from tqdm import tqdm
from pathlib import Path
import sfs


from skimage import measure, segmentation, morphology
from skimage.measure import CircleModel
from scipy import ndimage as ndi


# ==========================================
# TAUBIN ELLIPSE FITTING (DROP-IN FOR cv2.fitEllipse)
# ==========================================
def fit_ellipse_taubin(pts):
    """
    Taubin algebraic ellipse fitting using normalized direct least squares.
    
    Drop-in replacement for cv2.fitEllipse.
    
    Args:
        pts: Points array in cv2 format: (N, 1, 2) or (N, 2), dtype float32 or int
             Points are (x, y) coordinates.
    
    Returns:
        ((cx, cy), (w, h), angle) in cv2 format:
        - (cx, cy): center coordinates
        - (w, h): full width and height (not semi-axes)
        - angle: rotation in degrees (0-180)
        
    Raises:
        ValueError: if fitting fails (caller should catch and fallback)
    """
    # Reshape to (N, 2)
    pts = np.asarray(pts).reshape(-1, 2).astype(np.float64)
    
    if len(pts) < 5:
        raise ValueError("Need at least 5 points for ellipse fitting")
    
    # Center and normalize for numerical stability
    mean_x = np.mean(pts[:, 0])
    mean_y = np.mean(pts[:, 1])
    
    x = pts[:, 0] - mean_x
    y = pts[:, 1] - mean_y
    
    # Normalize scale
    scale = np.sqrt(np.mean(x**2 + y**2))
    if scale < 1e-10:
        raise ValueError("Points are too close together")
    
    x = x / scale
    y = y / scale
    
    # Build design matrix for conic: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])
    
    # Scatter matrix
    S = D.T @ D
    
    # Constraint matrix for ellipse (Taubin: 4AC - B^2 = 1)
    # Using the constraint that guarantees ellipse: C = [[0,0,2,0,0,0], [0,-1,0,0,0,0], [2,0,0,0,0,0], ...]
    C = np.zeros((6, 6))
    C[0, 2] = 2.0
    C[2, 0] = 2.0
    C[1, 1] = -1.0
    
    # Solve generalized eigenvalue problem: S @ v = lambda * C @ v
    try:
        # Use pseudo-inverse approach for stability
        S_inv = np.linalg.pinv(S)
        M = S_inv @ C
        
        eigvals, eigvecs = np.linalg.eig(M)
        
        # Find the eigenvector with positive eigenvalue that satisfies ellipse constraint
        # 4*A*C - B^2 > 0 for ellipse
        valid_idx = None
        best_val = -np.inf
        
        for i in range(6):
            if np.isreal(eigvals[i]) and np.real(eigvals[i]) > 1e-10:
                v = np.real(eigvecs[:, i])
                # Check ellipse constraint: 4*A*C - B^2 > 0
                constraint = 4.0 * v[0] * v[2] - v[1]**2
                if constraint > 0 and np.real(eigvals[i]) > best_val:
                    best_val = np.real(eigvals[i])
                    valid_idx = i
        
        if valid_idx is None:
            raise ValueError("No valid ellipse solution found")
        
        a_vec = np.real(eigvecs[:, valid_idx])
        
    except np.linalg.LinAlgError:
        raise ValueError("Numerical error in eigenvalue computation")
    
    # Extract conic coefficients
    A, B, C_coef, D_coef, E_coef, F_coef = a_vec
    
    # Denominator for center calculation
    denom = B**2 - 4.0 * A * C_coef
    
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate conic (not an ellipse)")
    
    # Calculate center in normalized coordinates
    cx_norm = (2.0 * C_coef * D_coef - B * E_coef) / denom
    cy_norm = (2.0 * A * E_coef - B * D_coef) / denom
    
    # Transform center back to original coordinates
    cx = cx_norm * scale + mean_x
    cy = cy_norm * scale + mean_y
    
    # Calculate rotation angle
    if abs(A - C_coef) < 1e-12:
        theta = np.pi / 4.0 if B > 0 else -np.pi / 4.0
    else:
        theta = 0.5 * np.arctan2(B, A - C_coef)
    
    # Calculate semi-axes lengths
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Rotate coefficients
    A_rot = A * cos_t**2 + B * cos_t * sin_t + C_coef * sin_t**2
    C_rot = A * sin_t**2 - B * cos_t * sin_t + C_coef * cos_t**2
    
    # Value at center
    F_center = A * cx_norm**2 + B * cx_norm * cy_norm + C_coef * cy_norm**2 + \
               D_coef * cx_norm + E_coef * cy_norm + F_coef
    
    if abs(A_rot) < 1e-12 or abs(C_rot) < 1e-12:
        raise ValueError("Degenerate ellipse axes")
    
    # Semi-axes in normalized coordinates
    a_sq = -F_center / A_rot
    b_sq = -F_center / C_rot
    
    if a_sq <= 0 or b_sq <= 0:
        raise ValueError("Invalid ellipse (negative axes)")
    
    # Semi-axes, scaled back
    a_axis = np.sqrt(a_sq) * scale
    b_axis = np.sqrt(b_sq) * scale
    
    # Convert to cv2 format: (width, height) = (2*a, 2*b), angle in degrees
    # cv2 angle is from x-axis to major axis, in range [0, 180)
    angle_deg = np.degrees(theta)
    
    # Ensure angle in [0, 180)
    while angle_deg < 0:
        angle_deg += 180
    while angle_deg >= 180:
        angle_deg -= 180
    
    # cv2 convention: width is along the rotated x-axis direction
    w = 2.0 * a_axis
    h = 2.0 * b_axis
    
    # Sanity check
    if not (np.isfinite(cx) and np.isfinite(cy) and 
            np.isfinite(w) and np.isfinite(h) and np.isfinite(angle_deg)):
        raise ValueError("Non-finite ellipse parameters")
    
    if w < 1e-3 or h < 1e-3:
        raise ValueError("Ellipse too small")
    
    return (cx, cy), (w, h), angle_deg


def fit_ellipse_taubin_safe(pts):
    """
    Safe wrapper for fit_ellipse_taubin that falls back to cv2.fitEllipse on failure.
    
    Args:
        pts: Points array in cv2 format: (N, 1, 2) or (N, 2)
    
    Returns:
        ((cx, cy), (w, h), angle) in cv2 format
    """
    try:
        return fit_ellipse_taubin(pts)
    except (ValueError, np.linalg.LinAlgError, IndexError):
        # Fallback to cv2
        pts_cv2 = np.asarray(pts).reshape(-1, 1, 2).astype(np.float32)
        return cv2.fitEllipse(pts_cv2)


# ==========================================
# GEOMETRIC ELLIPSE FITTING (ORTHOGONAL DISTANCE)
# ==========================================
def _ellipse_point_distance(px, py, cx, cy, a, b, theta):
    """
    Compute the orthogonal (geometric) distance from point (px, py) to the ellipse.
    Uses Newton's method to find the closest point on the ellipse.
    
    Ellipse: center (cx, cy), semi-axes a, b, rotation theta (radians)
    """
    # Transform point to ellipse-centered, axis-aligned coordinates
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Translate
    dx = px - cx
    dy = py - cy
    
    # Rotate to align with ellipse axes
    x_local = dx * cos_t + dy * sin_t
    y_local = -dx * sin_t + dy * cos_t
    
    # Now find closest point on axis-aligned ellipse centered at origin
    # Ellipse: (x/a)^2 + (y/b)^2 = 1
    
    # Handle degenerate cases
    if a < 1e-6 or b < 1e-6:
        return np.sqrt(dx*dx + dy*dy)
    
    # Use symmetry - work in first quadrant
    x_abs = abs(x_local)
    y_abs = abs(y_local)
    
    # Initial guess for parameter t (angle on ellipse)
    t = np.arctan2(a * y_abs, b * x_abs)
    
    # Newton's method to find closest point
    for _ in range(10):
        cos_phi = np.cos(t)
        sin_phi = np.sin(t)
        
        # Point on ellipse
        ex = a * cos_phi
        ey = b * sin_phi
        
        # Difference
        rx = ex - x_abs
        ry = ey - y_abs
        
        # Gradient of distance squared w.r.t. t
        # d/dt [(a*cos(t) - x)^2 + (b*sin(t) - y)^2]
        # = 2*(a*cos(t) - x)*(-a*sin(t)) + 2*(b*sin(t) - y)*(b*cos(t))
        grad = -2 * a * sin_phi * rx + 2 * b * cos_phi * ry
        
        # Second derivative (Hessian)
        hess = -2 * a * cos_phi * rx - 2 * a * a * sin_phi * sin_phi + \
               -2 * b * sin_phi * ry + 2 * b * b * cos_phi * cos_phi
        
        if abs(hess) < 1e-12:
            break
            
        # Newton step
        dt = -grad / hess
        t = t + dt
        
        if abs(dt) < 1e-8:
            break
    
    # Final closest point on ellipse (in local coords)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    
    # Distance
    dist = np.sqrt((x_abs - ex)**2 + (y_abs - ey)**2)
    
    return dist


def _ellipse_residuals(params, points):
    """
    Compute orthogonal distances from all points to the ellipse.
    Used as residual function for least_squares optimization.
    """
    cx, cy, a, b, theta = params
    
    # Ensure valid ellipse
    a = max(abs(a), 1.0)
    b = max(abs(b), 1.0)
    
    residuals = np.zeros(len(points))
    for i, (px, py) in enumerate(points):
        residuals[i] = _ellipse_point_distance(px, py, cx, cy, a, b, theta)
    
    return residuals


def fit_ellipse_geometric(pts, max_iter=50):
    """
    Geometric ellipse fitting using Levenberg-Marquardt optimization.
    Minimizes sum of squared ORTHOGONAL distances from points to ellipse.
    
    This is more accurate than algebraic methods (cv2.fitEllipse, Taubin)
    because it minimizes actual geometric distance, not algebraic residual.
    
    Args:
        pts: Points array in cv2 format: (N, 1, 2) or (N, 2), dtype float32 or int
        max_iter: Maximum optimization iterations
    
    Returns:
        ((cx, cy), (w, h), angle) in cv2 format:
        - (cx, cy): center coordinates
        - (w, h): full width and height (not semi-axes)
        - angle: rotation in degrees (0-180)
    
    Raises:
        ValueError: if fitting fails
    """
    from scipy.optimize import least_squares
    
    # Reshape to (N, 2)
    pts = np.asarray(pts).reshape(-1, 2).astype(np.float64)
    
    if len(pts) < 5:
        raise ValueError("Need at least 5 points for ellipse fitting")
    
    # Get initial estimate from cv2 (algebraic fit)
    pts_cv2 = pts.reshape(-1, 1, 2).astype(np.float32)
    try:
        (cx0, cy0), (w0, h0), angle0 = cv2.fitEllipse(pts_cv2)
        a0 = w0 / 2.0
        b0 = h0 / 2.0
        theta0 = np.deg2rad(angle0)
    except:
        # Fallback: use mean and std
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        std_x = np.std(pts[:, 0])
        std_y = np.std(pts[:, 1])
        a0 = max(std_x * 2, 5)
        b0 = max(std_y * 2, 5)
        theta0 = 0.0
    
    # Initial parameters
    x0 = np.array([cx0, cy0, a0, b0, theta0])
    
    # Bounds to prevent degenerate solutions
    lb = [cx0 - 100, cy0 - 100, 3.0, 3.0, -np.pi]
    ub = [cx0 + 100, cy0 + 100, max(a0, b0) * 3, max(a0, b0) * 3, np.pi]
    
    # Optimize
    try:
        result = least_squares(
            _ellipse_residuals,
            x0,
            args=(pts,),
            bounds=(lb, ub),
            method='trf',  # Trust Region Reflective
            max_nfev=max_iter * 5,
            ftol=1e-6,
            xtol=1e-6
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        cx, cy, a, b, theta = result.x
        
    except Exception as e:
        raise ValueError(f"Geometric fitting failed: {e}")
    
    # Ensure a, b are positive
    a = abs(a)
    b = abs(b)
    
    # Convert to cv2 format
    w = 2.0 * a
    h = 2.0 * b
    angle_deg = np.degrees(theta)
    
    # Normalize angle to [0, 180)
    while angle_deg < 0:
        angle_deg += 180
    while angle_deg >= 180:
        angle_deg -= 180
    
    # Sanity check
    if not (np.isfinite(cx) and np.isfinite(cy) and 
            np.isfinite(w) and np.isfinite(h) and np.isfinite(angle_deg)):
        raise ValueError("Non-finite ellipse parameters")
    
    if w < 1.0 or h < 1.0:
        raise ValueError("Ellipse too small")
    
    return (cx, cy), (w, h), angle_deg


def fit_ellipse_geometric_safe(pts, max_iter=50):
    """
    Safe wrapper for fit_ellipse_geometric that falls back to cv2.fitEllipse on failure.
    
    Args:
        pts: Points array in cv2 format: (N, 1, 2) or (N, 2)
        max_iter: Maximum optimization iterations
    
    Returns:
        ((cx, cy), (w, h), angle) in cv2 format
    """
    try:
        return fit_ellipse_geometric(pts, max_iter)
    except (ValueError, Exception):
        # Fallback to cv2
        pts_cv2 = np.asarray(pts).reshape(-1, 1, 2).astype(np.float32)
        return cv2.fitEllipse(pts_cv2)




# ==========================================
# RECTANGULAR IMAGE UTILS
# ==========================================
def compute_ellipse_iou(c1, c2, n_samples=72):
    """
    Approximate IoU between two ellipses by sampling points.
    Returns overlap ratio relative to the smaller ellipse.
    """
    # Get areas
    area1 = math.pi * c1['a'] * c1['b']
    area2 = math.pi * c2['a'] * c2['b']
    
    # Quick distance check
    dist = math.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
    max_r1 = max(c1['a'], c1['b'])
    max_r2 = max(c2['a'], c2['b'])
    
    if dist > max_r1 + max_r2:
        return 0.0  # No overlap possible
    
    # Check if one center is inside the other ellipse
    # Simplified check: if center distance < larger radius, likely overlap
    center_in_larger = dist < max(max_r1, max_r2) * 0.8
    
    if center_in_larger:
        # Smaller ellipse is likely inside larger
        smaller_area = min(area1, area2)
        larger_area = max(area1, area2)
        return smaller_area / larger_area
    
    return 0.0

def outer_rim_points(binary_mask, thickness=1):
    """
    Extract outer rim pixels only.
    thickness = how many pixels inward to ignore
    
    OPTIMIZED: Uses morphological erosion instead of distance_transform_edt.
    Erosion is O(n) vs EDT's O(n²), providing ~10x speedup for typical regions.
    """
    mask_u8 = binary_mask.astype(np.uint8)
    
    # Create eroded version (interior)
    if thickness == 1:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*thickness+1, 2*thickness+1))
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
    
    # Rim = mask - eroded interior
    rim = mask_u8 - eroded
    ys, xs = np.where(rim > 0)
    return np.stack([xs, ys], axis=1)

def angular_sample(points, center, bins=64):
    """
    Sample points uniformly around the perimeter by selecting the outermost point in each angular bin.
    
    OPTIMIZED: Vectorized implementation using numpy instead of Python loop.
    """
    cx, cy = center
    pts = points.astype(np.float32)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx*dx + dy*dy)
    
    # Map angles from [-pi, pi] to [0, bins-1]
    bin_ids = ((angles + np.pi) / (2 * np.pi) * bins).astype(np.int32)
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    
    # Vectorized: for each unique bin, find the point with maximum radius
    # Use negative radii trick with np.lexsort or np.unique with return_index
    unique_bins = np.unique(bin_ids)
    selected_indices = []
    
    for b in unique_bins:
        mask = bin_ids == b
        bin_radii = radii[mask]
        bin_indices = np.where(mask)[0]
        max_idx = bin_indices[np.argmax(bin_radii)]
        selected_indices.append(max_idx)
    
    return pts[selected_indices]


def valid_ellipse(w, h):
    if w < 5 or h < 5:
        return False
    ratio = max(w, h) / (min(w, h) + 1e-6)
    return ratio < 4.0


def enforce_eccentricity_floor(w, h, min_e=0.15):
    import math
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0
    e = math.sqrt(1.0 - (b*b)/(a*a))
    if e >= min_e:
        return w, h
    # stretch slightly along major axis
    target_b = a * math.sqrt(1 - min_e*min_e)
    scale = target_b / b
    return w, h * scale


def find_optimal_scale_from_rim(cx, cy, w_base, h_base, angle, rim_prob,
                                 scale_min=1.0, scale_max=1.35, steps=15, n_samples = 72):
    """
    Find the scale factor where rim probability is maximized along the ellipse perimeter.
    
    OPTIMIZED: Precomputes trig values and uses fewer samples (48 vs 72).
    """
    H, W = rim_prob.shape
    
    thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Precompute all trig values ONCE
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    cos_ang = np.cos(np.deg2rad(angle))
    sin_ang = np.sin(np.deg2rad(angle))
    
    best_scale = 1.14  # Fallback to fixed scale
    best_prob = 0.0
    
    a_base = w_base / 2.0
    b_base = h_base / 2.0
    
    # Precompute the rotation-transformed unit ellipse vectors
    # x_unit = cos_thetas * cos_ang - sin_thetas * sin_ang
    # y_unit = cos_thetas * sin_ang + sin_thetas * cos_ang
    x_cos_component = cos_thetas * cos_ang
    x_sin_component = sin_thetas * sin_ang
    y_cos_component = cos_thetas * sin_ang
    y_sin_component = sin_thetas * cos_ang
    
    scales = np.linspace(scale_min, scale_max, steps)
    
    for scale in scales:
        a = a_base * scale
        b = b_base * scale
        
        # Sample points along ellipse perimeter
        x_p = cx + a * x_cos_component - b * x_sin_component
        y_p = cy + a * y_cos_component + b * y_sin_component
        
        # Check bounds
        valid = (x_p >= 0) & (x_p < W) & (y_p >= 0) & (y_p < H)
        n_valid = np.sum(valid)
        if n_valid < n_samples // 2:  # Need at least half the points
            continue
        
        x_idx = x_p[valid].astype(np.int32)
        y_idx = y_p[valid].astype(np.int32)
        
        avg_prob = np.mean(rim_prob[y_idx, x_idx])
        
        if avg_prob > best_prob:
            best_prob = avg_prob
            best_scale = scale
    
    return best_scale, best_prob


def compute_ellipse_confidence(cx, cy, a, b, angle, rim_prob, global_prob, n_samples=120):
    """
    Compute confidence score using ranking-inspired features.
    
    Combines:
        - rim_prob_frac_above_50: Fraction of perimeter with rim_prob > 0.5 (weight 0.7)
        - geometry_angular_std: Normalized std of angular bin counts (weight 0.15)
        - geometry_max_gap: 1 - max angular gap (weight 0.15)
    
    Returns:
        confidence: float in [0, 1]
        details: dict with component values
    """
    H, W = rim_prob.shape
    thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Precompute trig values
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    cos_ang = np.cos(np.deg2rad(angle))
    sin_ang = np.sin(np.deg2rad(angle))
    
    # Precompute rotation components
    x_cos = cos_thetas * cos_ang
    x_sin = sin_thetas * sin_ang
    y_cos = cos_thetas * sin_ang
    y_sin = sin_thetas * cos_ang
    
    # Sample perimeter
    x_p = cx + a * x_cos - b * x_sin
    y_p = cy + a * y_cos + b * y_sin
    
    valid = (x_p >= 0) & (x_p < W) & (y_p >= 0) & (y_p < H)
    n_valid = np.sum(valid)
    
    if n_valid < n_samples // 2:
        return 0.0, {'rim_prob_frac_above_50': 0, 'geometry_angular_std_norm': 0, 
                     'geometry_max_gap': 1.0, 'angular_completeness': 0}
    
    x_idx = x_p[valid].astype(np.int32)
    y_idx = y_p[valid].astype(np.int32)
    valid_thetas = thetas[valid]
    
    # 1. rim_prob_frac_above_50: Fraction with rim_prob > 0.5
    rim_probs = rim_prob[y_idx, x_idx]
    rim_prob_frac_above_50 = np.mean(rim_probs > 0.5)
    
    # 2. geometry_angular_std: Standard deviation of angular bin counts
    # Normalize angles to [0, 2pi]
    angles = np.mod(valid_thetas, 2 * np.pi)
    n_bins = 36  # 10 degrees each, matches ranking_features.py
    bins = np.histogram(angles, bins=n_bins, range=(0, 2*np.pi))[0]
    angular_std = np.std(bins)
    
    # Normalize angular_std to [0, 1] range
    # Expected std for uniform distribution: sqrt(n_valid / n_bins) approximately
    # For n_valid=120, n_bins=36: expected ~1.8, max ~3.3
    # Higher std = more uneven distribution = worse
    # Invert: geometry_angular_std_norm = 1 - (std / max_std)
    expected_per_bin = n_valid / n_bins
    max_std = np.sqrt(expected_per_bin * (n_bins - 1))  # Worst case: all in one bin
    angular_std_norm = 1.0 - min(angular_std / (max_std + 1e-6), 1.0)
    
    # 3. geometry_max_gap: Maximum angular gap as fraction of 2pi
    sorted_angles = np.sort(angles)
    if len(sorted_angles) > 1:
        gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 2 * np.pi]))
        max_gap = gaps.max() / (2 * np.pi)
    else:
        max_gap = 1.0
    
    # Invert max_gap: smaller gap = better coverage = higher score
    max_gap_score = 1.0 - max_gap
    
    # 4. geometry_support_ratio: number of rim points / expected perimeter length
    # Higher = more dense rim coverage = better match (correlation +0.42)
    expected_perimeter = 2 * np.pi * np.sqrt((a * a + b * b) / 2)
    geometry_support_ratio = n_valid / (expected_perimeter + 1e-6)
    # Normalize to ~[0, 1] range (typical values 0.5-2.0, cap at 1.0)
    geometry_support_ratio = min(geometry_support_ratio, 1.0)
    
    # 5. geometry_angular_coverage: fraction of angular bins with at least 1 point
    # Higher = more complete rim = better match (correlation +0.39)
    geometry_angular_coverage = np.count_nonzero(bins) / n_bins
    
    # rim_score for backward compatibility
    rim_score = np.mean(rim_probs)
    
    # FAILED FORMULA v1 (dropped rim_score weight too much - 2% score drop):
    # confidence = 0.25 * geometry_support_ratio + 0.25 * geometry_angular_coverage + 0.20 * max_gap_score + 0.15 * rim_prob_frac_above_50 + 0.15 * rim_score
    
    # FAILED FORMULA v2 (still 0.8% drop):
    # confidence = 0.50 * rim_score + 0.20 * rim_prob_frac_above_50 + 0.10 * geometry_support_ratio + 0.10 * geometry_angular_coverage + 0.10 * max_gap_score
    
    # ORIGINAL WORKING FORMULA (98.12% score):
    confidence = 0.4 * rim_score + 0.3 * rim_prob_frac_above_50 + 0.15 * angular_std_norm + 0.15 * max_gap_score
    
    # Angular completeness for backward compatibility (used in filtering)
    angular_completeness = np.sum(rim_probs > 0.2) / n_valid
    
    details = {
        'rim_prob_frac_above_50': rim_prob_frac_above_50,
        'geometry_angular_std_norm': angular_std_norm,
        'geometry_max_gap': max_gap,
        'geometry_support_ratio': geometry_support_ratio,
        'geometry_angular_coverage': geometry_angular_coverage,
        'angular_completeness': angular_completeness,
        # Keep old keys for compatibility
        'rim_score': rim_score,
        'global_coverage': 0.0  # Not computed in new version
    }
    
    return confidence, details


def compute_ellipse_confidence_with_gradient(cx, cy, a, b, angle, rim_prob, grad_mag, n_samples=120):
    """
    Shadow-aware confidence: uses gradient magnitude when rim_prob is weak.
    
    In shadow/terminator regions:
    - rim_prob is low (UNet sees no texture)
    - gradient is still strong (shadow edge creates intensity change)
    
    This function computes a hybrid signal that blends rim_prob and gradient
    based on rim_prob strength at each sample point.
    
    Returns:
        confidence: float in [0, 1]
        details: dict with component values
    """
    H, W = rim_prob.shape
    thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Precompute trig values
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    cos_ang = np.cos(np.deg2rad(angle))
    sin_ang = np.sin(np.deg2rad(angle))
    
    # Precompute rotation components
    x_cos = cos_thetas * cos_ang
    x_sin = sin_thetas * sin_ang
    y_cos = cos_thetas * sin_ang
    y_sin = sin_thetas * cos_ang
    
    # Sample perimeter
    x_p = cx + a * x_cos - b * x_sin
    y_p = cy + a * y_cos + b * y_sin
    
    valid = (x_p >= 0) & (x_p < W) & (y_p >= 0) & (y_p < H)
    n_valid = np.sum(valid)
    
    if n_valid < n_samples // 2:
        return 0.0, {'rim_prob_frac_above_50': 0, 'geometry_angular_std_norm': 0, 
                     'geometry_max_gap': 1.0, 'angular_completeness': 0}
    
    x_idx = x_p[valid].astype(np.int32)
    y_idx = y_p[valid].astype(np.int32)
    valid_thetas = thetas[valid]
    
    # Sample rim_prob and gradient at perimeter points
    rim_vals = rim_prob[y_idx, x_idx]
    grad_vals = grad_mag[y_idx, x_idx]
    
    # Normalize gradient to [0, 1] range using 95th percentile
    grad_p95 = np.percentile(grad_mag, 95)
    grad_norm = np.clip(grad_vals / (grad_p95 + 1e-6), 0, 1)
    
    # HYBRID SIGNAL: Use gradient where rim_prob is weak
    # If rim_prob > 0.5: trust rim_prob fully
    # If rim_prob < 0.2: trust gradient more (scaled to 0.7 max confidence contribution)
    # Smooth transition between 0.2 and 0.5
    alpha = np.clip((rim_vals - 0.2) / 0.3, 0, 1)
    hybrid_signal = alpha * rim_vals + (1 - alpha) * grad_norm * 0.7
    
    # 1. Hybrid-based metrics (replace rim_prob metrics)
    hybrid_score = np.mean(hybrid_signal)
    hybrid_frac_above_40 = np.mean(hybrid_signal > 0.4)
    
    # 2. geometry_angular_std (same as original)
    angles = np.mod(valid_thetas, 2 * np.pi)
    n_bins = 36
    bins = np.histogram(angles, bins=n_bins, range=(0, 2*np.pi))[0]
    angular_std = np.std(bins)
    expected_per_bin = n_valid / n_bins
    max_std = np.sqrt(expected_per_bin * (n_bins - 1))
    angular_std_norm = 1.0 - min(angular_std / (max_std + 1e-6), 1.0)
    
    # 3. geometry_max_gap (same as original)
    sorted_angles = np.sort(angles)
    if len(sorted_angles) > 1:
        gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 2 * np.pi]))
        max_gap = gaps.max() / (2 * np.pi)
    else:
        max_gap = 1.0
    max_gap_score = 1.0 - max_gap
    
    # Combined confidence using hybrid signal
    # Same weights as original, but on hybrid metrics
    confidence = 0.4 * hybrid_score + 0.3 * hybrid_frac_above_40 + 0.15 * angular_std_norm + 0.15 * max_gap_score
    
    # Angular completeness for filtering (uses hybrid signal threshold)
    angular_completeness = np.sum(hybrid_signal > 0.2) / n_valid
    
    # Also track shadow fraction for diagnostics
    shadow_fraction = np.mean(rim_vals < 0.3)
    
    details = {
        'rim_prob_frac_above_50': np.mean(rim_vals > 0.5),  # Original for comparison
        'geometry_angular_std_norm': angular_std_norm,
        'geometry_max_gap': max_gap,
        'angular_completeness': angular_completeness,
        # New gradient-specific metrics
        'rim_score': np.mean(rim_vals),
        'hybrid_score': hybrid_score,
        'shadow_fraction': shadow_fraction,
        'global_coverage': 0.0
    }
    
    return confidence, details


def compute_ellipse_confidence_old(cx, cy, a, b, angle, rim_prob, global_prob, n_samples=120):
    """
    Compute a comprehensive confidence score for an ellipse detection.
    
    OPTIMIZED: Precomputes trig values and uses fewer samples (48 vs 72).
    """
    H, W = rim_prob.shape
    thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Precompute trig values
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    cos_ang = np.cos(np.deg2rad(angle))
    sin_ang = np.sin(np.deg2rad(angle))
    
    # Precompute rotation components
    x_cos = cos_thetas * cos_ang
    x_sin = sin_thetas * sin_ang
    y_cos = cos_thetas * sin_ang
    y_sin = sin_thetas * cos_ang
    
    # Sample perimeter
    x_p = cx + a * x_cos - b * x_sin
    y_p = cy + a * y_cos + b * y_sin
    
    valid = (x_p >= 0) & (x_p < W) & (y_p >= 0) & (y_p < H)
    n_valid = np.sum(valid)
    
    if n_valid < n_samples // 2:
        return 0.0, {'rim_score': 0, 'angular_completeness': 0, 'global_coverage': 0, 'snr': 0, 'low_snr': False}
    
    x_idx = x_p[valid].astype(np.int32)
    y_idx = y_p[valid].astype(np.int32)
    
    # 1. Average rim probability along perimeter
    rim_probs = rim_prob[y_idx, x_idx]
    rim_score = np.mean(rim_probs)
    
    # 2. Angular completeness: fraction of perimeter with rim_prob > 0.2
    angular_completeness = np.sum(rim_probs > 0.2) / n_valid
    
    # 3. Global coverage: sample interior points and check global mask
    # OPTIMIZED: Vectorize across all scales at once
    inner_scales = np.array([0.3, 0.5, 0.7])
    sparse_thetas_idx = slice(0, n_samples, 4)  # Every 4th theta
    
    x_cos_sparse = x_cos[sparse_thetas_idx]
    x_sin_sparse = x_sin[sparse_thetas_idx]
    y_cos_sparse = y_cos[sparse_thetas_idx]
    y_sin_sparse = y_sin[sparse_thetas_idx]
    
    global_hits = 0
    global_total = 0
    
    for s in inner_scales:
        x_inner = cx + (a * s) * x_cos_sparse - (b * s) * x_sin_sparse
        y_inner = cy + (a * s) * y_cos_sparse + (b * s) * y_sin_sparse
        
        inner_valid = (x_inner >= 0) & (x_inner < W) & (y_inner >= 0) & (y_inner < H)
        n_inner_valid = np.sum(inner_valid)
        if n_inner_valid > 0:
            x_i = x_inner[inner_valid].astype(np.int32)
            y_i = y_inner[inner_valid].astype(np.int32)
            global_hits += np.sum(global_prob[y_i, x_i] > 0.5)
            global_total += n_inner_valid
    
    global_coverage = global_hits / (global_total + 1e-6)
    
    # Combined confidence
    #confidence = 0.5 * rim_score + 0.3 * angular_completeness + 0.2 * global_coverage
    confidence = 0.8 * rim_score + 0.1 * angular_completeness + 0.1 * global_coverage
    details = {
        'rim_score': rim_score,
        'angular_completeness': angular_completeness,
        'global_coverage': global_coverage
    }
    
    return confidence, details

def extract_craters_cv2_adaptive_selection_rescue(pred_tensor,
                                  threshold=0.85,
                                  rim_thresh=0.25,
                                  ecc_floor=0.15,
                                  scale_min=1.0,
                                  scale_max=1.4,
                                  scale_steps=10,
                                  min_confidence=0.97,
                                  min_confidence_min=0.90,
                                  fit_method='cv2',
                                  min_semi_axis=0,
                                  max_size_ratio=1.0,
                                  require_fully_visible=False,
                                  image_shape=None,
                                  downsample_factor=1.0,
                                  skip_prop_image=False,
                                  return_all_candidates=False,
                                  n_samples_scale=72,
                                  n_samples_conf=120,
                                  angular_completeness_thresh=0.35,
                                  use_grad_conf=False,
                                  grad_mag=None):
    """
    Crater extraction with adaptive per-crater scaling and confidence-based FP suppression.
    selection version: iteratively lowers min_confidence until at least 10 craters are found.
    
    Args:
        fit_method: Ellipse fitting method. Options:
            - 'cv2': OpenCV's fitEllipse (fast, algebraic)
            - 'taubin': Taubin algebraic method (similar to cv2)
            - 'geometric': Geometric fitting minimizing orthogonal distance (slower but more accurate)
        min_semi_axis: Minimum semi-minor axis size (pixels). Craters smaller than this are filtered. Default 0 (no filter).
        max_size_ratio: Maximum (bbox_w + bbox_h) / min(img_w, img_h) ratio. Craters larger than this are filtered. Default 1.0 (no filter).
        require_fully_visible: If True, filter craters whose bounding box extends outside image bounds. Default False.
        image_shape: Tuple (height, width) of the original image. Required if require_fully_visible=True or max_size_ratio<1.0.
        downsample_factor: Factor to downsample masks for EDT/watershed (1.0=full res, 1.5=0.67x, 2.0=0.5x).
                          Higher values = faster but may reduce accuracy. Default 1.0 (full resolution).
        skip_prop_image: If True, avoid expensive prop.image access by slicing labels array directly.
                        Can provide ~30% speedup on propAccess. Default False.
        return_all_candidates: If True, return ALL candidates before confidence-based selection.
                              This is useful for generating ranking training data with hard negatives.
                              Default False (use adaptive selection to return ~10 best candidates).
        n_samples_scale: Number of samples for find_optimal_scale_from_rim (default 72).
        n_samples_conf: Number of samples for compute_ellipse_confidence (default 120).
        angular_completeness_thresh: Minimum angular completeness to accept a candidate (default 0.35).
                                    Use lower values (e.g. 0.2) for training data to get more candidates.
    """
    import time as _time
    _t_start = _time.perf_counter()
    
    # Keep full-resolution data for precise ellipse fitting
    rim_prob = pred_tensor[2]
    global_prob = pred_tensor[1]
    H_full, W_full = rim_prob.shape
    
    # === SEGMENTATION (optionally downsampled) ===
    # downsample_factor=1.0: full resolution (default, accurate)
    # downsample_factor=1.5: ~2x speedup with minimal accuracy loss
    # downsample_factor=2.0: ~4x speedup but may lose ~1% accuracy
    DOWNSAMPLE_FACTOR = downsample_factor
    
    # Create masks
    mask_core_full = (pred_tensor[0] > threshold).astype(np.uint8)
    mask_global_full = (pred_tensor[1] > threshold).astype(np.uint8)
    
    if DOWNSAMPLE_FACTOR > 1.0:
        # Downsample for faster segmentation
        H_ds = int(H_full / DOWNSAMPLE_FACTOR)
        W_ds = int(W_full / DOWNSAMPLE_FACTOR)
        mask_core = cv2.resize(mask_core_full, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
        mask_global = cv2.resize(mask_global_full, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
    else:
        # Full resolution (default)
        mask_core = mask_core_full
        mask_global = mask_global_full
    
    _t_mask = _time.perf_counter()

    markers = measure.label(mask_core, background=0)
    _t_label = _time.perf_counter()
    
    distance = ndi.distance_transform_edt(mask_global)
    _t_edt = _time.perf_counter()
    
    labels = segmentation.watershed(-distance, markers, mask=mask_global)
    _t_watershed = _time.perf_counter()

    candidates = []
    _t_outer_rim_total = 0.0
    _t_angular_sample_total = 0.0
    _t_fit_ellipse_total = 0.0
    _t_find_scale_total = 0.0
    _t_compute_conf_total = 0.0
    _n_regions = 0
    _n_candidates = 0
    _t_regionprops_access = 0.0
    _t_misc_ops = 0.0

    _t0 = _time.perf_counter()
    all_props = measure.regionprops(labels)
    _t_regionprops_call = _time.perf_counter() - _t0

    for prop in all_props:
        '''
        if prop.area < 30: continue
        if prop.minor_axis_length < 3: continue
        if prop.eccentricity > 0.95: continue
        if prop.solidity < 0.7: continue
        '''
        _n_regions += 1
        
        # === EARLY REJECTION WITH CHEAP PROPERTIES (at downsampled scale) ===
        # NOTE: Areas and lengths are at 0.5x scale, so thresholds are scaled by DOWNSAMPLE_FACTOR²
        
        # 1. Area check (CHEAPEST) - scaled threshold
        _t0 = _time.perf_counter()
        area_ds = prop.area  # At downsampled scale
        area_full = area_ds * (DOWNSAMPLE_FACTOR ** 2)  # Approximate full-res area
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if area_full < 40:
            continue
        
        #if area_full > 100000:
        #    continue
            
        # 2. Minor axis check (MODERATE) - scaled threshold
        _t0 = _time.perf_counter()
        minor_axis_ds = prop.minor_axis_length
        minor_axis_full = minor_axis_ds * DOWNSAMPLE_FACTOR
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if minor_axis_full < max(3, 0.02 * np.sqrt(area_full)):
            continue
        
        # 3. Eccentricity check (scale-invariant)
        _t0 = _time.perf_counter()
        ecc = prop.eccentricity
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if ecc > 0.95 and area_full < 200:
            continue
        
        # 4. Solidity check (scale-invariant)
        _t0 = _time.perf_counter()
        sol = prop.solidity
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if sol < 0.7 and area_full < 150:
            continue

        # === MAP BBOX TO FULL RESOLUTION ===
        if DOWNSAMPLE_FACTOR > 1.0:
            # === DOWNSAMPLED PATH: Map bbox to full resolution ===
            _t0 = _time.perf_counter()
            minr_ds, minc_ds, maxr_ds, maxc_ds = prop.bbox
            pad = 5  # Extra padding at full resolution for accuracy
            minr_full = int(max(0, minr_ds * DOWNSAMPLE_FACTOR - pad))
            minc_full = int(max(0, minc_ds * DOWNSAMPLE_FACTOR - pad))
            maxr_full = int(min(H_full, maxr_ds * DOWNSAMPLE_FACTOR + pad))
            maxc_full = int(min(W_full, maxc_ds * DOWNSAMPLE_FACTOR + pad))
            _t_regionprops_access += _time.perf_counter() - _t0
            
            # Extract rim points from full-res rim probability
            _t0 = _time.perf_counter()
            rim_region = rim_prob[minr_full:maxr_full, minc_full:maxc_full]
            global_region = (global_prob[minr_full:maxr_full, minc_full:maxc_full] > threshold).astype(np.uint8)
            
            rim_mask = (rim_region > rim_thresh) & (global_region > 0)
            ys, xs = np.where(rim_mask)
            
            if len(xs) < 20:
                _t_outer_rim_total += _time.perf_counter() - _t0
                continue
            
            rim_pts = np.stack([xs + minc_full, ys + minr_full], axis=1).astype(np.float32)
            _t_outer_rim_total += _time.perf_counter() - _t0

            # Get center from downsampled centroid (scaled to full res)
            _t0 = _time.perf_counter()
            cx_ds, cy_ds = prop.centroid[1], prop.centroid[0]
            cx = cx_ds * DOWNSAMPLE_FACTOR
            cy = cy_ds * DOWNSAMPLE_FACTOR
            _t_regionprops_access += _time.perf_counter() - _t0
        else:
            # === FULL RESOLUTION PATH ===
            _t0 = _time.perf_counter()
            minr, minc, maxr, maxc = prop.bbox
            
            if skip_prop_image:
                # OPTIMIZED: Avoid prop.image by slicing labels directly
                # prop.image creates a new boolean array, this just references existing data
                label_val = prop.label
                region_mask = (labels[minr:maxr, minc:maxc] == label_val)
            else:
                # Original: use prop.image (slower but guaranteed correct)
                region_mask = prop.image
            _t_regionprops_access += _time.perf_counter() - _t0
            
            _t0 = _time.perf_counter()
            rim_pts = outer_rim_points(region_mask, thickness=max(2, int(0.03 * np.sqrt(area_full))))
            _t_outer_rim_total += _time.perf_counter() - _t0

            if len(rim_pts) < 20: continue

            _t0 = _time.perf_counter()
            rim_pts[:, 0] += minc
            rim_pts[:, 1] += minr

            probs = rim_prob[rim_pts[:, 1].astype(int), rim_pts[:, 0].astype(int)]
            rim_pts = rim_pts[probs > rim_thresh]
            _t_misc_ops += _time.perf_counter() - _t0

            if len(rim_pts) < 20: continue

            _t0 = _time.perf_counter()
            cx, cy = prop.centroid[1], prop.centroid[0]
            _t_regionprops_access += _time.perf_counter() - _t0
        
        _t0 = _time.perf_counter()
        rim_pts = angular_sample(rim_pts, (cx, cy), bins=64)
        _t_angular_sample_total += _time.perf_counter() - _t0
        if len(rim_pts) < 16: continue

        _t0 = _time.perf_counter()
        pts = rim_pts.reshape(-1, 1, 2).astype(np.float32)
        _t_misc_ops += _time.perf_counter() - _t0
        
        _t0 = _time.perf_counter()
        try:
            if fit_method == 'geometric':
                (cx, cy), (w, h), angle = fit_ellipse_geometric(pts)
            elif fit_method == 'taubin':
                (cx, cy), (w, h), angle = fit_ellipse_taubin(pts)
            else:  # cv2 (default)
                (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
        except Exception as e:
            # Primary method failed - try cv2 as fallback
            try:
                (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
            except:
                # Both failed - use bbox
                cx, cy = prop.centroid[1], prop.centroid[0]
                w, h = prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]
                angle = 0.0
        _t_fit_ellipse_total += _time.perf_counter() - _t0

        if not valid_ellipse(w, h):
            w, h = prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]

        _t0 = _time.perf_counter()
        optimal_scale, rim_confidence = find_optimal_scale_from_rim(
            cx, cy, w, h, angle, rim_prob,
            scale_min=scale_min, scale_max=scale_max, steps=scale_steps, n_samples=n_samples_scale
        )
        _t_find_scale_total += _time.perf_counter() - _t0

        w_scaled = w * optimal_scale
        h_scaled = h * optimal_scale
        w_scaled, h_scaled = enforce_eccentricity_floor(w_scaled, h_scaled, ecc_floor)

        _t0 = _time.perf_counter()
        if use_grad_conf and grad_mag is not None:
            confidence, conf_details = compute_ellipse_confidence_with_gradient(
                cx, cy, w_scaled / 2.0, h_scaled / 2.0, angle, rim_prob, grad_mag, n_samples=n_samples_conf
            )
        else:
            confidence, conf_details = compute_ellipse_confidence(
                cx, cy, w_scaled / 2.0, h_scaled / 2.0, angle, rim_prob, global_prob, n_samples=n_samples_conf
            )
        _t_compute_conf_total += _time.perf_counter() - _t0
        
        if conf_details['angular_completeness'] < angular_completeness_thresh:
            continue
        
        _n_candidates += 1
        candidates.append({
            'x': cx,
            'y': cy,
            'a': w_scaled / 2.0,
            'b': h_scaled / 2.0,
            'angle': angle,
            '_scale': optimal_scale,
            '_confidence': confidence,
            '_rim_score': conf_details['rim_score'],
            '_angular_completeness': conf_details['angular_completeness'],
            '_global_coverage': conf_details['global_coverage'],
            "_source": "watershed",
        })

    _t_loop_end = _time.perf_counter()
    
    # Print timing breakdown (only if significant time spent)
    _total_time = _t_loop_end - _t_start
    if _total_time > 0.5:  # Only print if > 500ms
        print(f"  [EXTRACTION TIMING] Total: {_total_time*1000:.0f}ms | "
              f"mask:{(_t_mask-_t_start)*1000:.0f} label:{(_t_label-_t_mask)*1000:.0f} "
              f"EDT:{(_t_edt-_t_label)*1000:.0f} watershed:{(_t_watershed-_t_edt)*1000:.0f} "
              f"regionprops:{_t_regionprops_call*1000:.0f} | "
              f"regions:{_n_regions} candidates:{_n_candidates} | "
              f"propAccess:{_t_regionprops_access*1000:.0f} outer_rim:{_t_outer_rim_total*1000:.0f} "
              f"misc:{_t_misc_ops*1000:.0f} angular:{_t_angular_sample_total*1000:.0f} "
              f"fitEllipse:{_t_fit_ellipse_total*1000:.0f} findScale:{_t_find_scale_total*1000:.0f} "
              f"conf:{_t_compute_conf_total*1000:.0f}")

    # Sort candidates by confidence (descending) BEFORE selection
    candidates.sort(key=lambda x: x.get('_confidence', 0.0), reverse=True)

    # --- SIZE/VISIBILITY FILTER (Applied early, before NMS) ---
    if min_semi_axis > 0 or max_size_ratio < 1.0 or require_fully_visible:
        if image_shape is None:
            # Infer from pred_tensor if not provided
            h_img, w_img = pred_tensor.shape[1], pred_tensor.shape[2]
        else:
            h_img, w_img = image_shape
        
        S = min(w_img, h_img)
        filtered_candidates = []
        
        for c in candidates:
            # 1. Too Small: filter if semi-minor axis < min_semi_axis
            semi_minor = min(c['a'], c['b'])
            if semi_minor < min_semi_axis:
                continue
            
            # Calculate BBox for size and visibility checks
            center = (c['x'], c['y'])
            size = (c['a'] * 2, c['b'] * 2)
            angle = c['angle']
            rect = (center, size, angle)
            box = cv2.boxPoints(rect)
            
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])
            
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            # 2. Too Large: filter if (bbox_w + bbox_h) >= max_size_ratio * S
            if max_size_ratio < 1.0:
                if (bbox_w + bbox_h) >= (max_size_ratio * S):
                    continue
            
            # 3. Not Fully Visible: filter if bbox extends outside image
            if require_fully_visible:
                if x_min < 0 or y_min < 0 or x_max > w_img or y_max > h_img:
                    continue
            
            filtered_candidates.append(c)
        
        candidates = filtered_candidates

    # === RETURN ALL CANDIDATES (for ranking data generation) ===
    if return_all_candidates:
        # Return all candidates sorted by confidence - skip adaptive selection
        # This is essential for generating ranking training data with hard negatives
        return candidates

    # Primary Selection (Standard Confidence)
    step = 0.01
    current_thresh = min_confidence
    
    final_predictions = []
    
    # Iteratively lower threshold until at least 10 craters found
    while current_thresh >= (min_confidence_min - 1e-6):
        final_predictions = [c for c in candidates if c['_confidence'] >= current_thresh]
        
        if len(final_predictions) >= 10:
            break
            
        current_thresh -= step
                
    return final_predictions


def extract_craters_cv2_adaptive_selection_polar(pred_tensor,
                                  threshold=0.85,
                                  rim_thresh=0.25,
                                  ecc_floor=0.15,
                                  scale_min=1.0,
                                  scale_max=1.4,
                                  scale_steps=10,
                                  min_confidence=0.97,
                                  min_confidence_min=0.90,
                                  nms_filter=False,
                                  enable_rescue=False,
                                  enable_completeness=False,
                                  fit_method='polar',  # Default to polar
                                  min_semi_axis=0,
                                  max_size_ratio=1.0,
                                  require_fully_visible=False,
                                  image_shape=None,
                                  downsample_factor=1.0,
                                  skip_prop_image=False,
                                  polar_num_bins=120,
                                  polar_min_support=0.4,
                                  polar_refine_center=True,
                                  polar_max_center_shift=15.0,
                                  no_fallback=False):
    """
    Crater extraction with POLAR ellipse fitting for improved accuracy.
    
    This variant uses angular-domain ellipse fitting from polar.py which:
    1. Uses relative radial error (scale-free, matches dGA scorer)
    2. Explicitly tracks angular coverage/completeness  
    3. Enforces physical constraint: outer rim must be outside inner core
    4. Uses Huber loss for robustness to outlier rim points
    5. Has axis ratio regularization to prevent degenerate (near-circular) fits
    
    The polar approach is fundamentally better for craters because they are
    radially symmetric features. Thinking in polar coordinates naturally captures
    angular completeness and rim-core consistency.
    
    Args:
        fit_method: Ellipse fitting method. Options:
            - 'polar': Polar/angular domain fitting (default, recommended)
            - 'cv2': OpenCV's fitEllipse (fast, algebraic fallback)
            - 'geometric': Geometric fitting (slower but accurate fallback)
        polar_num_bins: Number of angular bins for polar representation (default 120)
        polar_min_support: Minimum fraction of angular bins with valid data (default 0.4)
        polar_refine_center: If True, jointly optimize center with ellipse (default True)
        polar_max_center_shift: Maximum center shift in pixels (default 15.0)
        no_fallback: If True, skip candidates where polar fitting fails (no cv2 fallback). Default False.
        [Other args same as extract_craters_cv2_adaptive_selection_rescue]
    """
    from polar import fit_crater_ellipse_polar
    import time as _time
    _t_start = _time.perf_counter()
    
    # Keep full-resolution data for precise ellipse fitting
    rim_prob = pred_tensor[2]
    global_prob = pred_tensor[1]
    core_prob = pred_tensor[0]  # Needed for polar fitting
    H_full, W_full = rim_prob.shape
    
    # === SEGMENTATION (optionally downsampled) ===
    DOWNSAMPLE_FACTOR = downsample_factor
    
    # Create masks
    mask_core_full = (pred_tensor[0] > threshold).astype(np.uint8)
    mask_global_full = (pred_tensor[1] > threshold).astype(np.uint8)
    
    if DOWNSAMPLE_FACTOR > 1.0:
        H_ds = int(H_full / DOWNSAMPLE_FACTOR)
        W_ds = int(W_full / DOWNSAMPLE_FACTOR)
        mask_core = cv2.resize(mask_core_full, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
        mask_global = cv2.resize(mask_global_full, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
    else:
        mask_core = mask_core_full
        mask_global = mask_global_full
    
    _t_mask = _time.perf_counter()

    markers = measure.label(mask_core, background=0)
    _t_label = _time.perf_counter()
    
    distance = ndi.distance_transform_edt(mask_global)
    _t_edt = _time.perf_counter()
    
    labels = segmentation.watershed(-distance, markers, mask=mask_global)
    _t_watershed = _time.perf_counter()

    candidates = []
    _t_outer_rim_total = 0.0
    _t_angular_sample_total = 0.0
    _t_fit_ellipse_total = 0.0
    _t_find_scale_total = 0.0
    _t_compute_conf_total = 0.0
    _t_polar_fit_total = 0.0
    _n_regions = 0
    _n_candidates = 0
    _n_polar_success = 0
    _n_polar_fallback = 0
    _t_regionprops_access = 0.0
    _t_misc_ops = 0.0

    _t0 = _time.perf_counter()
    all_props = measure.regionprops(labels)
    _t_regionprops_call = _time.perf_counter() - _t0

    for prop in all_props:
        _n_regions += 1
        
        # === EARLY REJECTION WITH CHEAP PROPERTIES ===
        _t0 = _time.perf_counter()
        area_ds = prop.area
        area_full = area_ds * (DOWNSAMPLE_FACTOR ** 2)
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if area_full < 40:
            continue
        if area_full > 100000:
            continue
        _t0 = _time.perf_counter()
        minor_axis_ds = prop.minor_axis_length
        minor_axis_full = minor_axis_ds * DOWNSAMPLE_FACTOR
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if minor_axis_full < max(3, 0.02 * np.sqrt(area_full)):
            continue
        
        _t0 = _time.perf_counter()
        ecc = prop.eccentricity
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if ecc > 0.95 and area_full < 200:
            continue
        
        _t0 = _time.perf_counter()
        sol = prop.solidity
        _t_regionprops_access += _time.perf_counter() - _t0
        
        if sol < 0.7 and area_full < 150:
            continue

        # === EXTRACT RIM AND CORE POINTS FOR POLAR FITTING ===
        if DOWNSAMPLE_FACTOR > 1.0:
            # Map bbox to full resolution
            _t0 = _time.perf_counter()
            minr_ds, minc_ds, maxr_ds, maxc_ds = prop.bbox
            pad = 5
            minr_full = int(max(0, minr_ds * DOWNSAMPLE_FACTOR - pad))
            minc_full = int(max(0, minc_ds * DOWNSAMPLE_FACTOR - pad))
            maxr_full = int(min(H_full, maxr_ds * DOWNSAMPLE_FACTOR + pad))
            maxc_full = int(min(W_full, maxc_ds * DOWNSAMPLE_FACTOR + pad))
            _t_regionprops_access += _time.perf_counter() - _t0
            
            _t0 = _time.perf_counter()
            rim_region = rim_prob[minr_full:maxr_full, minc_full:maxc_full]
            global_region = (global_prob[minr_full:maxr_full, minc_full:maxc_full] > threshold).astype(np.uint8)
            core_region = core_prob[minr_full:maxr_full, minc_full:maxc_full]
            
            # Rim points: high rim probability within global mask
            rim_mask = (rim_region > rim_thresh) & (global_region > 0)
            ys_rim, xs_rim = np.where(rim_mask)
            
            # Core points: high core probability (for inner-outer coupling)
            core_mask = core_region > threshold
            ys_core, xs_core = np.where(core_mask)
            
            if len(xs_rim) < 20:
                _t_outer_rim_total += _time.perf_counter() - _t0
                continue
            
            rim_pts = np.stack([xs_rim + minc_full, ys_rim + minr_full], axis=1).astype(np.float32)
            core_pts = np.stack([xs_core + minc_full, ys_core + minr_full], axis=1).astype(np.float32) if len(xs_core) > 0 else np.zeros((0, 2), dtype=np.float32)
            _t_outer_rim_total += _time.perf_counter() - _t0

            _t0 = _time.perf_counter()
            cx_ds, cy_ds = prop.centroid[1], prop.centroid[0]
            cx = cx_ds * DOWNSAMPLE_FACTOR
            cy = cy_ds * DOWNSAMPLE_FACTOR
            _t_regionprops_access += _time.perf_counter() - _t0
        else:
            # Full resolution path
            _t0 = _time.perf_counter()
            minr, minc, maxr, maxc = prop.bbox
            
            if skip_prop_image:
                label_val = prop.label
                region_mask = (labels[minr:maxr, minc:maxc] == label_val)
            else:
                region_mask = prop.image
            _t_regionprops_access += _time.perf_counter() - _t0
            
            _t0 = _time.perf_counter()
            rim_pts = outer_rim_points(region_mask, thickness=max(2, int(0.03 * np.sqrt(area_full))))
            _t_outer_rim_total += _time.perf_counter() - _t0

            if len(rim_pts) < 20: continue

            _t0 = _time.perf_counter()
            rim_pts[:, 0] += minc
            rim_pts[:, 1] += minr

            probs = rim_prob[rim_pts[:, 1].astype(int), rim_pts[:, 0].astype(int)]
            rim_pts = rim_pts[probs > rim_thresh]
            _t_misc_ops += _time.perf_counter() - _t0

            if len(rim_pts) < 20: continue
            
            # Extract core points for polar fitting
            _t0 = _time.perf_counter()
            core_region = core_prob[minr:maxr, minc:maxc]
            core_mask = (core_region > threshold) & region_mask
            ys_core, xs_core = np.where(core_mask)
            core_pts = np.stack([xs_core + minc, ys_core + minr], axis=1).astype(np.float32) if len(xs_core) > 0 else np.zeros((0, 2), dtype=np.float32)
            _t_misc_ops += _time.perf_counter() - _t0

            _t0 = _time.perf_counter()
            cx, cy = prop.centroid[1], prop.centroid[0]
            _t_regionprops_access += _time.perf_counter() - _t0
        
        # === POLAR ELLIPSE FITTING ===
        _t0 = _time.perf_counter()
        polar_result = None
        
        if fit_method == 'polar':
            try:
                polar_result = fit_crater_ellipse_polar(
                    rim_xs=rim_pts[:, 0],
                    rim_ys=rim_pts[:, 1],
                    core_xs=core_pts[:, 0] if len(core_pts) > 0 else np.array([cx]),
                    core_ys=core_pts[:, 1] if len(core_pts) > 0 else np.array([cy]),
                    center=(cx, cy),
                    num_bins=polar_num_bins,
                    min_support=polar_min_support,
                    refine_center=polar_refine_center,
                    max_center_shift=polar_max_center_shift
                )
            except Exception as e:
                import traceback
                print(f'POLAR EXCEPTION: {type(e).__name__}: {e}')
                traceback.print_exc()
                polar_result = None

        
        polar_succeeded = False
        if polar_result is not None:
            # Polar fit succeeded - it already optimized the ellipse to match rim
            cx, cy, a, b, phi = polar_result
            w = a * 2.0
            h = b * 2.0
            angle = np.degrees(phi)
            _n_polar_success += 1
            polar_succeeded = True
        else:
            # Polar fit failed
            if no_fallback:
                # Skip this candidate entirely (no cv2 fallback)
                _t_polar_fit_total += _time.perf_counter() - _t0
                continue
            
            # Fallback to cv2 fitting
            _t0_angular = _time.perf_counter()
            rim_pts_sampled = angular_sample(rim_pts, (cx, cy), bins=64)
            _t_angular_sample_total += _time.perf_counter() - _t0_angular
            
            if len(rim_pts_sampled) < 16:
                _t_fit_ellipse_total += _time.perf_counter() - _t0
                continue
            
            pts = rim_pts_sampled.reshape(-1, 1, 2).astype(np.float32)
            
            try:
                if fit_method == 'geometric':
                    (cx, cy), (w, h), angle = fit_ellipse_geometric(pts)
                elif fit_method == 'taubin':
                    (cx, cy), (w, h), angle = fit_ellipse_taubin(pts)
                else:
                    (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
            except Exception:
                try:
                    (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
                except:
                    cx, cy = prop.centroid[1], prop.centroid[0]
                    w, h = prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]
                    angle = 0.0
            _n_polar_fallback += 1
        
        _t_polar_fit_total += _time.perf_counter() - _t0

        if not valid_ellipse(w, h):
            w, h = prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]

        # Scale search is needed for ALL fits (including polar) because:
        # - Polar fits to thresholded rim mask points (inner boundary of rim)
        # - Scale search finds where rim probability is MAXIMUM (rim crest)
        _t0 = _time.perf_counter()
        optimal_scale, rim_confidence = find_optimal_scale_from_rim(
            cx, cy, w, h, angle, rim_prob,
            scale_min=scale_min, scale_max=scale_max, steps=scale_steps
        )
        _t_find_scale_total += _time.perf_counter() - _t0

        w_scaled = w * optimal_scale
        h_scaled = h * optimal_scale
        w_scaled, h_scaled = enforce_eccentricity_floor(w_scaled, h_scaled, ecc_floor)

        _t0 = _time.perf_counter()
        confidence, conf_details = compute_ellipse_confidence(
            cx, cy, w_scaled / 2.0, h_scaled / 2.0, angle, rim_prob, global_prob
        )
        _t_compute_conf_total += _time.perf_counter() - _t0
        
        if conf_details['angular_completeness'] < 0.35:
            continue
            
        confidence_rescue = confidence
        
        if enable_rescue:
            radius = (w_scaled + h_scaled) / 4.0
            if radius > 15: 
                 if conf_details['global_coverage'] > 0.6:
                     confidence_rescue = max(confidence_rescue, conf_details['global_coverage'] * 0.9)
                     
        if enable_completeness:
            if conf_details['angular_completeness'] > 0.7:
                 confidence_rescue = max(confidence_rescue, (confidence + conf_details['angular_completeness']) / 2.0)
        
        _n_candidates += 1
        candidates.append({
            'x': cx,
            'y': cy,
            'a': w_scaled / 2.0,
            'b': h_scaled / 2.0,
            'angle': angle,
            '_scale': optimal_scale,
            '_confidence': confidence,
            '_confidence_rescue': confidence_rescue,
            '_rim_score': conf_details['rim_score'],
            '_angular_completeness': conf_details['angular_completeness'],
            '_global_coverage': conf_details['global_coverage'],
            "_source": "watershed_polar" if polar_result else "watershed_cv2",
        })

    _t_loop_end = _time.perf_counter()
    
    # Print timing breakdown
    _total_time = _t_loop_end - _t_start
    if _total_time > 0.5:
        print(f"  [EXTRACTION POLAR TIMING] Total: {_total_time*1000:.0f}ms | "
              f"mask:{(_t_mask-_t_start)*1000:.0f} label:{(_t_label-_t_mask)*1000:.0f} "
              f"EDT:{(_t_edt-_t_label)*1000:.0f} watershed:{(_t_watershed-_t_edt)*1000:.0f} "
              f"regionprops:{_t_regionprops_call*1000:.0f} | "
              f"regions:{_n_regions} candidates:{_n_candidates} | "
              f"polar_ok:{_n_polar_success} polar_fallback:{_n_polar_fallback} | "
              f"propAccess:{_t_regionprops_access*1000:.0f} outer_rim:{_t_outer_rim_total*1000:.0f} "
              f"misc:{_t_misc_ops*1000:.0f} angular:{_t_angular_sample_total*1000:.0f} "
              f"polarFit:{_t_polar_fit_total*1000:.0f} findScale:{_t_find_scale_total*1000:.0f} "
              f"conf:{_t_compute_conf_total*1000:.0f}")

    # Sort candidates by confidence (descending)
    candidates.sort(key=lambda x: x.get('_confidence', 0.0), reverse=True)

    # --- SIZE/VISIBILITY FILTER ---
    if min_semi_axis > 0 or max_size_ratio < 1.0 or require_fully_visible:
        if image_shape is None:
            h_img, w_img = pred_tensor.shape[1], pred_tensor.shape[2]
        else:
            h_img, w_img = image_shape
        
        S = min(w_img, h_img)
        filtered_candidates = []
        
        for c in candidates:
            semi_minor = min(c['a'], c['b'])
            if semi_minor < min_semi_axis:
                continue
            
            center = (c['x'], c['y'])
            size = (c['a'] * 2, c['b'] * 2)
            angle = c['angle']
            rect = (center, size, angle)
            box = cv2.boxPoints(rect)
            
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])
            
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            if max_size_ratio < 1.0:
                if (bbox_w + bbox_h) >= (max_size_ratio * S):
                    continue
            
            if require_fully_visible:
                if x_min < 0 or y_min < 0 or x_max > w_img or y_max > h_img:
                    continue
            
            filtered_candidates.append(c)
        
        candidates = filtered_candidates

    # --- NMS / Overlap Filter ---
    if nms_filter:
        accepted = []
        for cand in candidates:
            is_bad = False
            r_cand = math.sqrt(cand['a'] * cand['b'])
            
            for kept in accepted:
                dx = cand['x'] - kept['x']
                dy = cand['y'] - kept['y']
                dist = math.hypot(dx, dy)
                
                r_kept = math.sqrt(kept['a'] * kept['b'])
                
                if dist < abs(r_cand - r_kept):
                    is_bad = True
                    break
                    
                if dist < (r_cand + r_kept) * 0.80:
                    is_bad = True
                    break
            
            if not is_bad:
                accepted.append(cand)
        
        candidates = accepted

    # Selection logic
    step = 0.01
    current_thresh = min_confidence
    final_predictions = []
    
    while current_thresh >= (min_confidence_min - 1e-6):
        final_predictions = [c for c in candidates if c['_confidence'] >= current_thresh]
        
        if len(final_predictions) >= 10:
            break
            
        current_thresh -= step

    # Rescue Fallback
    if len(final_predictions) < 10 and (enable_rescue or enable_completeness):
        print(f"  [DEBUG POLAR] Rescue Activated! Found {len(final_predictions)} < 10. Checking rescue candidates...")
        
        current_ids = [(c['x'], c['y']) for c in final_predictions]
        remaining = [c for c in candidates if (c['x'], c['y']) not in current_ids]
        remaining.sort(key=lambda x: x.get('_confidence_rescue', 0.0), reverse=True)
        
        print(f"  [DEBUG POLAR] {len(remaining)} candidates available for rescue.")
        
        added_count = 0
        for cand in remaining:
            if len(final_predictions) >= 10:
                break
                
            if cand['_confidence_rescue'] > 0.6:
                 print(f"    [DEBUG POLAR] Cand (r={math.sqrt(cand['a']*cand['b']):.1f}): Conf={cand['_confidence']:.3f} -> Rescue={cand['_confidence_rescue']:.3f} (Req: {min_confidence_min})")

            if cand['_confidence_rescue'] >= 0.85:
                final_predictions.append(cand)
                added_count += 1
                print(f"      -> ADDED via Rescue!")
                
        print(f"  [DEBUG POLAR] Added {added_count} craters via Rescue.")
                
    return final_predictions