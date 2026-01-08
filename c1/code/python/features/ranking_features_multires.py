import numpy as np
import cv2
from skimage.measure import regionprops, EllipseModel
import math
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def safe_div(a, b):
    return a / (b + 1e-6)

def ellipse_distance(pts, cx, cy, a, b, theta):
    angle = np.deg2rad(theta)
    ct, st = np.cos(angle), np.sin(angle)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy
    xr = ct * x + st * y
    yr = -st * x + ct * y

    a = max(a, 1e-4)
    b = max(b, 1e-4)
    return np.abs((xr / a) ** 2 + (yr / b) ** 2 - 1.0)


# ------------------------------------------------------------
# dGA Metric (Gaussian Angle distance for crater matching)
# ------------------------------------------------------------

def _calcYmat(a, b, phi):
    """Configuration matrix for dGA calculation."""
    unit_1 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    unit_2 = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
    unit_3 = np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])
    return unit_1 @ unit_2 @ unit_3


def _calc_dGA(Yi, Yj, yi, yj):
    """Compute Gaussian Angle distance."""
    detYi = np.linalg.det(Yi)
    detYj = np.linalg.det(Yj)
    detSum = np.linalg.det(Yi + Yj)
    
    if detSum < 1e-12:
        return 1.57  # Pi/2
    
    multiplicand = 4 * np.sqrt(detYi * detYj) / detSum
    diff = yi - yj
    
    try:
        invSum = np.linalg.inv(Yi + Yj)
    except np.linalg.LinAlgError:
        return 1.57
    
    exponent = (-0.5 * diff.T @ Yi @ invSum @ Yj @ diff)
    e = exponent[0, 0]
    cos = multiplicand * np.exp(e)
    cos = max(-1, min(1, cos))
    return np.arccos(cos)


def dGA_metric(crater_A, crater_B):
    """
    Calculate dGA and xi_2 between two craters.
    
    Args:
        crater_A, crater_B: dicts with keys {'x', 'y', 'a', 'b', 'angle'}
        
    Returns:
        tuple: (dGA_val, xi_2)
    """
    A_a, A_b = crater_A['a'], crater_A['b']
    A_xc, A_yc = crater_A['x'], crater_A['y']
    A_phi = np.deg2rad(crater_A['angle'])
    
    B_a, B_b = crater_B['a'], crater_B['b']
    B_xc, B_yc = crater_B['x'], crater_B['y']
    B_phi = np.deg2rad(crater_B['angle'])
    
    A_Y = _calcYmat(A_a, A_b, A_phi)
    B_Y = _calcYmat(B_a, B_b, B_phi)
    
    A_y = np.array([[A_xc], [A_yc]])
    B_y = np.array([[B_xc], [B_yc]])
    
    dGA_val = _calc_dGA(A_Y, B_Y, A_y, B_y)
    
    # Xi2 logic from scorer
    XI_2_THRESH = 13.277
    NN_PIX_ERR_RATIO = 0.07
    
    ab_min = min(A_a, A_b)
    comparison_sig = NN_PIX_ERR_RATIO * ab_min
    ref_sig = 0.85 / (np.sqrt(A_a * A_b) + 1e-6) * comparison_sig
    
    xi_2 = (dGA_val * dGA_val) / (ref_sig * ref_sig + 1e-12)
    
    return dGA_val, xi_2


def compute_gt_ranking_label(pred_ellipse_dict, gt_list):
    """
    Compute the scorer-aligned ranking label for a single prediction.
    
    Args:
        pred_ellipse_dict: dict with keys {'x', 'y', 'a', 'b', 'angle'}
        gt_list: list of dicts with same keys + optional 'class' or 'crater_classification'
        
    Returns:
        tuple: (score_label, best_xi2, matched_class_label)
        - score_label: float [0, 1] where 1 = perfect match
        - best_xi2: float (inf if no match)
        - matched_class_label: int class of matched GT, or -1 if no match
    """
    pred = {
        'x': pred_ellipse_dict['x'],
        'y': pred_ellipse_dict['y'],
        'a': pred_ellipse_dict['a'],
        'b': pred_ellipse_dict['b'],
        'angle': pred_ellipse_dict['angle']
    }
    
    best_dGA = math.pi / 2
    best_xi2 = float('inf')
    best_cls = -1
    
    XI_2_THRESH = 13.277
    
    if not gt_list:
        return 0.0, best_xi2, best_cls
    
    for gt in gt_list:
        # Standardize GT keys
        if 'x' in gt:
            gt_std = {'x': gt['x'], 'y': gt['y'], 'a': gt['a'], 'b': gt['b'], 'angle': gt['angle']}
        else:
            gt_std = {
                'x': gt['ellipseCenterX(px)'],
                'y': gt['ellipseCenterY(px)'],
                'a': gt['ellipseSemimajor(px)'],
                'b': gt['ellipseSemiminor(px)'],
                'angle': gt['ellipseRotation(deg)']
            }
        
        # Quick rejection checks
        rA = min(gt_std['a'], gt_std['b'])
        rB = min(pred['a'], pred['b'])
        
        if rA > 1.5 * rB or rB > 1.5 * rA:
            continue
        r = min(rA, rB)
        if abs(gt_std['x'] - pred['x']) > r:
            continue
        if abs(gt_std['y'] - pred['y']) > r:
            continue
        
        d, xi2 = dGA_metric(gt_std, pred)
        
        if d < best_dGA:
            best_dGA = d
            best_xi2 = xi2
            best_cls = gt.get('class', gt.get('crater_classification', 4))
    
    if best_xi2 < XI_2_THRESH:
        label = max(0.0, 1.0 - best_dGA / math.pi)
        return label, best_xi2, best_cls
    else:
        return 0.0, best_xi2, -1



# ------------------------------------------------------------
# Geometry features (SCALE SAFE)
# ------------------------------------------------------------

def geometry_features(rim_pts, ellipse, prop, img_shape):
    cx, cy, a, b, theta = ellipse
    h, w = img_shape
    img_area = h * w
    img_diag = math.hypot(h, w)

    d = ellipse_distance(rim_pts, cx, cy, a, b, theta)

    angles = np.mod(np.arctan2(rim_pts[:, 1] - cy, rim_pts[:, 0] - cx), 2 * np.pi)
    bins = np.histogram(angles, bins=36, range=(0, 2*np.pi))[0]

    if len(angles) > 1:
        sa = np.sort(angles)
        gaps = np.diff(np.concatenate([sa, sa[:1] + 2*np.pi]))
        max_gap = gaps.max() / (2*np.pi)
    else:
        max_gap = 1.0

    ellipse_area = np.pi * a * b

    return {
        "geometry_eccentricity": np.sqrt(max(0, 1 - (b / (a + 1e-6)) ** 2)),
        "geometry_axis_ratio": b / (a + 1e-6),

        # 🔧 normalized
        "geometry_ellipse_area": ellipse_area / img_area,
        "geometry_area_ratio": prop.area / (ellipse_area + 1e-6),

        "geometry_resid_rms": np.sqrt(np.mean(d ** 2)) if len(d) else 1.0,
        "geometry_resid_p90": np.percentile(d, 90) if len(d) else 1.0,

        "geometry_support_ratio":
            len(rim_pts) / (2 * np.pi * np.sqrt((a*a + b*b)/2) + 1e-6),

        "geometry_angular_coverage": np.count_nonzero(bins) / len(bins),
        "geometry_angular_std": np.std(bins) / (len(rim_pts) + 1e-6),  # normalized
        "geometry_max_gap": max_gap,
        "geometry_condition": a / (b + 1e-6)
    }


def geometry_features_fast(rim_pts, ellipse, img_shape):
    """
    FAST version: No regionprops (prop) required.
    Skips geometry_area_ratio which needs prop.area.
    """
    cx, cy, a, b, theta = ellipse
    h, w = img_shape
    img_area = h * w

    d = ellipse_distance(rim_pts, cx, cy, a, b, theta)

    angles = np.mod(np.arctan2(rim_pts[:, 1] - cy, rim_pts[:, 0] - cx), 2 * np.pi)
    bins = np.histogram(angles, bins=36, range=(0, 2*np.pi))[0]

    if len(angles) > 1:
        sa = np.sort(angles)
        gaps = np.diff(np.concatenate([sa, sa[:1] + 2*np.pi]))
        max_gap = gaps.max() / (2*np.pi)
    else:
        max_gap = 1.0

    ellipse_area = np.pi * a * b

    return {
        "geometry_eccentricity": np.sqrt(max(0, 1 - (b / (a + 1e-6)) ** 2)),
        "geometry_axis_ratio": b / (a + 1e-6),
        "geometry_ellipse_area": ellipse_area / img_area,
        "geometry_area_ratio": 1.0,  # Placeholder (no prop available)
        "geometry_resid_rms": np.sqrt(np.mean(d ** 2)) if len(d) else 1.0,
        "geometry_resid_p90": np.percentile(d, 90) if len(d) else 1.0,
        "geometry_support_ratio":
            len(rim_pts) / (2 * np.pi * np.sqrt((a*a + b*b)/2) + 1e-6),
        "geometry_angular_coverage": np.count_nonzero(bins) / len(bins),
        "geometry_angular_std": np.std(bins) / (len(rim_pts) + 1e-6),
        "geometry_max_gap": max_gap,
        "geometry_condition": a / (b + 1e-6)
    }

# ------------------------------------------------------------
# Rim probability features (SAFE)
# ------------------------------------------------------------

def rim_prob_features(rim_pts, rim_prob):
    h, w = rim_prob.shape
    xs = np.clip(np.round(rim_pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(rim_pts[:, 1]).astype(int), 0, h - 1)

    probs = np.clip(rim_prob[ys, xs], 1e-6, 1 - 1e-6)

    return {
        "rim_prob_mean": probs.mean(),
        "rim_prob_std": probs.std(),
        "rim_prob_min": probs.min(),
        "rim_prob_p20": np.percentile(probs, 20),
        "rim_prob_p80": np.percentile(probs, 80),
        "rim_prob_frac_above_50": np.mean(probs > 0.5),
        "rim_prob_frac_above_70": np.mean(probs > 0.7),
        "rim_prob_entropy": -np.mean(probs * np.log(probs))
    }


# ------------------------------------------------------------
# Polar-inspired features (NEW - high value)
# ------------------------------------------------------------

def polar_features(rim_pts, ellipse, rim_prob):
    """
    Features inspired by polar.py analysis.
    These capture signal not in standard geometry features:
    - Directional asymmetry (opposing angles)
    - Roughness (high-frequency residual energy)
    - Angular support fraction (rim completeness)
    
    FIXED: Now properly transforms to ellipse-local coordinates before
    computing polar features.
    """
    cx, cy, a, b, theta = ellipse
    theta_rad = np.deg2rad(theta)
    num_bins = 36
    
    if len(rim_pts) < 5:
        return {
            "polar_angular_support": 0.0,
            "polar_mean_asymmetry": 1.0,
            "polar_max_asymmetry": 1.0,
            "polar_roughness_ratio": 1.0
        }
    
    # Transform rim points to ellipse-local coordinates (centered, unrotated)
    dx = rim_pts[:, 0] - cx
    dy = rim_pts[:, 1] - cy
    
    # Rotate to align with ellipse axes (undo ellipse rotation)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    dx_local = ct * dx + st * dy
    dy_local = -st * dx + ct * dy
    
    # Compute angles and radii in local frame
    angles_local = np.arctan2(dy_local, dx_local)  # [-π, π]
    radii = np.sqrt(dx_local**2 + dy_local**2)
    
    # Bin into angular sectors
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_idx = np.digitize(angles_local, bins) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)  # Handle edge cases
    
    # For each bin, take max radius (outermost rim point)
    r_theta = np.full(num_bins, np.nan)
    for i in range(num_bins):
        mask = bin_idx == i
        if np.any(mask):
            r_theta[i] = np.max(radii[mask])
    
    valid = ~np.isnan(r_theta)
    
    # Feature 1: Angular support fraction (rim completeness)
    angular_support = float(valid.mean())
    
    # Feature 2: Directional asymmetry (opposing angles should have similar r)
    # For perfect ellipse, r(θ) = r(θ + π) due to symmetry
    half = num_bins // 2
    asymmetries = []
    for i in range(half):
        opposite_idx = i + half
        if not np.isnan(r_theta[i]) and not np.isnan(r_theta[opposite_idx]):
            # Normalize by expected radius to make scale-invariant
            theta_i = -np.pi + (i + 0.5) * (2 * np.pi / num_bins)
            r_expected = (a * b) / (np.sqrt((b * np.cos(theta_i))**2 + (a * np.sin(theta_i))**2) + 1e-9)
            
            avg = (r_theta[i] + r_theta[opposite_idx]) / 2
            diff = abs(r_theta[i] - r_theta[opposite_idx])
            asymmetries.append(diff / (r_expected + 1e-6))
    
    mean_asymmetry = float(np.mean(asymmetries)) if asymmetries else 1.0
    max_asymmetry = float(np.max(asymmetries)) if asymmetries else 1.0
    
    # Feature 3: Roughness (deviation from expected ellipse radius)
    theta_centers = 0.5 * (bins[:-1] + bins[1:])  # Bin centers
    if valid.sum() >= 10:
        # Predicted radius at each bin center using ellipse equation
        # r = ab / sqrt((b*cos(θ))² + (a*sin(θ))²)
        cos_tc = np.cos(theta_centers[valid])
        sin_tc = np.sin(theta_centers[valid])
        r_pred = (a * b) / (np.sqrt((b * cos_tc)**2 + (a * sin_tc)**2) + 1e-9)
        
        residuals = r_theta[valid] - r_pred
        
        # Normalize residuals by expected radius
        residuals_norm = residuals / (r_pred + 1e-6)
        
        # Roughness = gradient energy (high freq) / total energy
        if len(residuals_norm) > 2:
            grad = np.diff(residuals_norm)
            roughness_ratio = float(np.sum(grad**2) / (np.sum(residuals_norm**2) + 1e-10))
        else:
            roughness_ratio = 0.0
    else:
        roughness_ratio = 1.0
    
    return {
        "polar_angular_support": angular_support,
        "polar_mean_asymmetry": mean_asymmetry,
        "polar_max_asymmetry": max_asymmetry,
        "polar_roughness_ratio": roughness_ratio
    }

# ------------------------------------------------------------
# Morphology features (SCALE FIXED)
# ------------------------------------------------------------

def morphology_features(prop, img_shape):
    h, w = img_shape
    img_area = h * w

    area = prop.area
    perim = prop.perimeter

    return {
        "morph_solidity": prop.solidity,
        "morph_convexity": safe_div(prop.area, prop.convex_area),

        "morph_extent": prop.extent,

        # 🔧 fixed
        "morph_perim_area_ratio": safe_div(perim, np.sqrt(area)),
        "morph_minor_axis": safe_div(prop.minor_axis_length, np.sqrt(area)),
        "morph_area_log": np.log(area / img_area + 1e-6)
    }

# ------------------------------------------------------------
# Stability features (CRITICAL FIX)
# ------------------------------------------------------------

def stability_features(rim_pts, ellipse):
    cx, cy, a, b, theta = ellipse

    if len(rim_pts) < 10:
        return dict.fromkeys([
            "stab_jitter_a", "stab_jitter_b",
            "stab_jitter_theta", "stab_center_shift",
            "stab_iou_subsample"
        ], 1.0)

    shifts_a, shifts_b, shifts_th, shifts_c = [], [], [], []

    for _ in range(5):
        idx = np.random.choice(len(rim_pts), int(0.8 * len(rim_pts)), replace=False)
        pts = rim_pts[idx]

        model = EllipseModel()
        if not model.estimate(pts):
            continue

        cxx, cyy, aa, bb, tt = model.params
        if aa < bb:
            aa, bb = bb, aa
            tt += np.pi / 2

        tt = np.mod(tt, np.pi)
        theta0 = np.mod(np.deg2rad(theta), np.pi)

        shifts_a.append(abs(aa - a) / (a + 1e-6))
        shifts_b.append(abs(bb - b) / (b + 1e-6))
        shifts_th.append(min(abs(tt - theta0), np.pi - abs(tt - theta0)))

        # 🔧 normalized center shift
        shifts_c.append(
            np.hypot(cxx - cx, cyy - cy) / (np.sqrt(a * b) + 1e-6)
        )

    if not shifts_a:
        return dict.fromkeys([
            "stab_jitter_a", "stab_jitter_b",
            "stab_jitter_theta", "stab_center_shift",
            "stab_iou_subsample"
        ], 1.0)

    ma = np.mean(shifts_a)
    mb = np.mean(shifts_b)

    return {
        "stab_jitter_a": ma,
        "stab_jitter_b": mb,
        "stab_jitter_theta": np.mean(shifts_th),
        "stab_center_shift": np.mean(shifts_c),
        "stab_iou_subsample": 1 / (1 + ma + mb)
    }

# ------------------------------------------------------------
# Context features (GRADIENT NORMALIZED)
# ------------------------------------------------------------

def context_features(mask, img, rim_pts, precomputed_grads=None):
    if img is None:
        return dict.fromkeys([
            "ctx_inside_mean", "ctx_outside_mean",
            "ctx_diff", "ctx_grad_mean", "ctx_grad_std"
        ], 0.0)

    mask = mask.astype(bool)
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), kernel)
    outside = (dil > 0) & (~mask)

    inside_vals = img[mask]
    outside_vals = img[outside]

    if precomputed_grads is not None:
        grads = precomputed_grads
    else:
        gy, gx = np.gradient(img)
        grads = np.hypot(gx, gy)

    xs = np.clip(np.round(rim_pts[:, 0]).astype(int), 0, img.shape[1]-1)
    ys = np.clip(np.round(rim_pts[:, 1]).astype(int), 0, img.shape[0]-1)
    rim_grads = grads[ys, xs]

    g_mean = np.mean(grads) + 1e-6
    g_std = np.std(grads) + 1e-6

    return {
        "ctx_inside_mean": inside_vals.mean() if inside_vals.size else 0,
        "ctx_outside_mean": outside_vals.mean() if outside_vals.size else 0,
        "ctx_diff": (
            inside_vals.mean() - outside_vals.mean()
            if inside_vals.size and outside_vals.size else 0
        ),
        "ctx_grad_mean": rim_grads.mean() / g_mean if rim_grads.size else 0,
        "ctx_grad_std": rim_grads.std() / g_std if rim_grads.size else 0
    }

# ------------------------------------------------------------
# Mahanti features (NORMALIZED, SAFE)
# ------------------------------------------------------------

def mahanti_features(rim_pts, ellipse, img):
    if img is None:
        return dict.fromkeys([
            "mahanti_median_slope",
            "mahanti_slope_std",
            "mahanti_depth_ratio",
            "mahanti_rim_sharpness"
        ], 0.0)

    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)

    xs = np.clip(np.round(rim_pts[:, 0]).astype(int), 0, img.shape[1]-1)
    ys = np.clip(np.round(rim_pts[:, 1]).astype(int), 0, img.shape[0]-1)

    rim_grad = grad[ys, xs]
    norm = np.mean(grad) + 1e-6

    return {
        "mahanti_median_slope": np.median(rim_grad) / norm,
        "mahanti_slope_std": np.std(rim_grad) / norm,
        "mahanti_depth_ratio":
            (rim_grad.max() - rim_grad.min()) / norm if rim_grad.size else 0,
        "mahanti_rim_sharpness": rim_grad.mean() / norm if rim_grad.size else 0
    }

# ------------------------------------------------------------
# Main feature extractor
# ------------------------------------------------------------

def extract_crater_features(
    rim_pts, ellipse, prop, rim_prob, img, mask,
    fast_mode=False, precomputed_grads=None
):
    h, w = img.shape if img is not None else mask.shape

    feats = {}
    feats.update(geometry_features(rim_pts, ellipse, prop, (h, w)))
    feats.update(rim_prob_features(rim_pts, rim_prob))
    #feats.update(polar_features(rim_pts, ellipse, rim_prob))  # NEW polar features
    #feats.update(morphology_features(prop, (h, w)))

    if not fast_mode:
        #feats.update(stability_features(rim_pts, ellipse))
        feats.update(mahanti_features(rim_pts, ellipse, img))
    else:
        feats.update(dict.fromkeys([
            "stab_jitter_a", "stab_jitter_b",
            "stab_jitter_theta", "stab_center_shift",
            "stab_iou_subsample",
            "mahanti_median_slope", "mahanti_slope_std",
            "mahanti_depth_ratio", "mahanti_rim_sharpness"
        ], 0.0))

    feats.update(context_features(mask, img, rim_pts, precomputed_grads))

    cx, cy, a, b, _ = ellipse
    feats["meta_log_radius"] = np.log(np.sqrt(a * b) / np.sqrt(h * w) + 1e-6)
    # Compute meta_log_area directly - don't depend on morphology_features being called
    feats["meta_log_area"] = feats.get("morph_area_log", np.log(np.pi * a * b / (h * w) + 1e-6))

    return feats


# ------------------------------------------------------------
# Helper: Extract rim points for a candidate (needed by eval3rect.py)
# ------------------------------------------------------------

def get_rim_points_for_candidate(rim_prob, ellipse, prob_thresh=0.2, dilate_iter=1):
    """
    Extract rim points for a candidate ellipse from the rim probability map.
    
    Args:
        rim_prob: (H, W) float array [0, 1]
        ellipse: (cx, cy, a, b, theta) tuple
        prob_thresh: Minimum probability to consider a point part of the rim
        dilate_iter: Dilation iterations to define the search region around the ellipse perimeter
    
    Returns:
        (N, 2) array of (x, y) coordinates
    """
    cx, cy, a, b, theta = ellipse
    h, w = rim_prob.shape
    
    # Create an ellipse mask (ring)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw simple ellipse for region of interest
    cv2.ellipse(mask, (int(cx), int(cy)), (int(a), int(b)), theta, 0, 360, 255, 3) 
    
    # Create the search band
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)
    
    # Select points where rim probability is high within the ellipse band
    candidates = (rim_prob > prob_thresh) & (mask_dilated > 0)
    
    y_idxs, x_idxs = np.where(candidates)
    
    if len(x_idxs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
        
    return np.column_stack([x_idxs, y_idxs]).astype(np.float32)


def get_rim_points_for_candidate_fast(rim_prob, ellipse, n_samples=72, band_width=5, prob_thresh=0.2):
    """
    FAST version: Uses parametric sampling instead of full mask operations.
    Samples n_samples points around the ellipse perimeter and finds high-prob rim points nearby.
    
    ~10x faster than the mask-based version for large images.
    
    Args:
        rim_prob: (H, W) float array [0, 1]
        ellipse: (cx, cy, a, b, theta) tuple
        n_samples: Number of angular samples around ellipse
        band_width: Search band width in pixels around ellipse perimeter
        prob_thresh: Minimum probability to accept a point
    
    Returns:
        (N, 2) array of (x, y) coordinates
    """
    cx, cy, a, b, theta = ellipse
    h, w = rim_prob.shape
    
    # Generate points along ellipse perimeter
    theta_rad = np.deg2rad(theta)
    angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    
    # Ellipse points (before rotation)
    ex = a * np.cos(angles)
    ey = b * np.sin(angles)
    
    # Rotate and translate
    px = cx + cos_t * ex - sin_t * ey
    py = cy + sin_t * ex + cos_t * ey
    
    # Sample rim_prob in a band around each point
    collected_pts = []
    
    for i in range(n_samples):
        x0, y0 = int(round(px[i])), int(round(py[i]))
        
        # Search in a small window
        for dx in range(-band_width, band_width + 1):
            for dy in range(-band_width, band_width + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < w and 0 <= y < h:
                    if rim_prob[y, x] > prob_thresh:
                        collected_pts.append((x, y))
    
    if len(collected_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # Remove duplicates efficiently
    pts = np.array(collected_pts, dtype=np.float32)
    # Unique rows
    pts = np.unique(pts, axis=0)
    
    return pts


def extract_crater_features_ultra_fast(rim_pts, ellipse, rim_prob, img_shape):
    """
    ULTRA FAST feature extraction: only geometry + rim_prob features.
    
    No regionprops, no mask operations, no stability bootstrap.
    Uses only the features that work without prop:
    - geometry_features_fast (no prop.area needed)
    - rim_prob_features
    
    ~5-10x faster than fast_mode. Enables ranking 50+ candidates.
    
    Args:
        rim_pts: (N, 2) array of rim point coordinates
        ellipse: (cx, cy, a, b, theta) tuple  
        rim_prob: (H, W) rim probability map
        img_shape: (h, w) tuple
    
    Returns:
        dict: Feature dict with geometry and rim_prob features only
    """
    h, w = img_shape
    cx, cy, a, b, _ = ellipse
    
    feats = {}
    
    # Geometry features (fast version, no prop)
    feats.update(geometry_features_fast(rim_pts, ellipse, img_shape))
    
    # Rim probability features
    feats.update(rim_prob_features(rim_pts, rim_prob))
    
    # NEW: Polar features (high value, no extra dependencies)
    feats.update(polar_features(rim_pts, ellipse, rim_prob))
    
    # Fill in zeros for missing features (to match feature set expected by ranker)
    # Morphology features (would need prop)
    feats.update({
        "morph_solidity": 1.0,
        "morph_convexity": 1.0,
        "morph_extent": 1.0,
        "morph_perim_area_ratio": 1.0,
        "morph_minor_axis": 1.0,
        "morph_area_log": np.log(np.pi * a * b / (h * w) + 1e-6)
    })
    
    # Stability features (would need bootstrap)
    feats.update({
        "stab_jitter_a": 0.0,
        "stab_jitter_b": 0.0,
        "stab_jitter_theta": 0.0,
        "stab_center_shift": 0.0,
        "stab_iou_subsample": 1.0
    })
    
    # Mahanti features (would need gradient)
    feats.update({
        "mahanti_median_slope": 0.0,
        "mahanti_slope_std": 0.0,
        "mahanti_depth_ratio": 0.0,
        "mahanti_rim_sharpness": 0.0
    })
    
    # Context features (would need mask)
    feats.update({
        "ctx_inside_mean": 0.0,
        "ctx_outside_mean": 0.0,
        "ctx_diff": 0.0,
        "ctx_grad_mean": 0.0,
        "ctx_grad_std": 0.0
    })
    
    # Meta features
    feats["meta_log_radius"] = np.log(np.sqrt(a * b) / np.sqrt(h * w) + 1e-6)
    feats["meta_log_area"] = feats["morph_area_log"]
    
    return feats
