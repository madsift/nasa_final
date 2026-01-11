"""
Crater Classification using LightGBM
Uses ranking_train_data.csv to classify craters into morphology classes (A/AB/B/BC/C)
Similar report structure to train_cls.py

Features:
- Pre-filtering by ranker model (only keep high-confidence matches)
- Class weighting for imbalanced data
- Ordinal regression option (treats classes as ordinal scale)
- Detailed Mahanti feature analysis
- Comprehensive EDA and visualization
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import random
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


# ============================================================
# VISUALIZATION DIRECTORY
# ============================================================
VIZ_DIR = "viz_classifier"

# ============================================================
# GLOBAL FEATURE COLUMNS
# These are the features used for classification training.
# Must match the columns in the training CSV (from features_full.csv)
# ============================================================
FEATURE_COLS = [
    # Geometry features (11)
    "geometry_eccentricity",
    "geometry_axis_ratio",
    "geometry_ellipse_area",
    "geometry_area_ratio",
    "geometry_resid_rms",
    "geometry_resid_p90",
    "geometry_support_ratio",
    "geometry_angular_coverage",
    "geometry_angular_std",
    "geometry_max_gap",
    "geometry_condition",
    
    # Rim probability features (9)
    "rim_prob_mean",
    "rim_prob_std",
    "rim_prob_min",
    "rim_prob_p20",
    "rim_prob_p80",
    "rim_prob_frac_above_50",
    "rim_prob_frac_above_70",
    "rim_prob_entropy",
    
    # Polar features (4)
    #"polar_angular_support",
    #"polar_mean_asymmetry",
    #"polar_max_asymmetry",
    #"polar_roughness_ratio",
    
    # Morphology features (6)
    #"morph_solidity",
    #"morph_convexity",
    #"morph_extent",
    #"morph_perim_area_ratio",
    #"morph_minor_axis",
    #"morph_area_log",
    
    # Stability features (5)
    #"stab_jitter_a",
    #"stab_jitter_b",
    #"stab_jitter_theta",
    #"stab_center_shift",
    #"stab_iou_subsample",
    
    # Mahanti degradation features (4)
    "mahanti_median_slope",
    "mahanti_slope_std",
    "mahanti_depth_ratio",
    "mahanti_rim_sharpness",
    
    # Context features (5)
    "ctx_inside_mean",
    "ctx_outside_mean",
    "ctx_diff",
    "ctx_grad_mean",
    "ctx_grad_std",
    
    # Meta features (2)
    #"meta_log_radius",
    #"meta_log_area",
    
    # SNR features (3)
    "snr_raw",
    "snr_x_illumination",
    "snr_illumination_level",
]

# Meta columns (not features, used for filtering/grouping)
META_COLS = ['target_score', 'target_xi2', 'target_class', 'image_id', 'pred_x', 'pred_y', 'ranker_prob']

# ============================================================
# NON-LINEAR ORDINAL TARGET MAPPING
# ============================================================
# Encodes asymmetric morphological distance:
# - A→AB is subtle (small gap)
# - BC→C is large morphological change (big gap)
# - A→C errors are heavily punished
ORDINAL_TARGET_MAP = {
    0: 0.0,   # A   (Fresh)
    1: 0.6,   # AB
    2: 1.5,   # B
    3: 2.7,   # BC
    4: 4.0    # C   (Highly degraded)
}

# Thresholds for converting continuous predictions back to classes
# These preserve ordering and reduce A↔C collapses
ORDINAL_THRESHOLDS = [0.3, 1.0, 2.1, 3.3]

# ============================================================
# MAHANTI FEATURE WEIGHTING
# ============================================================
# Bias tree splits toward physically meaningful degradation cues
# LightGBM splits by gain - larger magnitude → larger potential gain
MAHANTI_FEATURES = [
    "mahanti_median_slope",
    "mahanti_slope_std",
    "mahanti_depth_ratio",
    "mahanti_rim_sharpness",
]
MAHANTI_SCALE = 1.8      # Amplify Mahanti features
GEOMETRY_SCALE = 0.85    # Dampen geometry features

def ensure_viz_dir():
    """Create visualization directory if it doesn't exist."""
    os.makedirs(VIZ_DIR, exist_ok=True)
    print(f"Visualizations will be saved to: {VIZ_DIR}/")


def scale_feature_groups(df):
    """
    Apply deterministic feature scaling to bias LightGBM toward Mahanti features.
    
    LightGBM splits by gain - larger numeric magnitude → larger potential gain.
    This biases tree growth without hard constraints, no randomness, no leakage.
    Must be called ONCE, BEFORE train/val split.
    """
    df = df.copy()
    
    scaled_mahanti = []
    scaled_geometry = []
    
    for f in MAHANTI_FEATURES:
        if f in df.columns:
            df[f] *= MAHANTI_SCALE
            scaled_mahanti.append(f)
    
    for f in df.columns:
        if f.startswith("geometry_"):
            df[f] *= GEOMETRY_SCALE
            scaled_geometry.append(f)
    
    print(f"\\n--- MAHANTI FEATURE WEIGHTING ---")
    print(f"Scaled {len(scaled_mahanti)} Mahanti features by {MAHANTI_SCALE}x: {scaled_mahanti}")
    print(f"Scaled {len(scaled_geometry)} geometry features by {GEOMETRY_SCALE}x: {scaled_geometry}")
    
    return df


def ordinal_to_class(y_cont):
    """
    Convert continuous ordinal predictions back to class labels using thresholds.
    
    Uses ORDINAL_THRESHOLDS to map:
      < 0.3 → class 0 (A)
      0.3 ≤ x < 1.0 → class 1 (AB)
      1.0 ≤ x < 2.1 → class 2 (B)
      2.1 ≤ x < 3.3 → class 3 (BC)
      ≥ 3.3 → class 4 (C)
    
    DO NOT use naive rounding - this preserves ordering and reduces A↔C collapses.
    """
    y_cls = np.zeros_like(y_cont, dtype=int)
    y_cls[y_cont >= ORDINAL_THRESHOLDS[0]] = 1  # 0.3
    y_cls[y_cont >= ORDINAL_THRESHOLDS[1]] = 2  # 1.0
    y_cls[y_cont >= ORDINAL_THRESHOLDS[2]] = 3  # 2.1
    y_cls[y_cont >= ORDINAL_THRESHOLDS[3]] = 4  # 3.3
    return y_cls

# ============================================================
# RANKER FILTERING
# ============================================================

def filter_by_ranker(df, feature_cols, ranker_path, threshold=0.1):
    """
    Filter dataset using trained ranker model.
    Only keep samples with predicted match probability >= threshold.
    """
    print("\n" + "="*60)
    print("FILTERING BY RANKER MODEL")
    print("="*60)
    
    print(f"Loading ranker model from: {ranker_path}")
    
    if not os.path.exists(ranker_path):
        print(f"WARNING: Ranker model not found at {ranker_path}")
        print("Skipping ranker filtering - using all samples")
        return df
    
    try:
        ranker = lgb.Booster(model_file=ranker_path)
        print(f"Loaded ranker model successfully")
    except Exception as e:
        print(f"WARNING: Failed to load ranker model: {e}")
        print("Skipping ranker filtering - using all samples")
        return df
    
    ranker_features = ranker.feature_name()
    print(f"Ranker uses {len(ranker_features)} features")
    
    missing_features = [f for f in ranker_features if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features for ranker: {missing_features}")
        print("Skipping ranker filtering - using all samples")
        return df
    
    print(f"Predicting match probabilities for {len(df)} samples...")
    X = df[ranker_features]
    probs = ranker.predict(X)
    
    df = df.copy()
    df['ranker_prob'] = probs
    
    print(f"\nRanker probability statistics:")
    print(f"  Min:    {probs.min():.4f}")
    print(f"  Max:    {probs.max():.4f}")
    print(f"  Mean:   {probs.mean():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print(f"\nSamples above each threshold:")
    for t in thresholds:
        count = (probs >= t).sum()
        pct = 100 * count / len(probs)
        print(f"  >= {t:.1f}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\nApplying threshold >= {threshold}")
    original_len = len(df)
    df_filtered = df[df['ranker_prob'] >= threshold].copy()
    filtered_len = len(df_filtered)
    
    print(f"Samples before filtering: {original_len}")
    print(f"Samples after filtering:  {filtered_len}")
    print(f"Retention rate: {100*filtered_len/original_len:.1f}%")
    
    print(f"\nClass distribution comparison:")
    print("Before filtering:")
    before_dist = df['target_class'].value_counts().sort_index()
    print(before_dist)
    
    print("\nAfter filtering:")
    after_dist = df_filtered['target_class'].value_counts().sort_index()
    print(after_dist)
    
    print("\nRetention rate per class:")
    class_names = {0: 'A', 1: 'AB', 2: 'B', 3: 'BC', 4: 'C'}
    for cls in sorted(before_dist.index):
        before = before_dist.get(cls, 0)
        after = after_dist.get(cls, 0)
        rate = 100 * after / before if before > 0 else 0
        print(f"  Class {int(cls)} ({class_names.get(int(cls), '?')}): {before} -> {after} ({rate:.1f}%)")
    
    return df_filtered


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

def perform_eda(df, feature_cols, output_prefix="classifier"):
    """Perform Exploratory Data Analysis on the dataset."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\n--- Dataset Overview ---")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Number of images: {df['image_id'].nunique()}")
    
    print("\n--- Target Class Distribution ---")
    class_dist = df['target_class'].value_counts().sort_index()
    print(class_dist)
    
    class_names_map = {0: 'A (Fresh)', 1: 'AB', 2: 'B', 3: 'BC', 4: 'C (Degraded)'}
    
    plt.figure(figsize=(10, 6))
    ax = class_dist.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Crater Class Distribution (After Ranker Filtering)', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    total = len(df)
    for i, v in enumerate(class_dist.values):
        ax.text(i, v + total*0.01, f'{v}\n({100*v/total:.1f}%)', ha='center', fontsize=10)
    
    xticklabels = [class_names_map.get(int(idx), str(idx)) for idx in class_dist.index]
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/{output_prefix}_class_distribution.png', dpi=150)
    print(f"Saved {VIZ_DIR}/{output_prefix}_class_distribution.png")
    plt.close()
    
    print("\n--- Class Imbalance Analysis ---")
    max_class = class_dist.max()
    min_class = class_dist.min()
    print(f"Most common class: {class_dist.idxmax()} with {max_class} samples")
    print(f"Least common class: {class_dist.idxmin()} with {min_class} samples")
    print(f"Imbalance ratio (max/min): {max_class/min_class:.2f}x")
    
    print("\n--- Feature Statistics ---")
    feature_stats = df[feature_cols].describe().T
    feature_stats['missing'] = df[feature_cols].isna().sum()
    feature_stats['missing_pct'] = 100 * df[feature_cols].isna().sum() / len(df)
    print(feature_stats[['mean', 'std', 'min', 'max', 'missing', 'missing_pct']].round(4))
    
    return class_dist


def add_interaction_features(df):
    """
    Synthesize new features by combining existing top-performing features.
    Focuses on interactions between Degradation (Mahanti) features and Geometry.
    """
    print("\n" + "="*60)
    print("GENERATING INTERACTION FEATURES")
    print("="*60)
    
    df = df.copy()
    new_cols = []
    
    # 1. Degradation Synergies (combining Mahanti features)
    if 'mahanti_rim_sharpness' in df.columns and 'mahanti_depth_ratio' in df.columns:
        df['inter_sharpness_x_depth'] = df['mahanti_rim_sharpness'] * df['mahanti_depth_ratio']
        new_cols.append('inter_sharpness_x_depth')
    
    if 'mahanti_slope_std' in df.columns and 'mahanti_median_slope' in df.columns:
        df['inter_slope_agg'] = df['mahanti_slope_std'] * df['mahanti_median_slope']
        new_cols.append('inter_slope_agg')

    # 2. Gradient Normalized by Size (Gradient often scales with size)
    if 'ctx_grad_mean' in df.columns and 'meta_log_radius' in df.columns:
        # Avoid division by zero if any
        df['inter_grad_over_rad'] = df['ctx_grad_mean'] / (df['meta_log_radius'] + 1e-6)
        new_cols.append('inter_grad_over_rad')

    if 'ctx_grad_std' in df.columns and 'meta_log_radius' in df.columns:
        df['inter_grad_std_over_rad'] = df['ctx_grad_std'] / (df['meta_log_radius'] + 1e-6)
        new_cols.append('inter_grad_std_over_rad')

    # 3. Geometry Interactions
    if 'geometry_ellipse_area' in df.columns and 'morph_solidity' in df.columns:
        df['inter_area_x_solidity'] = df['geometry_ellipse_area'] * df['morph_solidity']
        new_cols.append('inter_area_x_solidity')
        
    if 'geometry_area_ratio' in df.columns and 'rim_prob_mean' in df.columns:
        df['inter_area_ratio_x_rim'] = df['geometry_area_ratio'] * df['rim_prob_mean']
        new_cols.append('inter_area_ratio_x_rim')

    print(f"Added {len(new_cols)} interaction features: {new_cols}")
    return df



def analyze_mahanti_features(df, feature_cols):
    """Detailed analysis of Mahanti features and their discriminative power."""
    print("\n" + "="*60)
    print("MAHANTI FEATURE ANALYSIS")
    print("="*60)
    
    mahanti_features = [f for f in feature_cols if 'mahanti' in f.lower()]
    
    if not mahanti_features:
        print("No Mahanti features found in the dataset.")
        return [], {}
    
    print(f"\nMahanti Features Found: {mahanti_features}")
    
    class_names = {0: 'A', 1: 'AB', 2: 'B', 3: 'BC', 4: 'C'}
    
    print("\n--- Mahanti Feature Statistics by Class ---")
    for feat in mahanti_features:
        print(f"\n{feat}:")
        stats = df.groupby('target_class')[feat].agg(['mean', 'std', 'median'])
        stats.index = [class_names.get(int(idx), str(idx)) for idx in stats.index]
        print(stats.round(4))
    
    print("\n--- Mahanti Feature Correlation with Class (Ordinal) ---")
    correlations = {}
    for feat in mahanti_features:
        corr = df[feat].corr(df['target_class'])
        correlations[feat] = corr
        print(f"{feat}: {corr:.4f}")
    
    plt.figure(figsize=(14, 10))
    
    n_features = len(mahanti_features)
    n_cols = min(2, n_features)
    n_rows = (n_features + 1) // 2
    
    for i, feat in enumerate(mahanti_features):
        plt.subplot(n_rows, n_cols, i + 1)
        classes = sorted(df['target_class'].unique())
        data_by_class = [df[df['target_class'] == c][feat].dropna().values for c in classes]
        parts = plt.violinplot(data_by_class, positions=range(len(classes)), showmeans=True, showmedians=True)
        plt.xticks(range(len(classes)), [class_names.get(int(c), str(c)) for c in classes])
        plt.xlabel('Class')
        plt.ylabel(feat)
        plt.title(f'{feat}\n(corr with class: {correlations[feat]:.3f})', fontsize=11, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Mahanti Features Distribution by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/mahanti_features_by_class.png', dpi=150)
    print(f"\nSaved {VIZ_DIR}/mahanti_features_by_class.png")
    plt.close()
    
    print("\n--- Mahanti Feature Summary ---")
    print("Feature effectiveness for crater degradation classification:")
    print("(Higher absolute correlation = better discriminative power)")
    
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        direction = "increases" if corr > 0 else "decreases"
        print(f"  {feat}: {corr:+.4f} ({direction} with degradation)")
    
    return mahanti_features, correlations


# ============================================================
# DATA SPLITTING
# ============================================================

def get_split_mask(df, val_group_prefixes):
    """Returns boolean mask for validation set based on image_id prefixes."""
    val_mask = df['image_id'].astype(str).str.startswith(tuple(val_group_prefixes))
    return val_mask


# ============================================================
# ORDINAL REGRESSION
# ============================================================

def focal_mse_objective(gamma=2.0):
    """
    Focal MSE Loss for regression tasks.
    
    Standard MSE: L = (y - pred)^2
    Focal MSE:    L = (1 - exp(-|y - pred|))^gamma * (y - pred)^2
    
    This down-weights easy examples (small error) and focuses on hard ones.
    gamma controls the focusing strength:
    - gamma = 0: equivalent to standard MSE
    - gamma = 2: moderate focusing (recommended)
    - gamma = 5: aggressive focusing
    """
    def focal_mse(preds, train_data):
        y_true = train_data.get_label()
        residual = y_true - preds
        abs_residual = np.abs(residual)
        
        # Focal weight: (1 - exp(-|r|))^gamma
        # For small errors, exp(-|r|) ≈ 1, so weight ≈ 0
        # For large errors, exp(-|r|) ≈ 0, so weight ≈ 1
        focal_weight = np.power(1 - np.exp(-abs_residual), gamma)
        
        # Gradient of focal MSE
        # d/dpred [w * (y - pred)^2] = -2 * w * (y - pred) + (y-pred)^2 * dw/dpred
        # Simplified: gradient ≈ -2 * w * residual (ignoring dw/dpred for stability)
        grad = -2 * focal_weight * residual
        
        # Hessian (second derivative)
        # Approximate with 2 * focal_weight for stability
        hess = 2 * focal_weight + 1e-6  # Add small constant for numerical stability
        
        return grad, hess
    
    return focal_mse


def class_balanced_focal_mse(y_train_classes, gamma=2.0, class_weights=None):
    """
    Focal MSE with class-based weighting.
    Combines focal loss focusing with explicit class weights.
    
    y_train_classes: Original class labels (0-4) before ordinal mapping
    class_weights: Dict mapping class -> weight, e.g. {0: 200, 1: 100, 2: 20, 3: 10, 4: 1}
    """
    if class_weights is None:
        class_weights = {0: 200.0, 1: 100.0, 2: 20.0, 3: 10.0, 4: 1.0}
    
    # Pre-compute sample weights based on classes
    sample_class_weights = np.array([class_weights.get(int(c), 1.0) for c in y_train_classes])
    
    def focal_mse_weighted(preds, train_data):
        y_true = train_data.get_label()
        residual = y_true - preds
        abs_residual = np.abs(residual)
        
        # Focal weight
        focal_weight = np.power(1 - np.exp(-abs_residual), gamma)
        
        # Combined weight: focal * class_weight
        combined_weight = focal_weight * sample_class_weights
        
        # Gradient
        grad = -2 * combined_weight * residual
        
        # Hessian
        hess = 2 * combined_weight + 1e-6
        
        return grad, hess
    
    return focal_mse_weighted


def train_ordinal_regression(X_train, y_train, X_val, y_val, sample_weights, args, seed=42, params_override=None):
    """
    Train LightGBM as ordinal regressor.
    Treats class labels as continuous ordinal values and predicts regression.
    
    Supports focal loss via args.focal_loss flag.
    """
    print("\n" + "="*60)
    print("TRAINING ORDINAL REGRESSION (LightGBM)")
    print("="*60)
    
    # Store original class labels for class-balanced focal loss
    y_train_classes = y_train.copy() if isinstance(y_train, np.ndarray) else y_train.values.copy()
    
    # Convert labels to non-linear ordinal targets
    # This encodes asymmetric morphological distance:
    # - A→AB is subtle (0.0 → 0.6 = 0.6 gap)
    # - BC→C is large (2.7 → 4.0 = 1.3 gap)
    if isinstance(y_train, pd.Series):
        y_train_reg = y_train.map(ORDINAL_TARGET_MAP).astype(float)
        y_val_reg = y_val.map(ORDINAL_TARGET_MAP).astype(float)
    else:
        y_train_reg = pd.Series(y_train).map(ORDINAL_TARGET_MAP).astype(float).values
        y_val_reg = pd.Series(y_val).map(ORDINAL_TARGET_MAP).astype(float).values
    
    print(f"Ordinal target mapping: {ORDINAL_TARGET_MAP}")
    print(f"Train targets - min: {y_train_reg.min():.2f}, max: {y_train_reg.max():.2f}")
    
    # Create datasets
    # Note: For custom objective, we still pass weights to Dataset for potential use
    train_data = lgb.Dataset(X_train, label=y_train_reg, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val_reg, reference=train_data)
    
    # Regression parameters (using tuned defaults from args)
    lgb_params = {
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': getattr(args, 'feature_fraction', 0.8),
        'bagging_fraction': getattr(args, 'bagging_fraction', 0.8),
        'bagging_freq': 5,
        'min_child_samples': getattr(args, 'min_child_samples', 30),
        'lambda_l1': getattr(args, 'lambda_l1', 1.0),
        'lambda_l2': getattr(args, 'lambda_l2', 0.0),
        'verbose': -1,
        'device': 'cpu',
        'max_bin': 255,
        'random_state': seed,
    }
    
    lgb_params['metric'] = ['rmse', 'mae']
    
    # Set objective based on focal_loss flag
    if hasattr(args, 'focal_loss') and args.focal_loss:
        gamma = getattr(args, 'focal_gamma', 2.0)
        print(f"Using FOCAL LOSS (gamma={gamma})")
        
        if hasattr(args, 'aggressive_weights') and args.aggressive_weights:
            print("  + Class-balanced weighting")
            lgb_params['objective'] = class_balanced_focal_mse(y_train_classes, gamma=gamma)
        else:
            lgb_params['objective'] = focal_mse_objective(gamma=gamma)
    else:
        lgb_params['objective'] = 'regression'
    
    if params_override:
        print(f"Applying parameter overrides: {params_override}")
        lgb_params.update(params_override)
    
    print(f"Parameters: num_leaves={lgb_params['num_leaves']}, lr={lgb_params['learning_rate']}")
    
    # Custom evaluation function for Macro F1
    # This ensures early stopping uses Macro F1 (what we care about) instead of RMSE
    def macro_f1_eval(preds, train_data):
        """Custom eval function that computes Macro F1 on validation set."""
        y_true = train_data.get_label()
        # Convert ordinal predictions to classes
        y_pred_class = ordinal_to_class(preds)
        # Convert ordinal labels back to classes
        y_true_class = ordinal_to_class(y_true)
        
        # Compute Macro F1
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_class, y_pred_class, average=None, zero_division=0, labels=[0, 1, 2, 3, 4]
        )
        macro_f1 = np.mean(f1)
        
        # Return (name, value, is_higher_better)
        return 'macro_f1', macro_f1, True
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=args.num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        feval=macro_f1_eval,  # Custom eval for Macro F1
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # Now uses macro_f1 since is_higher_better=True
            lgb.log_evaluation(period=50)
        ]
    )
    
    return model


def evaluate_ordinal(models, X_val, y_val, classes, args):
    """
    Evaluate ordinal regression model(s).
    Supports single model or list of models (ensemble).
    
    Uses threshold-based class mapping (not naive rounding) and reports
    MAE on ordinal targets to reflect morphological distance.
    """
    if not isinstance(models, list):
        models = [models]

    print("\n" + "="*60)
    print("EVALUATION ON VALIDATION SET (ORDINAL)")
    print("="*60)
    
    # Get continuous predictions from all models and average
    print(f"Aggregating predictions from {len(models)} model(s)...")
    preds_list = []
    for m in models:
        preds = m.predict(X_val, num_iteration=m.best_iteration)
        preds_list.append(preds)
    
    y_pred_continuous = np.mean(preds_list, axis=0)
    
    # Convert true labels to ordinal targets for MAE calculation
    if isinstance(y_val, pd.Series):
        y_val_ordinal = y_val.map(ORDINAL_TARGET_MAP).values
    else:
        y_val_ordinal = pd.Series(y_val).map(ORDINAL_TARGET_MAP).values
    
    # Show prediction distribution
    print("\n--- Continuous Predictions Distribution ---")
    print(f"Min: {y_pred_continuous.min():.4f}")
    print(f"Max: {y_pred_continuous.max():.4f}")
    print(f"Mean: {y_pred_continuous.mean():.4f}")
    print(f"Std: {y_pred_continuous.std():.4f}")
    
    # MAE on ordinal targets (IMPORTANT: reflects morphological distance)
    # A→C error (|0.0 - 4.0| = 4.0) >> A→AB error (|0.0 - 0.6| = 0.6)
    mae_ordinal = mean_absolute_error(y_val_ordinal, y_pred_continuous)
    print(f"\n*** MAE on ordinal targets: {mae_ordinal:.4f} ***")
    print("(This reflects morphological distance - lower is better)")
    
    print("\n--- Comparing Class Mapping Strategies ---")
    
    all_class_names = {0: 'A', 1: 'AB', 2: 'B', 3: 'BC', 4: 'C'}
    target_names = [all_class_names.get(int(c), str(c)) for c in classes]
    
    # Method 1: Threshold-based mapping (RECOMMENDED)
    print("\n[Method 1: Threshold-based Mapping (ordinal_to_class)]")
    print(f"Using thresholds: {ORDINAL_THRESHOLDS}")
    y_pred_thresh = ordinal_to_class(y_pred_continuous)
    
    precision1, recall1, f11, support1 = precision_recall_fscore_support(y_val, y_pred_thresh, average=None, zero_division=0, labels=classes)
    macro_f1_thresh = np.mean(f11)
    weighted_f1_thresh = np.average(f11, weights=support1)
    accuracy_thresh = (y_pred_thresh == y_val).mean()
    mae_class_thresh = mean_absolute_error(y_val, y_pred_thresh)
    
    print(f"Macro F1: {macro_f1_thresh:.4f}")
    print(f"Weighted F1: {weighted_f1_thresh:.4f}")
    print(f"Accuracy: {accuracy_thresh:.4f}")
    print(f"MAE (class distance): {mae_class_thresh:.4f}")
    
    # Method 2: Grid-search optimized thresholds
    print("\n[Method 2: Optimized Thresholds (grid search)]")
    best_thresholds, best_f1, y_pred_optimized = optimize_ordinal_thresholds(y_pred_continuous, y_val, classes)
    
    precision2, recall2, f12, support2 = precision_recall_fscore_support(y_val, y_pred_optimized, average=None, zero_division=0, labels=classes)
    macro_f1_opt = np.mean(f12)
    weighted_f1_opt = np.average(f12, weights=support2)
    accuracy_opt = (y_pred_optimized == y_val).mean()
    mae_class_opt = mean_absolute_error(y_val, y_pred_optimized)
    
    print(f"Optimized thresholds: {best_thresholds}")
    print(f"Macro F1: {macro_f1_opt:.4f}")
    print(f"Weighted F1: {weighted_f1_opt:.4f}")
    print(f"Accuracy: {accuracy_opt:.4f}")
    print(f"MAE (class distance): {mae_class_opt:.4f}")
    
    # Use the better method
    if macro_f1_opt > macro_f1_thresh:
        y_pred = y_pred_optimized
        macro_f1 = macro_f1_opt
        weighted_f1 = weighted_f1_opt
        accuracy = accuracy_opt
        f1 = f12
        print("\n>>> Using optimized thresholds (better macro F1)")
    else:
        y_pred = y_pred_thresh
        macro_f1 = macro_f1_thresh
        weighted_f1 = weighted_f1_thresh
        accuracy = accuracy_thresh
        f1 = f11
        print("\n>>> Using threshold-based mapping (better macro F1)")
    
    # Print classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, digits=4, target_names=target_names, zero_division=0, labels=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Ordinal Regression - Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/ordinal_confusion_matrix.png', dpi=150)
    print(f"Saved {VIZ_DIR}/ordinal_confusion_matrix.png")
    plt.close()
    
    # Plot prediction distribution with ordinal thresholds
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_continuous, bins=50, alpha=0.7, edgecolor='black')
    # Show ORDINAL_THRESHOLDS as class boundaries
    for i, thresh in enumerate(ORDINAL_THRESHOLDS):
        label = 'Class boundaries' if i == 0 else None
        plt.axvline(x=thresh, color='r', linestyle='--', label=label)
    # Show ordinal target values
    for cls, target in ORDINAL_TARGET_MAP.items():
        plt.axvline(x=target, color='g', linestyle=':', alpha=0.5)
    plt.xlabel('Predicted Value', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Continuous Predictions\\nThresholds: {ORDINAL_THRESHOLDS}', fontsize=12, fontweight='bold')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for cls in classes:
        subset = y_pred_continuous[y_val == cls]
        plt.hist(subset, bins=30, alpha=0.5, label=f'True={all_class_names[cls]}', density=True)
    plt.xlabel('Predicted Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Predictions by True Class', fontsize=12, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/ordinal_prediction_distribution.png', dpi=150)
    print(f"Saved {VIZ_DIR}/ordinal_prediction_distribution.png")
    plt.close()
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'mae_ordinal': mae_ordinal,  # MAE on ordinal targets (reflects morphological distance)
        'f1_per_class': f1,
        'y_pred': y_pred,
        'y_pred_continuous': y_pred_continuous
    }


def optimize_ordinal_thresholds(y_pred_continuous, y_true, classes):
    """
    Find optimal thresholds for converting continuous predictions to classes.
    Uses grid search to maximize macro F1.
    """
    from itertools import product
    
    # Define search ranges for thresholds
    # We need 4 thresholds for 5 classes
    # threshold[i] separates class i from class i+1
    
    # Coarse grid search
    t_ranges = [
        np.linspace(0.0, 1.0, 5),   # threshold between 0 and 1
        np.linspace(1.0, 2.0, 5),   # threshold between 1 and 2
        np.linspace(2.0, 3.0, 5),   # threshold between 2 and 3
        np.linspace(3.0, 4.0, 5),   # threshold between 3 and 4
    ]
    
    best_f1 = -1
    best_thresholds = [0.5, 1.5, 2.5, 3.5]
    
    for t0, t1, t2, t3 in product(*t_ranges):
        if not (t0 < t1 < t2 < t3):
            continue
            
        # Apply thresholds
        y_pred = np.zeros_like(y_pred_continuous, dtype=int)
        y_pred[y_pred_continuous >= t0] = 1
        y_pred[y_pred_continuous >= t1] = 2
        y_pred[y_pred_continuous >= t2] = 3
        y_pred[y_pred_continuous >= t3] = 4
        
        # Calculate macro F1
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=classes)
        macro_f1 = np.mean(f1)
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thresholds = [t0, t1, t2, t3]
    
    # Apply best thresholds
    y_pred_optimized = np.zeros_like(y_pred_continuous, dtype=int)
    y_pred_optimized[y_pred_continuous >= best_thresholds[0]] = 1
    y_pred_optimized[y_pred_continuous >= best_thresholds[1]] = 2
    y_pred_optimized[y_pred_continuous >= best_thresholds[2]] = 3
    y_pred_optimized[y_pred_continuous >= best_thresholds[3]] = 4
    
    return best_thresholds, best_f1, y_pred_optimized


# ============================================================
# MULTICLASS CLASSIFICATION
# ============================================================

def train_multiclass(X_train, y_train_mapped, X_val, y_val_mapped, sample_weights, num_classes, args):
    """Train standard multiclass LightGBM classifier."""
    print("\n" + "="*60)
    print("TRAINING MULTICLASS CLASSIFIER (LightGBM)")
    print("="*60)
    
    train_data = lgb.Dataset(X_train, label=y_train_mapped, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': ['multi_error', 'multi_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.0,
        'lambda_l2': 0.1,
        'verbose': -1,
        'device': 'cpu',
        'max_bin': 255,
        'random_state': 42,
    }
    
    print(f"Parameters: num_leaves={args.num_leaves}, lr={args.learning_rate}")
    print("Using sample weights for class balancing")
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=args.num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    return model


def evaluate_multiclass(model, X_val, y_val, classes, class_to_idx, idx_to_class):
    """Evaluate multiclass classifier."""
    print("\n" + "="*60)
    print("EVALUATION ON VALIDATION SET (MULTICLASS)")
    print("="*60)
    
    y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_mapped = np.argmax(y_pred_prob, axis=1)
    y_pred = np.array([idx_to_class[i] for i in y_pred_mapped])
    
    all_class_names = {0: 'A', 1: 'AB', 2: 'B', 3: 'BC', 4: 'C'}
    target_names = [all_class_names.get(int(c), str(c)) for c in classes]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, digits=4, target_names=target_names, zero_division=0))
    
    precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, average=None, zero_division=0)
    
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    accuracy = (y_pred == y_val).mean()
    
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(y_val, y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Multiclass - Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/multiclass_confusion_matrix.png', dpi=150)
    print(f"Saved {VIZ_DIR}/multiclass_confusion_matrix.png")
    plt.close()
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'f1_per_class': f1,
        'y_pred': y_pred
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================


def tune_hyperparameters(X_train, y_train, X_val, y_val, sample_weights, classes, args):
    import random
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING (Random Search)")
    print("="*60)
    
    # Search Space
    param_grid = {
        'num_leaves': [15, 31, 63, 100],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'min_child_samples': [20, 30, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'lambda_l1': [0.0, 0.1, 1.0, 5.0],
        'lambda_l2': [0.0, 0.1, 1.0, 5.0],
    }
    
    num_trials = 20
    best_f1 = -1
    best_params = {}
    
    history = []
    
    print(f"Starting {num_trials} trials...")
    
    for i in range(num_trials):
        # Sample params
        params = {k: random.choice(v) for k, v in param_grid.items()}
        # Map to LightGBM names
        lgb_params = params.copy()
        lgb_params['bagging_fraction'] = lgb_params.pop('subsample')
        lgb_params['feature_fraction'] = lgb_params.pop('colsample_bytree')
        
        print(f"\n[Trial {i+1}/{num_trials}] Params: {lgb_params}")
        
        try:
            # Train
            model = train_ordinal_regression(
                X_train, y_train, X_val, y_val, sample_weights, args, 
                seed=42, params_override=lgb_params
            )
            
            # Evaluate (Valid F1)
            # Use method 2 (optimized thresholds) implicitly? 
            # evaluate_ordinal calculates both and picks best.
            # We need to capture the output of evaluate_ordinal without printing massive plots?
            # It prints plots. That's fine for now, user can scroll.
            
            results = evaluate_ordinal(model, X_val, y_val, classes, args)
            f1 = results['macro_f1']
            
            print(f"Trial {i+1} Result: Macro F1 = {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = lgb_params.copy()
                print(">>> NEW BEST!")
                
            history.append({'params': lgb_params, 'f1': f1})
            
        except Exception as e:
            print(f"Trial failed: {e}")
        
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Best Macro F1: {best_f1:.4f}")
    print("Best Params:")
    print(best_params)
    
    return best_params


def train_with_cv(df, feature_cols, args):
    """
    Train with Stratified 5-Fold Cross-Validation.
    Only supports ordinal regression mode.
    """
    print("\n" + "="*60)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    if not args.ordinal:
        print("WARNING: CV mode only supported with --ordinal. Enabling ordinal automatically.")
        args.ordinal = True
    
    # Prepare data
    X = df[feature_cols].values
    y = df['target_class'].astype(int).values
    
    classes = sorted(np.unique(y))
    num_classes = len(classes)
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {num_classes} - {classes}")
    print(f"Class distribution:")
    for c in classes:
        count = (y == c).sum()
        print(f"  Class {c}: {count} ({100*count/len(y):.1f}%)")
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results per fold
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_continuous = []
    all_fold_models = []
    
    # Tuned params (will be set on fold 1 if --tune is enabled)
    tuned_params = None
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n" + "-"*40)
        print(f"FOLD {fold_idx + 1}/5")
        print("-"*40)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Convert to pandas for compatibility
        X_train = pd.DataFrame(X_train, columns=feature_cols)
        X_val = pd.DataFrame(X_val, columns=feature_cols)
        y_train = pd.Series(y_train)
        y_val = pd.Series(y_val)
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"Train class dist: {y_train.value_counts().sort_index().to_dict()}")
        print(f"Val class dist: {y_val.value_counts().sort_index().to_dict()}")
        
        # Apply SMOTE if requested
        if args.smote:
            print(f"Applying SMOTE (Seed={42+fold_idx})...")
            if SMOTE is None:
                print("ERROR: imblearn not installed.")
            else:
                min_class_size = y_train.value_counts().min()
                k_neighbors = min(5, min_class_size - 1)
                if k_neighbors < 1:
                    print(f"WARNING: Smallest class has only {min_class_size} samples. SMOTE skipped.")
                else:
                    before_counts = y_train.value_counts().sort_index().to_dict()
                    smote = SMOTE(random_state=42+fold_idx, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    if not isinstance(y_train, pd.Series):
                        y_train = pd.Series(y_train)
                    after_counts = y_train.value_counts().sort_index().to_dict()
                    print(f"  SMOTE applied (k={k_neighbors}): {sum(before_counts.values())} -> {sum(after_counts.values())} samples")
                    print(f"  Before: {before_counts}")
                    print(f"  After:  {after_counts}")
        
        # Compute sample weights
        if args.aggressive_weights:
            class_weight_dict = {0: 200.0, 1: 100.0, 2: 20.0, 3: 10.0, 4: 1.0}
            sample_weights = y_train.map(class_weight_dict).values
        else:
            sample_weights = compute_sample_weight('balanced', y_train)
        
        # Hyperparameter tuning on FOLD 1 only (if --tune is enabled)
        if args.tune and fold_idx == 0 and tuned_params is None:
            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING ON FOLD 1")
            print("="*60)
            tuned_params = tune_hyperparameters(
                X_train, y_train, X_val, y_val, sample_weights, classes, args
            )
            print(f"\nTuned params will be used for all 5 folds: {tuned_params}")
        
        # Train ordinal regression model (with tuned params if available)
        model = train_ordinal_regression(
            X_train, y_train, X_val, y_val, sample_weights, args, 
            seed=42+fold_idx,
            params_override=tuned_params  # None for fold 1 if not tuning, or tuned params
        )
        all_fold_models.append(model)
        
        # Predict on validation fold
        y_pred_continuous = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = ordinal_to_class(y_pred_continuous)
        
        # Store predictions
        all_y_true.extend(y_val.values)
        all_y_pred.extend(y_pred)
        all_y_pred_continuous.extend(y_pred_continuous)
        
        # Calculate fold metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, average=None, zero_division=0, labels=classes
        )
        macro_f1 = np.mean(f1)
        accuracy = (y_pred == y_val.values).mean()
        
        # MAE on ordinal targets
        y_val_ordinal = y_val.map(ORDINAL_TARGET_MAP).values
        mae_ordinal = mean_absolute_error(y_val_ordinal, y_pred_continuous)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'mae_ordinal': mae_ordinal,
            'f1_per_class': f1
        })
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  MAE (ordinal): {mae_ordinal:.4f}")
    
    # Aggregate results
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_pred_continuous = np.array(all_y_pred_continuous)
    
    # Per-fold summary
    print("\n--- Per-Fold Results ---")
    for res in fold_results:
        print(f"Fold {res['fold']}: Macro F1={res['macro_f1']:.4f}, Acc={res['accuracy']:.4f}, MAE={res['mae_ordinal']:.4f}")
    
    # Average metrics
    avg_macro_f1 = np.mean([r['macro_f1'] for r in fold_results])
    std_macro_f1 = np.std([r['macro_f1'] for r in fold_results])
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    avg_mae = np.mean([r['mae_ordinal'] for r in fold_results])
    std_mae = np.std([r['mae_ordinal'] for r in fold_results])
    
    print("\n--- Average Metrics (± std) ---")
    print(f"Macro F1:  {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"MAE:       {avg_mae:.4f} ± {std_mae:.4f}")
    
    # Aggregate confusion matrix
    all_class_names = {0: 'A', 1: 'AB', 2: 'B', 3: 'BC', 4: 'C'}
    target_names = [all_class_names.get(int(c), str(c)) for c in classes]
    
    print("\n--- Aggregated Classification Report (all folds) ---")
    print(classification_report(all_y_true, all_y_pred, digits=4, target_names=target_names, zero_division=0, labels=classes))
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('5-Fold CV - Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/cv_confusion_matrix.png', dpi=150)
    print(f"\nSaved {VIZ_DIR}/cv_confusion_matrix.png")
    plt.close()
    
    # Save all fold models as separate .txt files
    print(f"\n--- Saving CV Fold Models ---")
    saved_model_paths = []
    for fold_i, fold_model in enumerate(all_fold_models):
        model_path = f'{VIZ_DIR}/{args.output_prefix}_cv_fold_{fold_i + 1}.txt'
        fold_model.save_model(model_path)
        saved_model_paths.append(model_path)
        print(f"  Fold {fold_i + 1}: {model_path}")
    
    # Also save a summary file listing all model paths (for easy loading)
    summary_path = f'{VIZ_DIR}/{args.output_prefix}_cv_models.txt'
    with open(summary_path, 'w') as f:
        f.write("# Cross-Validation Model Summary\n")
        f.write(f"# Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}\n")
        f.write(f"# Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"# MAE (ordinal): {avg_mae:.4f} ± {std_mae:.4f}\n")
        f.write(f"# Total samples: {len(df)}\n")
        f.write(f"# High Quality Only: {args.high_quality_only}\n")
        f.write("#\n")
        f.write("# Fold Results:\n")
        for res in fold_results:
            f.write(f"# Fold {res['fold']}: Macro F1={res['macro_f1']:.4f}, Acc={res['accuracy']:.4f}, MAE={res['mae_ordinal']:.4f}\n")
        f.write("#\n")
        f.write("# Model Files:\n")
        for path in saved_model_paths:
            f.write(f"{path}\n")
    print(f"\nSaved CV summary to {summary_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL CV SUMMARY")
    print("="*60)
    print(f"Mode: ORDINAL REGRESSION with 5-FOLD STRATIFIED CV")
    print(f"High Quality Only: {'YES (target_score >= 0.9)' if args.high_quality_only else 'NO'}")
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {num_classes}")
    print(f"Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"MAE (ordinal): {avg_mae:.4f} ± {std_mae:.4f}")
    
    return all_fold_models[-1]


def train_classifier(args):
    """Train LightGBM classifier for crater morphology classification."""
    ensure_viz_dir()
    
    # 1. Load Data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} samples")
    
    # Use global feature columns
    feature_cols = FEATURE_COLS.copy()
    
    # Validate that all expected features exist in the CSV
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"\nWARNING: Missing features in CSV: {missing_features}")
        print("Using only available features...")
        feature_cols = [f for f in feature_cols if f in df.columns]
    
    # Check for extra features in CSV that aren't in our global list
    extra_cols = [c for c in df.columns if c not in feature_cols and c not in META_COLS]
    if extra_cols:
        print(f"\nNOTE: Extra columns in CSV (not used): {extra_cols}")
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # 2. Clean Data
    print("\n--- Data Cleaning ---")
    print(f"Original class distribution:")
    print(df['target_class'].value_counts().sort_index())
    
    df = df[df['target_class'] >= 0]
    print(f"\nAfter removing class -1: {len(df)} samples")
    
    initial_len = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"After dropping NaN features: {len(df)} samples (dropped {initial_len - len(df)})")
    
    # 3. Filter by ranker
    if args.use_ranker:
        df = filter_by_ranker(df, feature_cols, args.ranker_model, args.ranker_thresh)
        if len(df) == 0:
            print("ERROR: No samples remaining after ranker filtering!")
            return None
    else:
        print("\nRanker filtering: DISABLED")
    
    # Perform EDA
    perform_eda(df, feature_cols, output_prefix=args.output_prefix)
    analyze_mahanti_features(df, feature_cols)
    
    # 3.5 Add Interactions (if requested)
    if args.interactions:
        df = add_interaction_features(df)
        # Add only the NEW interaction features to feature_cols (don't replace)
        interaction_cols = [c for c in df.columns if c.startswith('int_')]
        feature_cols = feature_cols + interaction_cols
        print(f"Added {len(interaction_cols)} interaction features: {interaction_cols}")
    
    # 3.6 Apply Mahanti Feature Weighting (before train/val split)
    # This biases LightGBM toward physically meaningful degradation cues
    if args.mahanti_weight:
        df = scale_feature_groups(df)
    else:
        print("\nMahanti feature weighting: DISABLED")
    
    # 4a. Filter to high-quality matches only (if requested)
    if args.high_quality_only:
        print("\n" + "="*60)
        print("FILTERING TO HIGH-QUALITY MATCHES ONLY")
        print("="*60)
        original_len = len(df)
        df = df[df['target_score'] >= 0.9].copy()
        print(f"Filtered from {original_len} to {len(df)} samples (target_score >= 0.9)")
        print(f"Retention rate: {100*len(df)/original_len:.1f}%")
        
        if len(df) == 0:
            print("ERROR: No samples remaining after high-quality filtering!")
            return None
    
    # 4. Split Data (CV or predefined)
    print("\n" + "="*60)
    print("DATA SPLITTING")
    print("="*60)
    
    if args.cv:
        print(f"Using STRATIFIED 5-FOLD CROSS-VALIDATION")
        return train_with_cv(df, feature_cols, args)
    
    # Original predefined split path
    val_groups = args.val_groups.split(',')
    print(f"Using predefined validation groups: {val_groups}")
    val_mask = get_split_mask(df, val_groups)
    
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()
    
    print(f"\nTrain Samples: {len(train_df)}")
    print(f"Val Samples: {len(val_df)}")
    
    if len(train_df) == 0 or len(val_df) == 0:
        print("ERROR: Empty train or validation set!")
        return None
    
    # 5. Class Distribution
    print("\n--- Train Class Distribution ---")
    train_class_dist = train_df['target_class'].value_counts().sort_index()
    print(train_class_dist)
    
    print("\n--- Val Class Distribution ---")
    val_class_dist = val_df['target_class'].value_counts().sort_index()
    print(val_class_dist)
    
    # 6. Prepare X, y
    X_train = train_df[feature_cols]
    y_train = train_df['target_class'].astype(int)
    
    X_val = val_df[feature_cols]
    y_val = val_df['target_class'].astype(int)

    classes = sorted(np.unique(np.concatenate([y_train.unique(), y_val.unique()])))
    num_classes = len(classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {classes}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_val_mapped = y_val.map(class_to_idx)
    
    # Back up for ensemble
    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()

    # --- Training Loop (Ensemble) ---
    models = []
    ensemble_size = args.ensemble_size if args.ordinal else 1
    # Hyperparameter Tuning
    if args.tune:
        if not args.ordinal:
            print("ERROR: --tune only supported with --ordinal")
            return None
            
        print("Preparing data for HPO...")
        X_train_hpo = X_train.copy()
        y_train_hpo = y_train.copy()
        
        if args.smote:
             if SMOTE is None:
                 print("ERROR: imblearn not installed.")
             else:
                 # Dynamically set k_neighbors based on smallest class size
                 min_class_size = y_train_hpo.value_counts().min()
                 k_neighbors = min(5, min_class_size - 1)
                 if k_neighbors < 1:
                     print(f"WARNING: Smallest class has only {min_class_size} samples. SMOTE skipped for HPO.")
                 else:
                     print(f"Applying SMOTE for HPO (Seed=42, k={k_neighbors})...")
                     smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                     X_train_hpo, y_train_hpo = smote.fit_resample(X_train_hpo, y_train_hpo)
        
        # Compute weights
        if args.aggressive_weights:
            class_weight_dict = {0: 200.0, 1: 100.0, 2: 20.0, 3: 10.0, 4: 1.0}
            if not isinstance(y_train_hpo, pd.Series):
                 sample_weights_hpo = pd.Series(y_train_hpo).map(class_weight_dict).values
            else:
                 sample_weights_hpo = y_train_hpo.map(class_weight_dict).values
        else:
             sample_weights_hpo = compute_sample_weight('balanced', y_train_hpo)
            
        best_params = tune_hyperparameters(X_train_hpo, y_train_hpo, X_val, y_val, sample_weights_hpo, classes, args)
        
        print("\nTraining final model with BEST parameters...")
        model = train_ordinal_regression(X_train_hpo, y_train_hpo, X_val, y_val, sample_weights_hpo, args, seed=42, params_override=best_params)
        models.append(model)
        ensemble_size = 0 # Skip standard loop
    
    
    if args.ensemble_size > 1 and not args.ordinal:
        print("WARNING: Ensemble currently only supported for Ordinal Regression. Training single Multiclass model.")
    
    for i in range(ensemble_size):
        if ensemble_size > 1:
            print(f"\n" + "-"*40)
            print(f"TRAINING MODEL {i+1}/{ensemble_size}")
            print("-" * 40)
            
        # Reset to original data
        X_train = X_train_orig.copy()
        y_train = y_train_orig.copy()
        
        # Apply SMOTE if requested (with varying seed)
        if args.smote:
            print(f"APPLYING SMOTE OVERSAMPLING (Seed={42+i})")
            if SMOTE is None:
                print("ERROR: imblearn not installed.")
            else:
                # Dynamically set k_neighbors based on smallest class size
                # SMOTE requires at least k+1 samples per class
                min_class_size = y_train.value_counts().min()
                k_neighbors = min(5, min_class_size - 1)  # Default is 5, but reduce if needed
                if k_neighbors < 1:
                    print(f"WARNING: Smallest class has only {min_class_size} samples. SMOTE skipped.")
                else:
                    print(f"  Using k_neighbors={k_neighbors} (min class size={min_class_size})")
                    smote = SMOTE(random_state=42+i, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Compute Weights
        if args.aggressive_weights:
            class_weight_dict = {0: 200.0, 1: 100.0, 2: 20.0, 3: 10.0, 4: 1.0}
            if not isinstance(y_train, pd.Series):
                 sample_weights = pd.Series(y_train).map(class_weight_dict).values
            else:
                 sample_weights = y_train.map(class_weight_dict).values
            print("Using CUSTOM AGGRESSIVE weights.")
        else:
             sample_weights = compute_sample_weight('balanced', y_train)

        # Train Model
        print("\n" + "="*60)
        if args.ordinal:
            print(f"TRAINING ORDINAL REGRESSION (Model {i+1})")
            model = train_ordinal_regression(X_train, y_train, X_val, y_val, sample_weights, args, seed=42+i)
            models.append(model)
        else:
            print("TRAINING MULTICLASS CLASSIFIER")
            if not isinstance(y_train, pd.Series):
                 y_train_mapped = pd.Series(y_train).map(class_to_idx)
            else:
                 y_train_mapped = y_train.map(class_to_idx)
            
            model = train_multiclass(X_train, y_train_mapped, X_val, y_val_mapped, sample_weights, num_classes, args)
            models.append(model)
            model_suffix = "multiclass"
            break

    # Evaluation
    
    if args.ordinal:
        results = evaluate_ordinal(models, X_val, y_val, classes, args)
        model_suffix = "ordinal"
        model = models[0]
    else:
        results = evaluate_multiclass(models[0], X_val, y_val_mapped, classes, class_to_idx, idx_to_class)
        model_suffix = "multiclass"
        model = models[0]
    
    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    importance_gain = model.feature_importance(importance_type='gain')
    
    feature_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Gain': importance_gain
    }).sort_values('Gain', ascending=False)
    
    print("\nTop 15 Features (Gain):")
    print(feature_imp.head(15).to_string(index=False))
    
    print("\n--- Mahanti Feature Importance ---")
    mahanti_imp = feature_imp[feature_imp['Feature'].str.contains('mahanti')]
    print(mahanti_imp.to_string(index=False))
    
    feature_imp['Rank'] = range(1, len(feature_imp) + 1)
    mahanti_ranks = feature_imp[feature_imp['Feature'].str.contains('mahanti')][['Feature', 'Rank', 'Gain']]
    print("\nMahanti Feature Ranks:")
    print(mahanti_ranks.to_string(index=False))
    
    plt.figure(figsize=(12, 8))
    top_features = feature_imp.head(20)
    colors = ['darkorange' if 'mahanti' in f.lower() else 'steelblue' for f in top_features['Feature']]
    plt.barh(range(len(top_features)), top_features['Gain'].values, color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.title(f'LightGBM ({model_suffix}) - Top 20 Feature Importance\n(Orange = Mahanti features)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/{model_suffix}_feature_importance.png', dpi=150)
    print(f"Saved {VIZ_DIR}/{model_suffix}_feature_importance.png")
    plt.close()
    
    # Save Model
    model_path = f'{VIZ_DIR}/{args.output_prefix}_{model_suffix}_model.txt'
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")
    
    # ============================================================
    # BASELINE COMPARISON
    # ============================================================
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    # Majority class baseline
    majority_class = y_val.mode()[0]
    y_pred_majority = np.full(len(y_val), majority_class)
    _, _, f1_baseline, support_baseline = precision_recall_fscore_support(y_val, y_pred_majority, average=None, zero_division=0, labels=classes)
    macro_f1_baseline = np.mean(f1_baseline)
    
    print(f"Majority Class Baseline (always predict C={majority_class}):")
    print(f"  Macro F1: {macro_f1_baseline:.4f}")
    print(f"  Accuracy: {(y_pred_majority == y_val).mean():.4f}")
    
    print(f"\nYour Model ({model_suffix}):")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Improvement: +{results['macro_f1'] - macro_f1_baseline:.4f} ({100*(results['macro_f1']/macro_f1_baseline - 1):.1f}%)")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Mode: {'ORDINAL REGRESSION' if args.ordinal else 'MULTICLASS CLASSIFICATION'}")
    print(f"Ranker Filtering: {'ENABLED (thresh={:.2f})'.format(args.ranker_thresh) if args.use_ranker else 'DISABLED'}")
    print(f"Mahanti Feature Weighting: {'ENABLED' if args.mahanti_weight else 'DISABLED'}")
    print(f"Train Samples: {len(train_df)}")
    print(f"Val Samples: {len(val_df)}")
    print(f"Number of Classes: {num_classes} (A, AB, B, BC, C)")
    print(f"Class Weighting: ENABLED (balanced)")
    print(f"\nBest Iteration: {model.best_iteration}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    if args.ordinal and 'mae_ordinal' in results:
        print(f"MAE on ordinal targets: {results['mae_ordinal']:.4f}")
    print(f"\nAll visualizations saved to: {VIZ_DIR}/")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Crater Classifier")
    parser.add_argument('--csv', type=str, default='ranking_train_data.csv',
                        help='Path to CSV file with features')
    parser.add_argument('--val_groups', type=str, 
                        default='altitude01_longitude10,altitude01_longitude14,altitude04_longitude13,altitude06_longitude13,altitude09_longitude08',
                        help='Comma-separated validation group prefixes')
    
    # Ranker filtering arguments
    parser.add_argument('--use_ranker', action='store_true', default=True,
                        help='Enable ranker-based filtering (default: True)')
    parser.add_argument('--no_ranker', action='store_false', dest='use_ranker',
                        help='Disable ranker-based filtering')
    parser.add_argument('--ranker_model', type=str, default='lightgbm_ranker.txt',
                        help='Path to trained ranker model')
    parser.add_argument('--ranker_thresh', type=float, default=0.1,
                        help='Ranker probability threshold (default: 0.1)')
    
    # Cross-validation and data filtering
    parser.add_argument('--cv', action='store_true', default=False,
                        help='Use stratified 5-fold cross-validation instead of predefined val groups')
    parser.add_argument('--high_quality_only', action='store_true', default=False,
                        help='Only use high quality matches (target_score >= 0.9)')
    
    # Model type
    parser.add_argument('--ordinal', action='store_true', default=False,
                        help='Use ordinal regression instead of multiclass classification')
    parser.add_argument('--aggressive_weights', action='store_true', default=False,
                        help='Use aggressive custom weights for minority classes (A=200, AB=100, B=20, BC=10, C=1)')
    parser.add_argument('--smote', action='store_true', default=False,
                        help='Use SMOTE oversampling for minority classes')
    parser.add_argument('--interactions', action='store_true', default=False,
                        help='Generate interaction features (e.g. sharpness * depth)')
    parser.add_argument('--focal_loss', action='store_true', default=False,
                        help='Use focal loss to focus on hard examples (helps with class imbalance)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (default: 2.0, higher = more focus on hard examples)')
    parser.add_argument('--tune', action='store_true', default=False,
                        help='Run hyperparameter tuning (Random Search)')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models to train for ensemble (Ordinal only). Default 1.')
    parser.add_argument('--mahanti_weight', action='store_true', default=True,
                        help='Apply Mahanti feature weighting (scale up Mahanti, scale down geometry)')
    parser.add_argument('--no_mahanti_weight', action='store_false', dest='mahanti_weight',
                        help='Disable Mahanti feature weighting')
    
    # LightGBM parameters (tuned defaults from HPO)
    parser.add_argument('--num_leaves', type=int, default=100,
                        help='LightGBM num_leaves parameter')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='LightGBM learning rate')
    parser.add_argument('--min_child_samples', type=int, default=30,
                        help='LightGBM min_child_samples parameter')
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help='LightGBM L1 regularization')
    parser.add_argument('--lambda_l2', type=float, default=0.0,
                        help='LightGBM L2 regularization')
    parser.add_argument('--bagging_fraction', type=float, default=0.8,
                        help='LightGBM bagging_fraction (subsample)')
    parser.add_argument('--feature_fraction', type=float, default=0.8,
                        help='LightGBM feature_fraction (colsample_bytree)')
    parser.add_argument('--num_rounds', type=int, default=1000,
                        help='Maximum number of boosting rounds')
    parser.add_argument('--output_prefix', type=str, default='classifier',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    train_classifier(args)


'''
python train_classifier.py  --high_quality_only --cv --csv ./ranking_data/o3/features_full.csv --ordinal --num_rounds 1000 --no_ranker --learning_rate 0.03 --interactions --aggressive_weights
Mode: ORDINAL REGRESSION with 5-FOLD STRATIFIED CV
High Quality Only: YES (target_score >= 0.9)
Total samples: 21143
Number of classes: 5
Macro F1: 0.3271 ± 0.0056
Accuracy: 0.8527 ± 0.0019
MAE (ordinal): 0.3256 ± 0.0029

python train_classifier.py  --high_quality_only --cv --csv ./ranking_data/o3/features_full.csv --ordinal --num_rounds 1000 --no_ranker --learning_rate 0.03 --interactions --aggressive_weights
============================================================
FINAL CV SUMMARY
============================================================
Mode: ORDINAL REGRESSION with 5-FOLD STRATIFIED CV
High Quality Only: YES (target_score >= 0.9)
Total samples: 21143
Number of classes: 5
Macro F1: 0.3606 ± 0.0119
Accuracy: 0.7708 ± 0.0087
MAE (ordinal): 0.4139 ± 0.0097

'''

'''
============================================================
TUNING COMPLETE
============================================================
Best Macro F1: 0.5440
Best Params:
{'num_leaves': 100, 'learning_rate': 0.1, 'min_child_samples': 30, 'lambda_l1': 1.0, 'lambda_l2': 0.0, 'bagging_fraction': 0.8, 'feature_fraction': 0.8}

Tuned params will be used for all 5 folds: {'num_leaves': 100, 'learning_rate': 0.1, 'min_child_samples': 30, 'lambda_l1': 1.0, 'lambda_l2': 0.0, 'bagging_fraction': 0.8, 'feature_fraction': 0.8}
Mode: ORDINAL REGRESSION with 5-FOLD STRATIFIED CV
High Quality Only: YES (target_score >= 0.9)
Total samples: 36647
Number of classes: 5
Macro F1: 0.5114 ± 0.0163
Accuracy: 0.6920 ± 0.0328
MAE (ordinal): 0.5432 ± 0.0391

'''