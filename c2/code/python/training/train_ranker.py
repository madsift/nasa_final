#!/usr/bin/env python3
"""
Train a LightGBM binary classifier for crater ranking.

Uses ultra-fast features (geometry + rim_prob) that match eval3rect.py --ranker_fast mode.
Trains to predict probability of a candidate being a true positive (matched to GT).

Usage:
    python train_ranker.py --input_data ranking_data/features.csv
    python train_ranker.py --input_data data1.csv data2.csv  # Multiple files
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

# Ultra-fast features: geometry + rim_prob + SNR
# These match extract_crater_features_ultra_fast() in ranking_features_multires.py
FEATURE_COLS = [
    # geometry_features_fast
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
    # rim_prob_features
    "rim_prob_mean",
    "rim_prob_std",
    "rim_prob_min",
    "rim_prob_p20",
    "rim_prob_p80",
    "rim_prob_frac_above_50",
    "rim_prob_frac_above_70",
    "rim_prob_entropy",
    # SNR features (local signal-to-noise ratio)
    #"snr_raw",
    #"snr_x_illumination",
    #"snr_illumination_level",
    # meta features
    "meta_log_radius",
    "meta_log_area",
]

# Validation groups (fixed for reproducibility)
VAL_GROUPS = [
    'altitude01_longitude10', 'altitude01_longitude14',
    'altitude04_longitude13', 'altitude06_longitude13',
    'altitude09_longitude08'
]



def get_split_mask(df, val_group_prefixes):
    """Returns boolean mask for validation set based on image_id prefixes."""
    return df['image_id'].astype(str).str.startswith(tuple(val_group_prefixes))


def train_ranker(csv_paths, output_model='lightgbm_ranker.txt'):
    """
    Train LightGBM binary classifier on crater features.
    
    Args:
        csv_paths: List of CSV file paths containing feature data
        output_model: Output model filename
    """
    # Load data
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    print(f"Loading data from {len(csv_paths)} file(s)...")
    dfs = []
    for path in csv_paths:
        print(f"  - {path}")
        dfs.append(pd.read_csv(path))
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")
    
    # Create binary label (matched = target_score > 0)
    df['matched'] = (df['target_score'] > 0).astype(int)
    
    # Filter to available features
    available = set(df.columns)
    feature_cols = [f for f in FEATURE_COLS if f in available]
    missing = set(FEATURE_COLS) - set(feature_cols)
    
    print(f"\nUsing {len(feature_cols)} features")
    if missing:
        print(f"  ⚠️ Missing: {sorted(missing)}")
    
    # Split data
    val_mask = get_split_mask(df, VAL_GROUPS)
    train_df = df[~val_mask]
    val_df = df[val_mask]
    
    print(f"\nTrain/Val Split: {len(train_df)} / {len(val_df)} candidates")
    print(f"Train images: {train_df['image_id'].nunique()}")
    print(f"Val images: {val_df['image_id'].nunique()}")
    
    # Prepare features
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    y_train = train_df['matched']
    y_val = val_df['matched']
    
    # Compute class imbalance for weighting
    n_pos = (train_df['matched'] == 1).sum()
    n_neg = (train_df['matched'] == 0).sum()
    neg_weight = n_pos / max(1, n_neg)
    
    print(f"\nClass distribution: {n_pos} positive ({100*n_pos/(n_pos+n_neg):.1f}%), {n_neg} negative")
    print(f"Using scale_pos_weight: {1/neg_weight:.3f} (upweight negatives for FP reduction)")
    
    # LightGBM parameters optimized for this task
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        'colsample_bytree': 0.85,
        'subsample': 0.8,
        'scale_pos_weight': 1/neg_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Create datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    # Train
    print("\nTraining LightGBM Binary Classifier...")
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'gain': bst.feature_importance(importance_type='gain'),
        'split': bst.feature_importance(importance_type='split')
    }).sort_values('gain', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))
    
    # Save importance plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='gain', y='feature', data=importance.head(20))
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('ranking_feature_importance.png')
    plt.close()
    
    # Validation metrics
    val_preds = bst.predict(X_val)
    auc = roc_auc_score(y_val, val_preds)
    ap = average_precision_score(y_val, val_preds)
    
    print(f"\nValidation Metrics:")
    print(f"  AUC: {auc:.5f}")
    print(f"  AP:  {ap:.5f}")
    
    # Threshold analysis
    val_df = val_df.copy()
    val_df['pred_score'] = val_preds
    image_ids = val_df['image_id'].unique()
    
    print(f"\n--- Threshold Analysis (Top-10 Selection) ---")
    print(f"{'Thresh':<8} {'Quality':>10} {'Avg/Img':>10} {'FPs':>6}")
    print("-" * 40)
    
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        quality = 0.0
        n_selected = 0
        fps = 0
        
        for img_id in image_ids:
            grp = val_df[val_df['image_id'] == img_id]
            selected = grp[grp['pred_score'] > thresh].nlargest(10, 'pred_score')
            quality += selected['target_score'].sum()
            n_selected += len(selected)
            fps += (selected['matched'] == 0).sum()
        
        avg_per_img = n_selected / len(image_ids)
        print(f"{thresh:<8.1f} {quality:>10.1f} {avg_per_img:>10.1f} {fps:>6d}")
    
    print(f"\n--- Discard Mode (threshold only, no top-10) ---")
    print(f"{'Thresh':<8} {'Selected':>10} {'TPs':>8} {'FPs':>6} {'Precision':>10}")
    print("-" * 50)
    
    for thresh in [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]:
        selected = val_df[val_df['pred_score'] > thresh]
        tps = (selected['matched'] == 1).sum()
        fps = (selected['matched'] == 0).sum()
        precision = tps / max(1, len(selected))
        print(f"{thresh:<8.2f} {len(selected):>10d} {tps:>8d} {fps:>6d} {precision:>10.3f}")
    
    # Save model
    bst.save_model(output_model)
    print(f"\nSaved model to {output_model}")
    
    return bst, importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LightGBM ranker for crater detection')
    parser.add_argument('--input_data', type=str, nargs='+', required=True,
                        help='One or more CSV files containing feature data')
    parser.add_argument('--output', type=str, default='lightgbm_ranker.txt',
                        help='Output model filename')
    args = parser.parse_args()
    
    train_ranker(args.input_data, args.output)
