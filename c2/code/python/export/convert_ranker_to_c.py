#!/usr/bin/env python3
"""
Convert LightGBM model(s) to pure C code for edge deployment.

Supports both single model and ensemble (multiple models averaged).

Usage:
    # Single model:
    python convert_ranker_to_c.py --input lightgbm_ranker.txt --output cplusplus/lightgbm_ranker.h
    
    # Ensemble (multiple models):
    python convert_ranker_to_c.py --input fold1.txt fold2.txt fold3.txt --output cplusplus/ensemble.h
    
    # Using glob pattern:
    python convert_ranker_to_c.py --input viz_classifier/classifier_cv_fold_*.txt --output classifier_ensemble.h

This will generate a C header that can be included in eval3.cpp
"""

import lightgbm as lgb
import argparse
import os
import glob

def main():
    parser = argparse.ArgumentParser(
        description='Convert LightGBM model(s) to pure C code for edge deployment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single model:
    python convert_ranker_to_c.py -i lightgbm_ranker.txt -o ranker.h
    
  Ensemble (CV folds):
    python convert_ranker_to_c.py -i viz_classifier/classifier_cv_fold_1.txt \\
                                     viz_classifier/classifier_cv_fold_2.txt \\
                                     viz_classifier/classifier_cv_fold_3.txt \\
                                     viz_classifier/classifier_cv_fold_4.txt \\
                                     viz_classifier/classifier_cv_fold_5.txt \\
                                  -o classifier_ensemble.h
        """
    )
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True,
                        help='Path(s) to input LightGBM model file(s). Supports multiple files for ensemble.')
    parser.add_argument('--output', '-o', type=str, default='lightgbm_model.h',
                        help='Path to output C header file (default: lightgbm_model.h)')
    parser.add_argument('--mode', '-m', type=str, choices=['regression', 'binary'], default='regression',
                        help='Model mode: regression (ordinal) or binary (classification with sigmoid). Default: regression')
    args = parser.parse_args()
    
    # Expand glob patterns
    input_files = []
    for pattern in args.input:
        matches = glob.glob(pattern)
        if matches:
            input_files.extend(sorted(matches))
        else:
            # No glob match, treat as literal path
            input_files.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    input_files = [f for f in input_files if not (f in seen or seen.add(f))]
    
    if len(input_files) == 0:
        print("ERROR: No input files found!")
        return
    
    print(f"Found {len(input_files)} model file(s):")
    for f in input_files:
        print(f"  - {f}")
    
    # Load all models
    models = []
    for path in input_files:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return
        print(f"\nLoading: {path}")
        model = lgb.Booster(model_file=path)
        models.append((path, model))
        print(f"  Features: {len(model.feature_name())}")
    
    # Verify all models have same features
    feature_names = models[0][1].feature_name()
    for path, model in models[1:]:
        if model.feature_name() != feature_names:
            print(f"ERROR: Feature mismatch between models!")
            print(f"  {models[0][0]}: {feature_names}")
            print(f"  {path}: {model.feature_name()}")
            return
    
    print(f"\n{'='*60}")
    print(f"Generating ensemble header with {len(models)} model(s)")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}")
    
    # Generate C code from models
    generate_c_ensemble(models, args.output, feature_names, args.mode)


def generate_c_ensemble(models, header_path, feature_names, mode):
    """
    Generate C code for ensemble inference from multiple LightGBM models.
    """
    num_models = len(models)
    num_features = len(feature_names)
    
    all_tree_functions = []
    all_model_score_functions = []
    total_trees = 0
    tree_offset = 0
    
    for model_idx, (model_path, model) in enumerate(models):
        model_json = model.dump_model()
        trees = model_json['tree_info']
        num_trees = len(trees)
        total_trees += num_trees
        
        print(f"Model {model_idx + 1}: {num_trees} trees")
        
        # Generate tree functions for this model
        tree_functions, shrinkages = generate_tree_functions(trees, tree_offset)
        all_tree_functions.extend(tree_functions)
        
        # Generate model score function
        model_score_func = generate_model_score_function(model_idx, tree_offset, num_trees, shrinkages)
        all_model_score_functions.append(model_score_func)
        
        tree_offset += num_trees
    
    # Generate ensemble score function
    ensemble_func = generate_ensemble_function(num_models, mode)
    
    # Assemble the header file
    model_paths_str = '\n'.join(f' *   Model {i+1}: {path}' for i, (path, _) in enumerate(models))
    
    header = f"""// Auto-generated LightGBM Ensemble Model
// DO NOT EDIT - regenerate with convert_ranker_to_c.py
// 
// Ensemble of {num_models} model(s), Total trees: {total_trees}, Features: {num_features}
// Mode: {mode}
//
// Source models:
{model_paths_str}

#ifndef LIGHTGBM_ENSEMBLE_H
#define LIGHTGBM_ENSEMBLE_H

#include <math.h>

/*
 * LightGBM Ensemble Model - Pure C Implementation (Header-Only)
 * 
 * Models: {num_models}
 * Total Trees: {total_trees}
 * Features: {num_features}
 *
 * Feature order ({len(feature_names)} features):
{chr(10).join(f' *   [{i:2d}] {name}' for i, name in enumerate(feature_names))}
 *
 * Usage in C++:
 *   #include "{os.path.basename(header_path)}"
 *   double features[{num_features}];
 *   // Fill features in order above
 *   double result = ensemble_predict(features);
 */

// ============================================================
// Configuration
// ============================================================

static const int NUM_FEATURES = {num_features};
static const int NUM_MODELS = {num_models};
static const int TOTAL_TREES = {total_trees};

// ============================================================
// Individual tree functions
// ============================================================

"""
    
    header += '\n\n'.join(all_tree_functions)
    
    header += '\n\n// ============================================================\n'
    header += '// Per-model scoring functions\n'
    header += '// ============================================================\n\n'
    
    header += '\n\n'.join(all_model_score_functions)
    
    header += '\n\n// ============================================================\n'
    header += '// Ensemble prediction (averages all models)\n'
    header += '// ============================================================\n\n'
    
    header += ensemble_func
    
    header += """

#endif // LIGHTGBM_ENSEMBLE_H
"""
    
    # Write file
    output_dir = os.path.dirname(header_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(header_path, 'w') as f:
        f.write(header)
    
    # Summary
    num_lines = header.count('\n')
    
    print(f"\n✅ Generated: {header_path}")
    print(f"   Models: {num_models}")
    print(f"   Total Trees: {total_trees}")
    print(f"   Features: {num_features}")
    print(f"   Lines: {num_lines:,}")
    print(f"   File size: {len(header):,} bytes")
    
    print("\nFeature order (must match this exactly in C++):")
    for i, name in enumerate(feature_names):
        print(f"  [{i:2d}] {name}")
    
    print(f"\n" + "="*60)
    print("Usage in C++:")
    print("="*60)
    print(f'  #include "{os.path.basename(header_path)}"')
    print(f"")
    print(f"  // Fill features array in exact order above")
    print(f"  double features[{num_features}];")
    print(f"  features[0] = {feature_names[0]};")
    if len(feature_names) > 1:
        print(f"  features[1] = {feature_names[1]};")
    print(f"  // ... etc ...")
    print(f"")
    print(f"  double result = ensemble_predict(features);")
    if mode == 'binary':
        print(f"  // result is probability [0, 1]")
        print(f"  if (result >= threshold) {{ /* keep */ }}")
    else:
        print(f"  // result is raw score (ordinal regression)")
        print(f"  int predicted_class = ordinal_to_class(result);")


def generate_tree_functions(trees, tree_offset):
    """Generate C code for tree functions."""
    
    def generate_tree_code(tree_struct, indent=1):
        """Recursively generate C code for a single tree."""
        lines = []
        prefix = "    " * indent
        
        if 'leaf_value' in tree_struct:
            # Leaf node
            lines.append(f"{prefix}return {tree_struct['leaf_value']:.15e};")
        else:
            # Decision node
            split_feature = tree_struct['split_feature']
            threshold = tree_struct['threshold']
            
            lines.append(f"{prefix}if (input[{split_feature}] <= {threshold:.15e}) {{")
            
            # Left child
            if 'left_child' in tree_struct and tree_struct['left_child'] is not None:
                lines.extend(generate_tree_code(tree_struct['left_child'], indent + 1))
            
            lines.append(f"{prefix}}} else {{")
            
            # Right child
            if 'right_child' in tree_struct and tree_struct['right_child'] is not None:
                lines.extend(generate_tree_code(tree_struct['right_child'], indent + 1))
            
            lines.append(f"{prefix}}}")
        
        return lines
    
    tree_functions = []
    shrinkages = []
    
    for i, tree_info in enumerate(trees):
        tree_struct = tree_info['tree_structure']
        shrinkage = tree_struct.get('shrinkage', 1.0)
        shrinkages.append(shrinkage)
        
        tree_idx = tree_offset + i
        func_lines = [f"static inline double tree_{tree_idx}(const double* input) {{"]
        func_lines.extend(generate_tree_code(tree_struct))
        func_lines.append("}")
        tree_functions.append('\n'.join(func_lines))
    
    return tree_functions, shrinkages


def generate_model_score_function(model_idx, tree_offset, num_trees, shrinkages):
    """Generate score function for a single model."""
    lines = [f"static inline double model_{model_idx}_score(const double* input) {{"]
    lines.append("    double sum = 0.0;")
    
    for i in range(num_trees):
        tree_idx = tree_offset + i
        shrinkage = shrinkages[i]
        if shrinkage == 1.0:
            lines.append(f"    sum += tree_{tree_idx}(input);")
        else:
            lines.append(f"    sum += {shrinkage:.6f} * tree_{tree_idx}(input);")
    
    lines.append("    return sum;")
    lines.append("}")
    
    return '\n'.join(lines)


def generate_ensemble_function(num_models, mode):
    """Generate the ensemble averaging function."""
    lines = [
        "static inline double ensemble_predict(const double* input) {",
        "    double sum = 0.0;"
    ]
    
    for i in range(num_models):
        lines.append(f"    sum += model_{i}_score(input);")
    
    if num_models > 1:
        lines.append(f"    double avg = sum / {float(num_models)};")
    else:
        lines.append("    double avg = sum;")
    
    if mode == 'binary':
        # Apply sigmoid for binary classification
        lines.append("    // Apply sigmoid for probability")
        lines.append("    return 1.0 / (1.0 + exp(-avg));")
    else:
        # Raw score for regression
        lines.append("    return avg;")
    
    lines.append("}")
    
    # Add helper for ordinal classification
    if mode == 'regression':
        lines.append("")
        lines.append("// Convert ordinal regression score to class (0-4)")
        lines.append("// Thresholds: [0.3, 1.0, 2.1, 3.3] -> A, AB, B, BC, C")
        lines.append("static inline int ordinal_to_class(double score) {")
        lines.append("    if (score >= 3.3) return 4;  // C")
        lines.append("    if (score >= 2.1) return 3;  // BC")
        lines.append("    if (score >= 1.0) return 2;  // B")
        lines.append("    if (score >= 0.3) return 1;  // AB")
        lines.append("    return 0;  // A")
        lines.append("}")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    main()
