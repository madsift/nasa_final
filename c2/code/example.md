# NASA Crater Detection - Usage Examples

This document provides example usage for both the C++ inference binaries and Python training/export scripts.

---

## Directory Structure

```
c1/code/
├── Dockerfile              # Docker build (builds C++ from cpp/)
├── requirements.txt        # Python dependencies
├── train.sh               # Training/export pipeline script
├── test.sh                # Inference test script
├── example.md             # This file
├── README.md              # Project documentation
│
├── cpp/                   # C++ Inference Engine
│   ├── src/
│   │   ├── test.cpp              # Main inference binary
│   │   ├── eval3_static.cpp      # Evaluation binary with GT scoring
│   │   ├── label.cpp
│   │   └── watershed_static.cpp
│   └── include/
│       ├── label.h
│       ├── watershed_static.h
│       ├── polar.hpp
│       ├── ranking_features_multires.hpp
│       └── lightgbm_ranker.h     # Auto-generated from Python
│
├── python/                # Python Training & Utilities
│   ├── training/          # Model training scripts
│   ├── data/              # Dataset & feature generation
│   ├── models/            # Model architectures
│   ├── features/          # Feature extraction
│   ├── evaluation/        # Scoring utilities
│   └── export/            # ONNX export & C code generation
│
└── models/                # Trained model weights
    ├── best_672x544.onnx
    └── lightgbm_ranker_n.txt
```

---

## C++ Binary Usage

All examples assume you are running from the `c1/code/` directory.

### Test Binary (Inference Only)

The `test` binary runs inference without ground truth evaluation.

```bash
# Basic usage - process validation images
./test --raw-dir /path/to/images --model models/best_672x544.onnx --tile-size 672x544

# Full example with all options
./test \
    --raw-dir /data/nasa_raw \
    --model models/best_672x544.onnx \
    --input-res 1296 \
    --tile-size 672x544 \
    --overlap 24x32 \
    --limit-craters 13 \
    --solution-out predictions.csv \
    --all \
    --no-ranker

# Process specific validation groups only (default behavior)
./test --raw-dir /data/nasa_raw --model models/best_672x544.onnx --tile-size 672x544

# Process ALL images with ranker enabled
./test --raw-dir /data/nasa_raw --model models/best_672x544.onnx --tile-size 672x544 --all --use-ranker --ranker-thresh 0.1
```

#### Test Binary Options

| Option | Description | Default |
|--------|-------------|---------|
| `--raw-dir` | Directory containing input images | Required |
| `--model` | Path to ONNX model file | Required |
| `--input-res` | Input resolution for resizing | 1024 |
| `--tile-size` | Tile size WxH (e.g., 672x544) | Full image |
| `--overlap` | Tile overlap WxH | Auto-computed |
| `--limit-craters` | Max craters per image | 12 |
| `--solution-out` | Output CSV path | solution.csv |
| `--all` | Process all images | Only val groups |
| `--use-ranker` / `--no-ranker` | Enable/disable ranker | Disabled |
| `--ranker-thresh` | Ranker score threshold | 0.1 |
| `--polar` | Use polar ellipse fitting | Disabled |

### Eval Binary (With Ground Truth Scoring)

The `eval3_static` binary runs inference AND computes NASA scores against ground truth.

```bash
./eval3_static \
    --raw-dir /data/nasa_raw \
    --gt /data/gt.csv \
    --model models/best_672x544.onnx \
    --input-res 1296 \
    --tile-size 672x544
```

---

## Python Script Usage

All Python scripts should be run from the `c1/code/` directory using module syntax.

### 1. Model Training

```bash
# Train segmentation model (multi-GPU)
python -m python.training.train_smp_multi \
    --data_dir ./train_tiles \
    --backbone mobileone_s2 \
    --batch_size 8 \
    --num_epochs 120 \
    --tile_size "672 544" \
    --gpus 2

# Train ranker model
python -m python.training.train_ranker \
    --input_data ../ranking_data/features_full.csv \
    --output models/lightgbm_ranker.txt

# Train classifier
python -m python.training.train_classifier \
    --input ../ranking_data/features_full.csv \
    --cv \
    --tune
```

### 2. ONNX Export

```bash
# Export PyTorch model to ONNX
python -m python.export.export_static_reparam \
    --backbone mobileone_s2 \
    --checkpoint ./checkpoints/best_model.pth \
    --im_dim 3 \
    --size 672x544 \
    --output models/best_672x544.onnx \
    --model_type CraterSMP
```

### 3. Convert Ranker to C Header

```bash
# Generate C header from LightGBM model
python -m python.export.convert_ranker_to_c \
    -i models/lightgbm_ranker.txt \
    -o cpp/include/lightgbm_ranker.h

# Ensemble from multiple folds (classifier)
python -m python.export.convert_ranker_to_c \
    -i viz_classifier/classifier_cv_fold_*.txt \
    -o cpp/include/classifier_ensemble.h \
    --mode regression
```

### 4. Feature Generation

```bash
# Generate crater features for training ranker/classifier
python -m python.data.gen_crater_features \
    --input ../predictions.csv \
    --gt ../gt.csv \
    --output ../ranking_data/features.csv
```

---

## Docker Build & Run

### Build

```bash
cd c1/code
docker build -t nasa-crater .
```

### Run Inference

```bash
docker run --rm -v /path/to/data:/data nasa-crater \
    ./test --raw-dir /data/images --model models/best_672x544.onnx --tile-size 672x544 --all
```

### Interactive Shell

```bash
docker run -it --rm -v /path/to/data:/data nasa-crater
```

---

## Full Training Pipeline

The `train.sh` script runs the complete training pipeline:

```bash
./train.sh
```

This executes:
1. Export PyTorch model to ONNX
2. Train LightGBM ranker
3. Convert ranker to C header

---

## Notes

- **Working Directory**: Always run scripts from `c1/code/`
- **Python Imports**: Scripts use package-style imports (`from python.data.datasets import ...`)
- **C++ Includes**: Source files reference headers via `../include/` paths
- **Docker Build**: Automatically builds C++ binaries during image creation
