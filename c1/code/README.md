# NASA Crater Detection

Docker-based environment for crater detection using deep learning segmentation + LightGBM ranking.

## Directory Structure

```
nasagit/
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Python dependencies
├── .dockerignore          # Exclude files from build
├── code/
│   ├── eval3.cpp          # C++ inference (ONNX Runtime)
│   ├── watershed.cpp/.h   # Watershed segmentation
│   ├── ranking_features.h # Feature extraction for ranker
│   ├── lightgbm_ranker.h  # LightGBM model (pure C header)
│   ├── train_smp_multi.py # Multi-GPU training script
│   ├── models_smp.py      # Segmentation models
│   ├── datasets.py        # Data loading
│   ├── losses.py          # Loss functions
│   └── models/
│       ├── s2_imdim1_768.onnx    # Pre-trained ONNX model
│       └── lightgbm_ranker.txt   # LightGBM ranker model
```

## Building the Docker Image

```bash
# Standard build
docker build -t nasa-crater .

# Build with custom OpenCV/ONNX versions
docker build -t nasa-crater \
    --build-arg OPENCV_VERSION=4.9.0 \
    --build-arg ONNX_VERSION=1.17.0 \
    .
```

## Running

### Python Training

```bash
docker run --gpus all -v /path/to/data:/data nasa-crater \
    python3 train_smp_multi.py --data_dir /data --epochs 100
```

### C++ Inference

```bash
docker run -v /path/to/data:/data nasa-crater \
    /app/eval3 --raw-dir /data --gt /data/craters.csv --model /app/models/s2_imdim1_768.onnx
```

### Interactive Shell

```bash
docker run -it --gpus all -v /path/to/data:/data nasa-crater
```

## C++ Binary Arguments

```
./eval3 [OPTIONS]

Options:
  --raw-dir PATH       Path to raw image directory
  --gt PATH            Path to ground truth CSV
  --model PATH         Path to ONNX model
  --limit-craters N    Maximum craters to output per image (0=no limit)
  --limit-images N     Maximum images to process (0=no limit)
  --ranker-thresh X    Ranker score threshold (default: 0.1)
  --no-ranker          Disable LightGBM ranker filtering
```

## Dependencies

### C++ (built in Docker)
- OpenCV 4.9.0 (core, imgproc, imgcodecs only - ~50MB)
- ONNX Runtime 1.17.0 (pre-built binary)

### Python
- PyTorch 2.0+
- segmentation-models-pytorch
- opencv-python-headless
- albumentations
- lightgbm
- bitsandbytes (8-bit optimization)

## Image Size

The final Docker image is approximately:
- Base: ~2.5 GB (CUDA runtime)
- OpenCV: ~50 MB
- ONNX Runtime: ~150 MB
- Python packages: ~2 GB
- **Total: ~5 GB**

For inference-only (no Python/CUDA), the image can be reduced to ~500 MB.
