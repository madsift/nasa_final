# ============================================================================
# NASA Crater Detection - Docker Build
# 
# Multi-stage build for minimal image size:
#   - Stage 1: Build C++ components
#   - Stage 2: Runtime with Python + C++ binaries
#
# Build: docker build -t nasa-crater .
# Run:   docker run --gpus all -v /path/to/data:/data nasa-crater
#
# LAYER CACHING STRATEGY:
#   - Least frequently changed items first (system deps, pip packages)
#   - Most frequently changed items last (C++ binaries, Python code)
# ============================================================================

# ===========================================================================
# STAGE 1: BUILD C++ COMPONENTS
# ===========================================================================
FROM ubuntu:22.04 AS cpp-builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ---------------------------------------------------------------------------
# 1. Download and build OpenCV (minimal/headless)
# Only include modules needed: core, imgproc, imgcodecs
# ---------------------------------------------------------------------------
ARG OPENCV_VERSION=4.9.0

RUN wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
    && unzip -q opencv.zip \
    && rm opencv.zip \
    && mkdir opencv-build && cd opencv-build \
    && cmake ../opencv-${OPENCV_VERSION} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_LIST=core,imgproc,imgcodecs \
        -DWITH_TBB=OFF \
        -DWITH_V4L=OFF \
        -DWITH_QT=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF \
        -DWITH_1394=OFF \
        -DWITH_OPENCL=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf opencv-${OPENCV_VERSION} opencv-build

# ---------------------------------------------------------------------------
# 2. Download ONNX Runtime (pre-built binary - fastest)
#    Auto-detect architecture: x64 for Intel/AMD, aarch64 for ARM64
# ---------------------------------------------------------------------------
ARG ONNX_VERSION=1.17.0

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        ONNX_ARCH="x64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        ONNX_ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    echo "Downloading ONNX Runtime for ${ONNX_ARCH}..." && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}.tgz \
    && tar xzf onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}.tgz \
    && cp -r onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}/lib/* /usr/local/lib/ \
    && rm -rf onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}*

RUN ldconfig

# ---------------------------------------------------------------------------
# 3. Copy and build C++ code
# ---------------------------------------------------------------------------
WORKDIR /build/code

# Copy C++ source files from new directory structure
COPY cpp/src/eval3_static.cpp cpp/src/test.cpp cpp/src/test_arm64.cpp cpp/src/watershed_static.cpp ./src/
COPY cpp/include/watershed_static.h cpp/include/label.h cpp/include/lightgbm_ranker.h \
     cpp/include/ranking_features_multires.hpp cpp/include/polar.hpp ./include/

# Build eval3_static (scoring/evaluation binary)
RUN g++ -O3 -std=c++17 \
    -o eval3_static \
    src/eval3_static.cpp src/watershed_static.cpp \
    -I./include \
    -I/usr/local/include/opencv4 \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -lonnxruntime \
    -lstdc++fs \
    && strip eval3_static

# Build test (inference-only binary, no GT loading)
RUN g++ -O3 -std=c++17 \
    -o test \
    src/test.cpp src/watershed_static.cpp \
    -I./include \
    -I/usr/local/include/opencv4 \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -lonnxruntime \
    -lstdc++fs \
    && strip test

# Build test_arm64 (ARM64-optimized inference binary)
# Uses -march=native on ARM64 for NEON SIMD optimizations
# On x86_64 this will still compile but without ARM-specific opts
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MARCH_FLAG="-march=native"; \
    else \
        MARCH_FLAG=""; \
    fi && \
    g++ -O3 -std=c++17 $MARCH_FLAG \
    -o test_arm64 \
    src/test_arm64.cpp src/watershed_static.cpp \
    -I./include \
    -I/usr/local/include/opencv4 \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -lonnxruntime \
    -lstdc++fs \
    && strip test_arm64


# ===========================================================================
# STAGE 2: RUNTIME IMAGE
# ===========================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    libpng16-16 \
    libjpeg8 \
    libtiff5 \
    libgomp1 \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.12, and python points to python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install pip for Python 3.12
RUN python3 -m ensurepip --upgrade && python3 -m pip install --upgrade pip

# Create non-root user
RUN useradd -m -u 1000 crater
WORKDIR /app

# ---------------------------------------------------------------------------
# Copy OpenCV and ONNX Runtime libraries from builder stage
# (These rarely change - good for caching)
# ---------------------------------------------------------------------------
COPY --from=cpp-builder /usr/local/lib/libopencv_* /usr/local/lib/
COPY --from=cpp-builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# ---------------------------------------------------------------------------
# Install Python dependencies FIRST (changes rarely)
# This layer is cached unless requirements.txt changes
# ---------------------------------------------------------------------------
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy models (changes occasionally)
# ---------------------------------------------------------------------------
COPY models/*.onnx /app/models/
COPY models/*.txt /app/models/

# ---------------------------------------------------------------------------
# Copy C++ binaries (changes when C++ code changes)
# Placed AFTER pip install so pip cache is preserved
# ---------------------------------------------------------------------------
COPY --from=cpp-builder /build/code/eval3_static /app/
COPY --from=cpp-builder /build/code/test /app/
COPY --from=cpp-builder /build/code/test_arm64 /app/

# ---------------------------------------------------------------------------
# Copy application code (changes most frequently)
# Placed LAST so all previous layers remain cached
# ---------------------------------------------------------------------------
COPY python/ /app/python/
COPY *.sh /app/
COPY *.yaml /app/

# ---------------------------------------------------------------------------
# Set ownership and permissions
# ---------------------------------------------------------------------------
RUN chown -R crater:crater /app && \
    chmod +x /app/test.sh /app/train.sh

USER crater
# Default environment
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Entry point
ENTRYPOINT ["/bin/bash"]
