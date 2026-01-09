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
# 2. Build ONNX Runtime from source WITH XNNPACK (ARM64-optimized)
#    XNNPACK provides highly optimized ARM NEON kernels for Conv operations
#    Build takes ~30-60 min but provides significant speedup on ARM64
# ---------------------------------------------------------------------------
ARG ONNX_VERSION=1.20.1

# Install additional build dependencies for ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade cmake>=3.28

# Clone ONNX Runtime
RUN git clone --depth 1 --branch v${ONNX_VERSION} --recursive \
    https://github.com/microsoft/onnxruntime.git /build/onnxruntime

WORKDIR /build/onnxruntime

# Build ONNX Runtime with XNNPACK on ARM64, without on x86
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        echo "Building ONNX Runtime with XNNPACK for ARM64..." && \
        ./build.sh --config Release \
            --build_shared_lib \
            --parallel $(nproc) \
            --use_xnnpack \
            --skip_tests \
            --skip_submodule_sync \
            --allow_running_as_root \
            --cmake_extra_defines CMAKE_INSTALL_PREFIX=/usr/local CMAKE_POLICY_VERSION_MINIMUM=3.5 ; \
    else \
        echo "Building ONNX Runtime (no XNNPACK) for x86_64..." && \
        ./build.sh --config Release \
            --build_shared_lib \
            --parallel $(nproc) \
            --skip_tests \
            --skip_submodule_sync \
            --allow_running_as_root \
            --cmake_extra_defines CMAKE_INSTALL_PREFIX=/usr/local CMAKE_POLICY_VERSION_MINIMUM=3.5 ; \
    fi

# Install built libraries
RUN cp -r /build/onnxruntime/build/Linux/Release/libonnxruntime*.so* /usr/local/lib/ && \
    cp -r /build/onnxruntime/include/onnxruntime /usr/local/include/ && \
    rm -rf /build/onnxruntime

WORKDIR /build

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
# Uses Graviton3/M8G optimized flags: fp16+dotprod for faster vectorization
# On x86_64 this will still compile but without ARM-specific opts
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        MARCH_FLAG="-march=armv8.2-a+fp16+dotprod -ffast-math -flto"; \
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
COPY models/*.pth /app/models/

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
