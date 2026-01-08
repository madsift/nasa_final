import cv2
import numpy as np
import scipy.fft as fft  # scipy.fft is faster than numpy.fft!
import sys
import time
import cProfile
import pstats

# ---------------------------------------------------------
# In-memory FFT cache (resolution dependent)
# ---------------------------------------------------------
_FFT_CACHE = {}

def _get_fft_cache(h, w):
    key = (h, w)
    if key in _FFT_CACHE:
        return _FFT_CACHE[key]

    ky = fft.fftfreq(h).astype(np.float32)
    kx = fft.fftfreq(w).astype(np.float32)
    ky, kx = np.meshgrid(ky, kx, indexing="ij")

    denom = kx * kx + ky * ky
    denom[0, 0] = 1.0

    _FFT_CACHE[key] = (kx, ky, denom)
    return _FFT_CACHE[key]


# ---------------------------------------------------------
# Fast Pseudo-DEM
# ---------------------------------------------------------
def compute_pseudo_dem(img_gray):
    h, w = img_gray.shape
    kx, ky, denom = _get_fft_cache(h, w)

    img = img_gray.astype(np.float32) * (1.0 / 255.0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    gx_fft = fft.fft2(gx)
    gy_fft = fft.fft2(gy)

    z_fft = (-1j * (kx * gx_fft + ky * gy_fft)) / denom
    dem = fft.ifft2(z_fft).real

    dem -= dem.min()
    dem /= (dem.max() + 1e-6)

    return dem.astype(np.float32)


# ---------------------------------------------------------
# Fast Gradient Map
# ---------------------------------------------------------
def compute_gradient_map(img_gray):
    img = img_gray.astype(np.float32) * (1.0 / 255.0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag /= (mag.max() + 1e-6)

    return mag.astype(np.float32)


# ---------------------------------------------------------
# Combined (FASTEST) - reuses Sobel computation
# ---------------------------------------------------------
def compute_dem_and_gradient(img_gray):
    h, w = img_gray.shape
    kx, ky, denom = _get_fft_cache(h, w)

    img = img_gray.astype(np.float32) * (1.0 / 255.0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude (reusing Sobel)
    grad = cv2.magnitude(gx, gy)
    grad /= (grad.max() + 1e-6)

    # DEM via FFT integration
    gx_fft = fft.fft2(gx)
    gy_fft = fft.fft2(gy)

    z_fft = (-1j * (kx * gx_fft + ky * gy_fft)) / denom
    dem = fft.ifft2(z_fft).real

    dem -= dem.min()
    dem /= (dem.max() + 1e-6)

    return dem.astype(np.float32), grad.astype(np.float32)


# ---------------------------------------------------------
# Main / Benchmark
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sfs_fast_runtime.py <image_path> [--profile]")
        sys.exit(1)

    img_path = sys.argv[1]
    do_profile = "--profile" in sys.argv

    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    h, w = img_gray.shape
    print(f"Image: {w} x {h}")

    # -----------------------------------------------------
    # Warm-up (important for FFT & cache)
    # -----------------------------------------------------
    for _ in range(3):
        compute_dem_and_gradient(img_gray)

    # -----------------------------------------------------
    # Timed runs
    # -----------------------------------------------------
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        dem, grad = compute_dem_and_gradient(img_gray)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / N

    print(f"Average inference time: {avg_ms:.2f} ms ({N} runs)")

    # -----------------------------------------------------
    # Optional profiler
    # -----------------------------------------------------
    if do_profile:
        print("\nRunning cProfile...\n")
        profiler = cProfile.Profile()
        profiler.enable()
        compute_dem_and_gradient(img_gray)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("tottime").print_stats(20)
