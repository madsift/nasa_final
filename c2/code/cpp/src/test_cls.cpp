/**
 * test_cls.cpp - Inference-only crater detection pipeline with CLASSIFICATION
 * 
 * Based on test.cpp with added crater morphology classification.
 * Features:
 * - Simple progress bar (no per-image output)
 * - --polar flag to use polar ellipse fitting from polar.hpp
 * - Uses ranking_features_multires.hpp for ranker features
 * - Crater classification using LightGBM ensemble (A/AB/B/BC/C classes)
 * - Final timing and memory reporting
 * 
 * Usage:
 *   ./test_cls --raw-dir /path/to/images --model /path/to/model.onnx [--polar] [--tile-size 544x416]
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <tuple>
#include <filesystem>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include "watershed_static.h"  // Static memory version
#include "label.h"             // For skimage-compatible regionprops
#include "lightgbm_ranker.h"   // Auto-generated LightGBM ranker model (pure C)
#include "classifier_ensemble.h"  // Auto-generated LightGBM classifier ensemble (pure C)
#include "ranking_features_multires.hpp"  // Multi-resolution ranking features (has full feature extraction)
#include "polar.hpp"           // Polar ellipse fitting

// Toggle between contour-based (OpenCV) and skimage-style regionprops
// Set to 1 to use skimage-style regionprops, 0 to use OpenCV findContours
#define USE_SKIMAGE_REGIONPROPS 1

// Toggle between custom label_skimage() and cv::connectedComponents
// Set to 1 to use label_skimage (matching skimage.measure.label exactly)
// Set to 0 to use cv::connectedComponents with 8-connectivity
#define USE_CUSTOM_LABEL 1

// Linux memory tracking
#include <unistd.h>  // For sysconf

namespace fs = std::filesystem;

// --- MATH CONSTANTS ---
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// --- CONFIGURATION ---
const int BATCH_SIZE = 1;
const int CHANNELS = 3;
const int OUT_CLASSES = 3; 

// --- TILE-BASED STATIC BUFFER SIZING ---
// Maximum tile dimensions used for inference (672x544 is the largest tile size)
// Using tile-sized buffers instead of full-image buffers reduces peak RSS significantly
const int MAX_TILE_W = 672;  // Maximum tile width
const int MAX_TILE_H = 544;  // Maximum tile height
const int MAX_TILE_PIXELS = MAX_TILE_W * MAX_TILE_H;  // 365,568 pixels

// Static buffers sized for single tile (used by tiled_inference)
// Input: CHANNELS * MAX_TILE_PIXELS = 3 * 365,568 = 1,096,704 floats (~4.2 MB)
// Output: OUT_CLASSES * MAX_TILE_PIXELS = 3 * 365,568 = 1,096,704 floats (~4.2 MB)
const int TILE_INPUT_ELEMENT_COUNT = BATCH_SIZE * CHANNELS * MAX_TILE_PIXELS;
const int TILE_OUTPUT_ELEMENT_COUNT = BATCH_SIZE * OUT_CLASSES * MAX_TILE_PIXELS;
static float static_tile_input[TILE_INPUT_ELEMENT_COUNT];
static float static_tile_output[TILE_OUTPUT_ELEMENT_COUNT];

// --- STATIC ACCUMULATORS FOR TILED BLENDING ---
// Maximum image dimensions after downsampling (1296x1024 for input_res=1296)
const int MAX_IMG_W = 1296;
const int MAX_IMG_H = 1024;
const int MAX_IMG_PIXELS = MAX_IMG_W * MAX_IMG_H;  // 1,327,104 pixels

// Static accumulators for tile blending (avoids per-image heap allocation)
// Output: OUT_CLASSES * MAX_IMG_PIXELS = 3 * 1,327,104 = 3,981,312 floats (~15.2 MB)
// Weight: 1 * MAX_IMG_PIXELS = 1,327,104 floats (~5.1 MB)
static float static_output_accum[OUT_CLASSES * MAX_IMG_PIXELS];
static float static_weight_accum[MAX_IMG_PIXELS];

// NOTE: Full-image mode is deprecated - tiling is always required.
// No static buffers are allocated for full-image inference.

// --- EXTRACTION PARAMETERS ---
const float FIXED_SCALE = 1.20f;
const float FIXED_THRESH = 0.6f;
const float MIN_CONFIDENCE = 0.7f;      // Minimum confidence for crater extraction
const float MIN_CONFIDENCE_MIN = 0.6f;   // Floor for iterative confidence lowering

// List of specific prefixes to process
const std::vector<std::string> TARGET_PREFIXES = {
    "altitude01_longitude10", "altitude01_longitude14",
    "altitude04_longitude13", "altitude06_longitude13",
    "altitude09_longitude08"
};

// --- DATA STRUCTURES ---
// Crater struct is defined in watershed_static.h

struct Args {
    std::string raw_data_dir; 
    std::string model_path;
    std::string solution_out = "solution.csv";  // Output CSV path
    int input_res = 1024 ; //1296;   // Input resolution (default 1296 to match eval3_static)
    int limit_craters = 12;  // 0 = no limit
    int limit_images = 0;   // 0 = no limit (for memory testing)
    float ranker_thresh = 0.1f;  // Ranker score threshold (default 0.1 as in Python)
    bool use_ranker = false;  // Enable/disable ranker (match eval3.cpp default)
    bool use_polar = false;   // Use polar ellipse fitting instead of cv2
    bool process_all = false; // Process all images (skip TARGET_PREFIXES filtering)
    bool use_classifier = true;  // Enable crater classification (A/AB/B/BC/C)
    
    // Tiling configuration (tile_w > 0 enables tiling)
    int tile_w = 0;         // Tile width (0 = no tiling, use full padded image)
    int tile_h = 0;         // Tile height
    int overlap_w = 0;      // Horizontal overlap between tiles
    int overlap_h = 0;      // Vertical overlap between tiles
};

// =================================================================
// HELPER FUNCTIONS 
// =================================================================

// --- PROGRESS BAR ---
void print_progress_bar(int current, int total, int bar_width = 50) {
    float progress = (float)current / total;
    int filled = (int)(progress * bar_width);
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) std::cout << "=";
        else if (i == filled) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0f) << "% "
              << "(" << current << "/" << total << ")";
    std::cout.flush();
}

// --- MEMORY TRACKING (Linux /proc/self/status) ---
struct MemoryStats {
    long baseline_kb = 0;
    long peak_kb = 0;
    std::vector<long> samples_kb;
};

// Global memory log file (opened in main, closed at end)
static std::ofstream g_memory_log;
static bool g_memory_log_initialized = false;

void init_memory_log(const std::string& log_path = "memory_debug.log") {
    g_memory_log.open(log_path, std::ios::out | std::ios::trunc);
    if (g_memory_log.is_open()) {
        g_memory_log_initialized = true;
        // Write CSV header
        g_memory_log << "timestamp,image_idx,filename,stage,rss_mb,delta_from_baseline_mb,peak_mb\n";
        g_memory_log.flush();
        std::cerr << "[INFO] Memory log initialized: " << log_path << std::endl;
    } else {
        std::cerr << "[WARN] Could not open memory log file: " << log_path << std::endl;
    }
}

void close_memory_log() {
    if (g_memory_log_initialized && g_memory_log.is_open()) {
        g_memory_log.close();
        g_memory_log_initialized = false;
    }
}

long get_current_rss_kb() {
    // Read from /proc/self/status (Linux only)
    std::ifstream status_file("/proc/self/status");
    std::string line;
    long rss_kb = 0;
    
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            iss >> rss_kb;  // Value is in kB
            break;
        }
    }
    return rss_kb;
}

void update_memory_stats(MemoryStats& stats) {
    long current = get_current_rss_kb();
    // Only store every 100th sample to prevent samples vector from growing large
    if (stats.samples_kb.size() < 200 || stats.samples_kb.size() % 100 == 0) {
        stats.samples_kb.push_back(current);
    }
    if (current > stats.peak_kb) {
        stats.peak_kb = current;
    }
}

void print_memory_report(const MemoryStats& stats) {
    std::cout << "\n--- MEMORY USAGE (RSS) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Baseline:       " << (stats.baseline_kb / 1024.0) << " MB" << std::endl;
    std::cout << "Peak:           " << (stats.peak_kb / 1024.0) << " MB" << std::endl;
    
    if (!stats.samples_kb.empty()) {
        long sum = 0;
        long min_val = stats.samples_kb[0];
        long max_val = stats.samples_kb[0];
        for (long v : stats.samples_kb) {
            sum += v;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        double mean = (double)sum / stats.samples_kb.size();
        std::cout << "Mean during:    " << (mean / 1024.0) << " MB" << std::endl;
        std::cout << "Min during:     " << (min_val / 1024.0) << " MB" << std::endl;
        std::cout << "Max during:     " << (max_val / 1024.0) << " MB" << std::endl;
    }
    std::cout << "---------------------------" << std::endl;
    
    // Also write summary to log file
    if (g_memory_log_initialized) {
        g_memory_log << "# SUMMARY: baseline=" << (stats.baseline_kb / 1024.0) 
                     << "MB, peak=" << (stats.peak_kb / 1024.0) << "MB\n";
        g_memory_log.flush();
    }
}

// DEBUG: Print detailed memory breakdown when threshold exceeded
const long MEMORY_DEBUG_THRESHOLD_KB = 400 * 1024;  // 400 MB in KB

// Log memory to file (always) and stderr (only if threshold exceeded)
void log_memory(int img_idx, const std::string& filename, const std::string& stage, 
                long baseline_kb, long peak_kb) {
    long current_kb = get_current_rss_kb();
    double current_mb = current_kb / 1024.0;
    double delta_mb = (current_kb - baseline_kb) / 1024.0;
    double peak_mb = peak_kb / 1024.0;
    
    // Always write to log file (CSV format for easy parsing)
    if (g_memory_log_initialized) {
        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        
        g_memory_log << std::put_time(std::localtime(&time_t_now), "%H:%M:%S") << ","
                     << img_idx << ","
                     << filename << ","
                     << stage << ","
                     << std::fixed << std::setprecision(1) << current_mb << ","
                     << delta_mb << ","
                     << peak_mb << "\n";
        
        // Flush every 10 images to ensure data is written
        if (img_idx % 10 == 0) {
            g_memory_log.flush();
        }
    }
    
    // Only print to stderr if threshold exceeded
    if (current_kb > MEMORY_DEBUG_THRESHOLD_KB) {
        std::cerr << "\n[MEM DEBUG] Image #" << img_idx << " (" << filename << ") @ " << stage << ":\n";
        std::cerr << "  Current RSS: " << std::fixed << std::setprecision(1) << current_mb << " MB\n";
        std::cerr << "  Delta from baseline: " << delta_mb << " MB\n";
        std::cerr.flush();
    }
}

// Wrapper for backward compatibility
void print_debug_memory(int img_idx, const std::string& filename, const std::string& stage, long baseline_kb) {
    // Use default peak of 0 for legacy calls
    static long last_peak = 0;
    long current = get_current_rss_kb();
    if (current > last_peak) last_peak = current;
    log_memory(img_idx, filename, stage, baseline_kb, last_peak);
}

// =================================================================
// NUMERICAL EQUIVALENT of sfs_fast.py compute_dem_and_gradient()
// =================================================================
// Returns: (dem, gradient) both as CV_32F matrices in [0, 1] range

std::pair<cv::Mat, cv::Mat> compute_dem_and_gradient_cpp(const cv::Mat& img_gray) {
    const int H = img_gray.rows;
    const int W = img_gray.cols;
    
    // Step 1: Convert to float [0, 1]
    cv::Mat img_float;
    img_gray.convertTo(img_float, CV_32F, 1.0 / 255.0);
    
    // Step 2: Compute Sobel gradients (same as Python cv2.Sobel)
    cv::Mat gx, gy;
    cv::Sobel(img_float, gx, CV_32F, 1, 0, 3);  // dx
    cv::Sobel(img_float, gy, CV_32F, 0, 1, 3);  // dy
    
    // Step 3: Gradient magnitude (normalized)
    cv::Mat grad;
    cv::magnitude(gx, gy, grad);
    double grad_max;
    cv::minMaxLoc(grad, nullptr, &grad_max);
    grad /= (grad_max + 1e-6);
    
    // Step 4: Build frequency grids matching scipy.fft.fftfreq exactly
    cv::Mat kx = cv::Mat::zeros(H, W, CV_32F);
    cv::Mat ky = cv::Mat::zeros(H, W, CV_32F);
    
    // Pre-compute frequency vectors
    std::vector<float> freq_h(H), freq_w(W);
    for (int i = 0; i < H; ++i) {
        freq_h[i] = (i < (H + 1) / 2) ? (float)i / H : (float)(i - H) / H;
    }
    for (int j = 0; j < W; ++j) {
        freq_w[j] = (j < (W + 1) / 2) ? (float)j / W : (float)(j - W) / W;
    }
    
    // meshgrid with indexing='ij': ky varies along rows (first dim), kx varies along cols
    for (int i = 0; i < H; ++i) {
        float* ky_row = ky.ptr<float>(i);
        float* kx_row = kx.ptr<float>(i);
        for (int j = 0; j < W; ++j) {
            ky_row[j] = freq_h[i];
            kx_row[j] = freq_w[j];
        }
    }
    
    // Step 5: Denominator: kx^2 + ky^2, with [0,0] = 1.0 to avoid division by zero
    cv::Mat denom = kx.mul(kx) + ky.mul(ky);
    denom.at<float>(0, 0) = 1.0f;
    
    // Step 6: FFT of gx and gy
    cv::Mat gx_complex, gy_complex;
    {
        // Scope temporary arrays so they get released after merge
        cv::Mat gx_planes[] = {gx.clone(), cv::Mat::zeros(H, W, CV_32F)};
        cv::Mat gy_planes[] = {gy.clone(), cv::Mat::zeros(H, W, CV_32F)};
        cv::merge(gx_planes, 2, gx_complex);
        cv::merge(gy_planes, 2, gy_complex);
        // gx_planes and gy_planes go out of scope here
    }
    cv::dft(gx_complex, gx_complex);
    cv::dft(gy_complex, gy_complex);
    
    // Release gx, gy - no longer needed  
    gx.release();
    gy.release();
    
    cv::Mat gx_fft[2], gy_fft[2];
    cv::split(gx_complex, gx_fft);
    cv::split(gy_complex, gy_fft);
    
    // Release original complex buffers after split
    gx_complex.release();
    gy_complex.release();
    
    // Step 7: z_fft = -1j * (kx * gx_fft + ky * gy_fft) / denom
    cv::Mat A_real = kx.mul(gx_fft[0]) + ky.mul(gy_fft[0]);
    cv::Mat A_imag = kx.mul(gx_fft[1]) + ky.mul(gy_fft[1]);
    
    // Release kx, ky, gx_fft, gy_fft
    kx.release();
    ky.release();
    gx_fft[0].release();
    gx_fft[1].release();
    gy_fft[0].release();
    gy_fft[1].release();
    
    cv::Mat z_fft_real = A_imag / denom;
    cv::Mat z_fft_imag = -A_real / denom;
    
    // Release intermediates
    A_real.release();
    A_imag.release();
    denom.release();
    
    // Step 8: Inverse FFT to get DEM
    cv::Mat z_complex;
    {
        cv::Mat z_planes[] = {z_fft_real, z_fft_imag};
        cv::merge(z_planes, 2, z_complex);
        // z_fft_real and z_fft_imag still hold refs until after merge
    }
    z_fft_real.release();
    z_fft_imag.release();
    
    cv::idft(z_complex, z_complex, cv::DFT_SCALE);
    
    cv::Mat z_result[2];
    cv::split(z_complex, z_result);
    z_complex.release();
    
    // CRITICAL: Clone the DEM to own its data independently
    // z_result[0] shares memory with z_complex internals
    cv::Mat dem = z_result[0].clone();
    z_result[0].release();
    z_result[1].release();  // Release imaginary part we don't use
    
    // Step 9: Normalize DEM to [0, 1]
    double dem_min, dem_max;
    cv::minMaxLoc(dem, &dem_min, &dem_max);
    dem = dem - dem_min;
    cv::minMaxLoc(dem, nullptr, &dem_max);
    dem /= (dem_max + 1e-6);
    
    // Clean up img_float
    img_float.release();
    
    // Return independent clones  
    return {dem, grad.clone()};
}

// =================================================================
// TILED INFERENCE (Memory Efficient, matches Python tiled_inference)
// =================================================================

// Create Gaussian weight matrix for smooth tile blending
// Create Gaussian weight matrix for smooth tile blending
cv::Mat create_gaussian_weight(int tile_h, int tile_w, float sigma_ratio = 0.25f) {
    int center_y = tile_h / 2;
    int center_x = tile_w / 2;
    float sigma_y = tile_h * sigma_ratio;
    float sigma_x = tile_w * sigma_ratio;
    
    cv::Mat weight(tile_h, tile_w, CV_32F);
    for (int y = 0; y < tile_h; ++y) {
        float* row = weight.ptr<float>(y);
        for (int x = 0; x < tile_w; ++x) {
            float dy = y - center_y;
            float dx = x - center_x;
            row[x] = std::exp(-(dx*dx / (2*sigma_x*sigma_x) + dy*dy / (2*sigma_y*sigma_y)));
        }
    }
    return weight;
}

// Create Tukey weight matrix for seamless tile blending
cv::Mat create_tukey_weight(int tile_h, int tile_w, int overlap_h, int overlap_w)
{
    const float pi = 3.14159265358979323846f;

    // alpha = fraction of window tapered (both ends)
    float alpha_h = tile_h > 0
        ? std::min(2.0f * overlap_h / tile_h, 1.0f)
        : 0.0f;

    float alpha_w = tile_w > 0
        ? std::min(2.0f * overlap_w / tile_w, 1.0f)
        : 0.0f;

    auto tukey_1d = [&](int N, float alpha, std::vector<float>& w)
    {
        w.resize(N, 1.0f);
        if (N <= 1 || alpha <= 0.0f)
            return;

        float edge = alpha * (N - 1) * 0.5f;

        for (int n = 0; n < N; ++n) {
            if (n < edge) {
                w[n] = 0.5f * (1.0f +
                    std::cos(pi * (2.0f * n / (alpha * (N - 1)) - 1.0f)));
            }
            else if (n > (N - 1 - edge)) {
                w[n] = 0.5f * (1.0f +
                    std::cos(pi * (2.0f * n / (alpha * (N - 1)) - 2.0f / alpha + 1.0f)));
            }
            else {
                w[n] = 1.0f;
            }
        }
    };

    std::vector<float> wy, wx;
    tukey_1d(tile_h, alpha_h, wy);
    tukey_1d(tile_w, alpha_w, wx);

    cv::Mat weight(tile_h, tile_w, CV_32F);

    for (int y = 0; y < tile_h; ++y) {
        float* row = weight.ptr<float>(y);
        for (int x = 0; x < tile_w; ++x) {
            row[x] = wy[y] * wx[x];
        }
    }

    return weight;
}

// Tiled inference: process image as overlapping tiles with Gaussian blending
// Input: input_planes is a vector of normalized cv::Mat (one per channel, CV_32F)
// Returns: output tensor [OUT_CLASSES, H, W] after sigmoid
// MEMORY OPTIMIZATION: Uses static tile buffers and static accumulators
std::vector<cv::Mat> tiled_inference(
    Ort::Session& session,
    const Ort::MemoryInfo& memory_info,
    const std::vector<cv::Mat>& input_planes,  // Vector of normalized planes
    int tile_h, int tile_w,
    int overlap_h, int overlap_w
) {
    const int n_channels = input_planes.size();
    const int H = input_planes[0].rows;
    const int W = input_planes[0].cols;
    
    // Validate tile dimensions against static buffer limits
    if (tile_h > MAX_TILE_H || tile_w > MAX_TILE_W) {
        fprintf(stderr, "[ERROR] Tile size %dx%d exceeds static buffer limit %dx%d\n",
                tile_w, tile_h, MAX_TILE_W, MAX_TILE_H);
        return {};
    }
    
    // Validate image dimensions against static accumulator limits
    if (H > MAX_IMG_H || W > MAX_IMG_W) {
        fprintf(stderr, "[ERROR] Image size %dx%d exceeds static accumulator limit %dx%d\n",
                W, H, MAX_IMG_W, MAX_IMG_H);
        return {};
    }
    
    const int stride_h = tile_h - overlap_h;
    const int stride_w = tile_w - overlap_w;
    
    // Calculate grid
    int n_rows = (stride_h > 0) ? ((H - tile_h + stride_h - 1) / stride_h + 1) : 1;
    int n_cols = (stride_w > 0) ? ((W - tile_w + stride_w - 1) / stride_w + 1) : 1;
    
    // Zero static accumulators (only the region we'll use: H x W)
    const int img_pixels = H * W;
    std::memset(static_output_accum, 0, OUT_CLASSES * img_pixels * sizeof(float));
    std::memset(static_weight_accum, 0, img_pixels * sizeof(float));
    
    // Pre-compute Tukey weight for blending (small, okay to be dynamic)
    cv::Mat tukey_weight = create_tukey_weight(tile_h, tile_w, overlap_h, overlap_w);
    
    // Use static tile buffers (sized for MAX_TILE_W x MAX_TILE_H)
    const int tile_pixels = tile_h * tile_w;
    
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    int64_t input_shape[] = {1, n_channels, tile_h, tile_w};
    int64_t output_shape[] = {1, OUT_CLASSES, tile_h, tile_w};
    
    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            // Calculate tile coordinates (clamp to valid region)
            int y1 = std::min(row * stride_h, H - tile_h);
            int x1 = std::min(col * stride_w, W - tile_w);
            y1 = std::max(0, y1);
            x1 = std::max(0, x1);
            int y2 = y1 + tile_h;
            int x2 = x1 + tile_w;
            
            // Fill static input buffer (planar NCHW format) for each channel
            int k = 0;
            for (int ch = 0; ch < n_channels; ++ch) {
                cv::Mat tile = input_planes[ch](cv::Rect(x1, y1, tile_w, tile_h));
                
                // Handle edge case padding
                cv::Mat tile_padded;
                if (tile.rows < tile_h || tile.cols < tile_w) {
                    int pad_b = tile_h - tile.rows;
                    int pad_r = tile_w - tile.cols;
                    cv::copyMakeBorder(tile, tile_padded, 0, pad_b, 0, pad_r, cv::BORDER_CONSTANT, 0);
                } else {
                    tile_padded = tile;
                }
                
                // Copy to static buffer in row-major order
                for (int ty = 0; ty < tile_h; ++ty) {
                    const float* row_ptr = tile_padded.ptr<float>(ty);
                    for (int tx = 0; tx < tile_w; ++tx) {
                        static_tile_input[k++] = row_ptr[tx];
                    }
                }
            }
            
            // Create tensors using static buffers and run inference
            const int input_element_count = n_channels * tile_pixels;
            const int output_element_count = OUT_CLASSES * tile_pixels;
            
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, static_tile_input, input_element_count, input_shape, 4);
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info, static_tile_output, output_element_count, output_shape, 4);
            
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, 
                       output_names, &output_tensor, 1);
            
            // Apply sigmoid and accumulate into static buffers
            int real_h = y2 - y1;
            int real_w = x2 - x1;
            
            for (int c = 0; c < OUT_CLASSES; ++c) {
                float* out_base = static_tile_output + c * tile_pixels;
                float* accum_base = static_output_accum + c * img_pixels;
                
                for (int ty = 0; ty < real_h; ++ty) {
                    int img_y = y1 + ty;
                    for (int tx = 0; tx < real_w; ++tx) {
                        int img_x = x1 + tx;
                        float logit = out_base[ty * tile_w + tx];
                        float sigmoid_val = 1.0f / (1.0f + std::exp(-logit));
                        float w = tukey_weight.at<float>(ty, tx);
                        
                        int img_idx = img_y * W + img_x;
                        accum_base[img_idx] += sigmoid_val * w;
                    }
                }
            }
            
            // Accumulate weight
            for (int ty = 0; ty < real_h; ++ty) {
                int img_y = y1 + ty;
                for (int tx = 0; tx < real_w; ++tx) {
                    int img_x = x1 + tx;
                    float w = tukey_weight.at<float>(ty, tx);
                    int img_idx = img_y * W + img_x;
                    static_weight_accum[img_idx] += w;
                }
            }
        }
    }
    
    // Normalize by weight sum and create output cv::Mat views
    std::vector<cv::Mat> output(OUT_CLASSES);
    for (int c = 0; c < OUT_CLASSES; ++c) {
        // Create cv::Mat that wraps the static buffer (no copy, just a view)
        cv::Mat accum_view(H, W, CV_32F, static_output_accum + c * img_pixels);
        
        // Normalize in-place
        for (int y = 0; y < H; ++y) {
            float* out_row = accum_view.ptr<float>(y);
            const float* weight_row = static_weight_accum + y * W;
            for (int x = 0; x < W; ++x) {
                if (weight_row[x] > 1e-8f) {
                    out_row[x] /= weight_row[x];
                }
            }
        }
        
        // Clone to create independent output (static buffer will be reused)
        output[c] = accum_view.clone();
    }
    
    // Release the tukey weight matrix
    tukey_weight.release();
    
    return output;
}

// =================================================================
// POLAR ELLIPSE FITTING WRAPPER
// =================================================================

// Refit a crater using polar ellipse fitting
bool refit_crater_polar(
    Crater& crater,
    const cv::Mat& rim_prob,    // CV_32F rim probability
    const cv::Mat& core_prob,   // CV_32F core probability (optional)
    int h_orig, int w_orig,
    int num_bins = 360,
    double min_support = 0.5
) {
    // Get rim points near the ellipse
    std::vector<cv::Point2f> rim_pts = RankingFeatures::get_rim_points_for_candidate_fast(
        rim_prob, crater.x, crater.y, crater.a, crater.b, crater.angle,
        48, 5, 0.2f
    );
    
    if (rim_pts.size() < 20) {
        return false;  // Not enough points
    }
    
    // Get core points if available
    std::vector<cv::Point2f> core_pts;
    if (!core_prob.empty()) {
        core_pts = RankingFeatures::get_rim_points_for_candidate_fast(
            core_prob, crater.x, crater.y, crater.a * 0.7f, crater.b * 0.7f, crater.angle,
            48, 5, 0.2f
        );
    }
    
    // Fit using polar method
    Polar::EllipseFitResult fit = Polar::fit_crater_ellipse_polar(
        rim_pts, core_pts, 
        cv::Point2f(crater.x, crater.y),
        num_bins, min_support
    );
    
    if (!fit.valid || fit.a < 3 || fit.b < 3) {
        return false;  // Fitting failed
    }
    
    // Boundary check
    if (fit.cx - fit.a < 0 || fit.cx + fit.a >= w_orig ||
        fit.cy - fit.b < 0 || fit.cy + fit.b >= h_orig) {
        return false;  // Out of bounds
    }
    
    // Size ratio check
    float max_dim = std::max(w_orig, h_orig);
    if (std::max(fit.a, fit.b) > max_dim * 0.6) {
        return false;  // Too large
    }
    
    // Update crater with polar fit results
    crater.x = (float)fit.cx;
    crater.y = (float)fit.cy;
    crater.a = (float)fit.a;
    crater.b = (float)fit.b;
    crater.angle = (float)(fit.phi * 180.0 / M_PI);  // Convert to degrees
    
    return true;
}

// =================================================================
// IMAGE PATH MATCHING
// =================================================================

bool matches_target_prefix(const std::string& path) {
    for (const auto& prefix : TARGET_PREFIXES) {
        size_t underscore_pos = prefix.find('_');
        if (underscore_pos == std::string::npos) continue;
        std::string alt_part = prefix.substr(0, underscore_pos);
        std::string lon_part = prefix.substr(underscore_pos + 1);
        
        // Use exact directory matching with slashes to avoid substring issues
        // e.g., "/altitude01/" and "/longitude10/" instead of just "altitude01"
        std::string alt_dir = "/" + alt_part + "/";
        std::string lon_dir = "/" + lon_part + "/";
        
        if (path.find(alt_dir) != std::string::npos && path.find(lon_dir) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// =================================================================
// ARGUMENT PARSING
// =================================================================

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--raw-dir" && i + 1 < argc) args.raw_data_dir = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.model_path = argv[++i];
        else if (arg == "--limit-craters" && i + 1 < argc) args.limit_craters = std::stoi(argv[++i]);
        else if (arg == "--limit-images" && i + 1 < argc) args.limit_images = std::stoi(argv[++i]);
        else if (arg == "--ranker-thresh" && i + 1 < argc) args.ranker_thresh = std::stof(argv[++i]);
        else if (arg == "--no-ranker") args.use_ranker = false;
        else if (arg == "--use-ranker") args.use_ranker = true;
        else if (arg == "--polar") args.use_polar = true;
        else if (arg == "--no-classifier") args.use_classifier = false;
        else if (arg == "--use-classifier") args.use_classifier = true;
        else if (arg == "--solution-out" && i + 1 < argc) args.solution_out = argv[++i];
        else if (arg == "--input-res" && i + 1 < argc) args.input_res = std::stoi(argv[++i]);
        else if (arg == "--all") args.process_all = true;
        else if (arg == "--tile-size" && i + 1 < argc) {
            // Parse WxH format (e.g., "544x416" or "544,416" or just "544" for square)
            std::string ts = argv[++i];
            size_t sep = ts.find_first_of("x,");
            if (sep != std::string::npos) {
                args.tile_w = std::stoi(ts.substr(0, sep));
                args.tile_h = std::stoi(ts.substr(sep + 1));
            } else {
                args.tile_w = args.tile_h = std::stoi(ts);
            }
        }
        else if (arg == "--overlap" && i + 1 < argc) {
            // Parse WxH format for overlap
            std::string ov = argv[++i];
            size_t sep = ov.find_first_of("x,");
            if (sep != std::string::npos) {
                args.overlap_w = std::stoi(ov.substr(0, sep));
                args.overlap_h = std::stoi(ov.substr(sep + 1));
            } else {
                args.overlap_w = args.overlap_h = std::stoi(ov);
            }
        }
    }
    
    // Default overlap: 32 pixels if not specified (matches Python default)
    // Default overlap based on tile size and input_res configuration
    if (args.tile_w > 0 && args.overlap_w == 0) {
        if (args.tile_w == 544 && args.tile_h == 416 && args.input_res == 1024) {
            // Configuration: 544x416 tiles with 1024 input_res
            args.overlap_w = 32;
            args.overlap_h = 11;
        } else if (args.tile_w == 672 && args.tile_h == 544 && args.input_res == 1296) {
            // Configuration: 672x544 tiles with 1296 input_res
            args.overlap_w = 24;
            args.overlap_h = 32;
        } else {
            // Default for other configurations
            args.overlap_w = 32;
            args.overlap_h = 32;
        }
    }
    
    return args;
}

// =================================================================
// MAIN
// =================================================================
int main(int argc, char* argv[]) {
    // ============================================
    // DETAILED TIMING TRACKING (matching Python)
    // ============================================
    struct TimingStats {
        double total = 0, max = 0;
        int count = 0;
        void add(double ms) { total += ms; if (ms > max) max = ms; count++; }
        double mean() const { return count > 0 ? total / count : 0; }
    };
    
    TimingStats time_image_loading, time_resizing, time_dem_gradient;
    TimingStats time_model_inference, time_ellipse_extraction, time_ranker_scoring;
    TimingStats time_polar_fitting, time_classification;
    double time_config_loading = 0, time_onnx_init = 0;
    
    auto t_init_start = std::chrono::high_resolution_clock::now();
    
    // -------------------------------------------------------------
    // SINGLE CORE MODE (deployment constraint)
    // -------------------------------------------------------------
    const int NUM_THREADS = 1;
    cv::setNumThreads(NUM_THREADS);
    
    // Initialize static memory buffers for watershed
    init_watershed_static_buffers(2048, 2592);  // Max expected image size
    
    // Initialize memory debug log file
    init_memory_log("memory_debug.log");
    // -------------------------------------------------------------

    auto t0 = std::chrono::high_resolution_clock::now();
    Args args = parse_args(argc, argv);
    if (args.raw_data_dir.empty() || args.model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --raw-dir <path> --model <path> [--polar] [--tile-size WxH]" << std::endl;
        return 1;
    }
    time_config_loading = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // 1. Setup Environment and Load Model
    t0 = std::chrono::high_resolution_clock::now();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CraterInference");
    Ort::SessionOptions session_options;
    
    // Single-threaded execution
    session_options.SetIntraOpNumThreads(NUM_THREADS);
    session_options.SetInterOpNumThreads(1);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    // Enable all graph optimizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // =============================================================
    // MEMORY OPTIMIZATION (Critical for ARM/low memory)
    // =============================================================
    // Disable CPU memory arena - prevents over-allocation (30-50% RSS reduction)
    session_options.DisableCpuMemArena();
    
    // Disable memory pattern optimization - prevents large pre-allocations
    session_options.DisableMemPattern();
    // =============================================================

    Ort::Session session(env, args.model_path.c_str(), session_options);
    
    // Use OrtDeviceAllocator instead of OrtArenaAllocator for lower memory
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    time_onnx_init = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    
    double time_total_init = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_init_start).count();
    
    std::cout << "\n=== CRATER DETECTION INFERENCE ===" << std::endl;
    std::cout << "Model: " << args.model_path << std::endl;
    std::cout << "Ellipse fitting: " << (args.use_polar ? "POLAR" : "CV2") << std::endl;
    std::cout << "Ranker: " << (args.use_ranker ? "enabled" : "disabled") << std::endl;
    std::cout << "Classifier: " << (args.use_classifier ? "enabled" : "disabled") << std::endl;
    if (args.tile_w > 0) {
        std::cout << "Tiling: " << args.tile_w << "x" << args.tile_h 
                  << " overlap " << args.overlap_w << "x" << args.overlap_h << std::endl;
    }
    std::cout << std::endl;

    auto total_start_time = std::chrono::high_resolution_clock::now();
    int images_processed = 0;
    int total_craters = 0;
    
    // Memory tracking
    MemoryStats mem_stats;
    mem_stats.baseline_kb = get_current_rss_kb();
    mem_stats.peak_kb = mem_stats.baseline_kb;
    
    // Data collection for CSV output
    std::vector<std::tuple<std::string, std::vector<Crater>>> all_predictions;
    
    // Classification labels: key is (img_path, crater_index), value is class (0-4)
    // We store this separately since Crater struct doesn't have a classification field
    std::map<std::pair<std::string, int>, int> crater_classifications;

    // Collect all matching image files first
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::recursive_directory_iterator(args.raw_data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            std::string full_path = entry.path().string();
            // Skip truth directory (contains ground truth masks, not input images)
            if (full_path.find("/truth/") != std::string::npos) continue;
            // Filter by TARGET_PREFIXES unless --all is specified
            if (args.process_all || matches_target_prefix(full_path)) {
                image_paths.push_back(full_path);
            }
        }
    }
    
    if (args.limit_images > 0 && (int)image_paths.size() > args.limit_images) {
        image_paths.resize(args.limit_images);
    }
    
    const int total_images = image_paths.size();
    std::cout << "Found " << total_images << " images to process\n" << std::endl;

    for (int img_idx = 0; img_idx < total_images; ++img_idx) {
        const std::string& raw_img_path = image_paths[img_idx];
        const std::string full_filename = fs::path(raw_img_path).filename().string();
        
        // === IMAGE LOADING ===
        auto t_stage = std::chrono::high_resolution_clock::now();
        cv::Mat img_orig = cv::imread(raw_img_path, cv::IMREAD_GRAYSCALE);
        if (img_orig.empty()) continue;
        double t_loading = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_stage).count();
        time_image_loading.add(t_loading);
        
        int h_orig = img_orig.rows;
        int w_orig = img_orig.cols;
        
        // DEBUG: Memory check after image loading
        print_debug_memory(img_idx, full_filename, "after_load", mem_stats.baseline_kb);
        
        auto pipeline_start_time = std::chrono::high_resolution_clock::now();

        // === RESIZING ===
        t_stage = std::chrono::high_resolution_clock::now();
        int resized_w, resized_h;
        float scale_factor;
        if (w_orig >= h_orig) {
            resized_w = args.input_res;
            scale_factor = (float)args.input_res / w_orig;
            resized_h = (int)std::round(h_orig * scale_factor);
        } else {
            resized_h = args.input_res;
            scale_factor = (float)args.input_res / h_orig;
            resized_w = (int)std::round(w_orig * scale_factor);
        }
        
        cv::Mat img_resized;
        cv::resize(img_orig, img_resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
        double t_resize = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_stage).count();
        time_resizing.add(t_resize);
        
        // === DEM + GRADIENT CREATION (matching sfs_fast.py exactly) ===
        t_stage = std::chrono::high_resolution_clock::now();
        auto [dem, grad] = compute_dem_and_gradient_cpp(img_resized);
        double t_dem_grad = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_stage).count();
        time_dem_gradient.add(t_dem_grad);
        
        // DEBUG: Memory check after DEM/gradient (FFT operations can cause leaks)
        print_debug_memory(img_idx, full_filename, "after_dem_grad", mem_stats.baseline_kb);

        // === MODEL INFERENCE (Tiled or Full) ===
        t_stage = std::chrono::high_resolution_clock::now();
        std::vector<cv::Mat> full_planes(OUT_CLASSES);
        
        bool use_tiled = (args.tile_w > 0);
        
        if (use_tiled) {
            // ===============================================
            // TILED INFERENCE (No padding, Gaussian blending)
            // ===============================================
            // Normalize all input channels
            cv::Mat img_float;
            img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);
            cv::Mat img_norm = (img_float - 0.5f) / 0.5f;
            img_float.release();  // No longer needed after normalization
            
            // Build input planes based on CHANNELS configuration
            std::vector<cv::Mat> input_planes;
            if constexpr (CHANNELS == 1) {
                input_planes.push_back(img_norm);
            } else if constexpr (CHANNELS == 2) {
                // Normalize gradient (already in [0,1])
                cv::Mat grad_norm = (grad - 0.5f) / 0.5f;
                input_planes.push_back(img_norm);
                input_planes.push_back(grad_norm);
            } else {
                // 3 channels: img + DEM + gradient
                cv::Mat dem_norm = (dem - 0.5f) / 0.5f;
                cv::Mat grad_norm = (grad - 0.5f) / 0.5f;
                input_planes.push_back(img_norm);
                input_planes.push_back(dem_norm);
                input_planes.push_back(grad_norm);
            }
            
            // Run tiled inference
            std::vector<cv::Mat> tiled_output = tiled_inference(
                session, memory_info, input_planes,
                args.tile_h, args.tile_w,
                args.overlap_h, args.overlap_w
            );
            
            // Release input_planes - no longer needed after inference
            for (auto& plane : input_planes) {
                plane.release();
            }
            input_planes.clear();
            
            // Resize each channel to original resolution
            for (int c = 0; c < OUT_CLASSES; ++c) {
                int interp = (c == 2) ? cv::INTER_NEAREST : cv::INTER_LINEAR;
                cv::resize(tiled_output[c], full_planes[c], cv::Size(w_orig, h_orig), 0, 0, interp);
                tiled_output[c].release();  // Release after resizing
            }
            tiled_output.clear();
        } else {
            // ===============================================
            // FULL IMAGE INFERENCE - DEPRECATED
            // Tiling is always required for memory-efficient inference.
            // ===============================================
            fprintf(stderr, "[ERROR] Full-image mode is disabled. Please use --tile-size to enable tiling.\\n");
            fprintf(stderr, "        Example: --tile-size 672x544 or --tile-size 544x416\\n");
            continue;  // Skip this image
        }
        
        double t_inference = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_stage).count();
        time_model_inference.add(t_inference);
        
        // DEBUG: Memory check after inference (ONNX can have internal memory pools)
        print_debug_memory(img_idx, full_filename, "after_inference", mem_stats.baseline_kb);

        // === ELLIPSE EXTRACTION ===
        t_stage = std::chrono::high_resolution_clock::now();
        // Use extraction function from watershed_static.cpp
        std::vector<Crater> extracted = extract_craters_cv2_adaptive_selection_rescue(
            full_planes,      // vector<cv::Mat> not merged Mat
            FIXED_THRESH,     // 0.75 threshold (matching eval3.cpp)
            0.25f,            // rim_thresh
            0.15f,            // ecc_floor
            1.0f, 1.4f, 40,   // scale_min, scale_max, scale_steps
            MIN_CONFIDENCE, MIN_CONFIDENCE_MIN,  // confidence thresholds
            false,            // nms_filter
            false,            // enable_rescue
            false,            // enable_completeness
            40,               // min_semi_axis
            0.6f,             // max_size_ratio
            true,             // require_fully_visible
            cv::Size(w_orig, h_orig)  // image_shape
        );
        double t_extraction = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_stage).count();
        time_ellipse_extraction.add(t_extraction);
        
        // DEBUG: Memory check after watershed/extraction (allocates RegionProps with pixel vectors)
        print_debug_memory(img_idx, full_filename, "after_extraction", mem_stats.baseline_kb);
        
        // === POLAR REFITTING (Optional) ===
        if (args.use_polar && !extracted.empty()) {
            t_stage = std::chrono::high_resolution_clock::now();
            
            const cv::Mat& rim_prob = full_planes[2];
            const cv::Mat& core_prob = full_planes[1];
            
            std::vector<Crater> polar_refitted;
            polar_refitted.reserve(extracted.size());
            
            for (auto& crater : extracted) {
                bool success = refit_crater_polar(
                    crater, rim_prob, core_prob, h_orig, w_orig, 360, 0.5
                );
                // Always keep the crater (original or refitted)
                polar_refitted.push_back(crater);
            }
            
            extracted = std::move(polar_refitted);
            
            double t_polar = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_stage).count();
            time_polar_fitting.add(t_polar);
        }

        // === RANKER SCORING (Optional - uses LightGBM model) ===
        if (args.use_ranker && !extracted.empty()) {
            t_stage = std::chrono::high_resolution_clock::now();
            
            // Get rim probability channel for feature extraction (from full_planes vector)
            const cv::Mat& rim_prob = full_planes[2];  // Channel 2 is rim
            
            std::vector<Crater> ranked_craters;
            ranked_craters.reserve(extracted.size());
            
            for (auto& crater : extracted) {
                // Get rim points near this ellipse using ultra-fast method
                auto rim_pts = RankingFeatures::get_rim_points_for_candidate_fast(
                    rim_prob, crater.x, crater.y, crater.a, crater.b, crater.angle,
                    48, 4, 0.2f  // n_samples=48, band_width=4, prob_thresh=0.2 (matches Python eval3rect.py)
                );
                
                if (rim_pts.size() < 5) {
                    // Not enough rim points - use confidence as fallback
                    crater.ranker_score = crater.confidence;
                } else {
                    // Compute illumination level for SNR features
                    double illumination_level = cv::mean(img_orig)[0];
                    
                    // Extract features using multi-resolution features
                    auto feats = RankingFeatures::extract_crater_features_ultra_fast(
                        rim_pts,
                        crater.x, crater.y, crater.a, crater.b, crater.angle,
                        rim_prob,
                        h_orig, w_orig,
                        img_orig,           // Pass original image for SNR
                        illumination_level  // Global brightness for SNR
                    );
                    
                    // Convert to ordered vector for model input
                    auto feat_vec = RankingFeatures::features_to_vector(feats);
                    
                    // Get ranker probability using pure-C LightGBM model
                    double prob = score_probability(feat_vec.data());
                    crater.ranker_score = static_cast<float>(prob);
                }
                
                // Filter by ranker threshold
                if (crater.ranker_score >= args.ranker_thresh) {
                    ranked_craters.push_back(crater);
                }
            }
            
            extracted = std::move(ranked_craters);
            
            double t_ranking = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_stage).count();
            time_ranker_scoring.add(t_ranking);
        }

        // === CLASSIFICATION SCORING (uses LightGBM ensemble) ===
        if (args.use_classifier && !extracted.empty()) {
            t_stage = std::chrono::high_resolution_clock::now();
            
            const cv::Mat& rim_prob_cls = full_planes[2];  // Channel 2 is rim
            
            // Get relative path for this image (for storing classification results)
            std::string img_key_for_cls;
            if (raw_img_path.find(args.raw_data_dir) == 0) {
                img_key_for_cls = raw_img_path.substr(args.raw_data_dir.length());
                if (!img_key_for_cls.empty() && img_key_for_cls[0] == '/') {
                    img_key_for_cls = img_key_for_cls.substr(1);
                }
                size_t ext_pos = img_key_for_cls.find_last_of('.');
                if (ext_pos != std::string::npos) {
                    img_key_for_cls = img_key_for_cls.substr(0, ext_pos);
                }
            } else {
                img_key_for_cls = full_filename.substr(0, full_filename.find_last_of('.'));
            }
            
            for (size_t crater_idx = 0; crater_idx < extracted.size(); ++crater_idx) {
                auto& crater = extracted[crater_idx];
                
                // Get rim points for feature extraction
                auto rim_pts = RankingFeatures::get_rim_points_for_candidate_fast(
                    rim_prob_cls, crater.x, crater.y, crater.a, crater.b, crater.angle,
                    48, 4, 0.2f
                );
                
                int predicted_class = 2;  // Default to class B if not enough points
                
                if (rim_pts.size() >= 5) {
                    // Compute illumination level for SNR features
                    double illumination_level = cv::mean(img_orig)[0];
                    
                    // Extract ONLY the 31 classifier features (lean, no morphology/stability/meta)
                    auto feat_vec = RankingFeatures::extract_classifier_features(
                        rim_pts,
                        crater.x, crater.y, crater.a, crater.b, crater.angle,
                        rim_prob_cls,
                        img_orig,
                        h_orig, w_orig,
                        illumination_level
                    );
                    
                    // Get classification score using ensemble model
                    double ordinal_score = cls_ensemble_predict(feat_vec.data());
                    predicted_class = cls_ordinal_to_class(ordinal_score);
                }
                
                // Store classification result
                crater_classifications[{img_key_for_cls, static_cast<int>(crater_idx)}] = predicted_class;
            }
            
            double t_cls = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_stage).count();
            time_classification.add(t_cls);
        }

        // Sort by ranker_score (if ranker used) or confidence (if not), then limit
        if (args.use_ranker) {
            // When ranker is used, sort by ranker_score (matches Python eval3rect.py)
            std::sort(extracted.begin(), extracted.end(), [](const Crater& a, const Crater& b){
                return a.ranker_score > b.ranker_score;
            });
        } else {
            // No ranker: sort by original confidence
            std::sort(extracted.begin(), extracted.end(), [](const Crater& a, const Crater& b){
                return a.confidence > b.confidence;
            });
        }
        if (args.limit_craters > 0 && (int)extracted.size() > args.limit_craters) {
            extracted.resize(args.limit_craters);
        }

        images_processed++;
        total_craters += extracted.size();
        
        // Track memory after inference
        update_memory_stats(mem_stats);
        
        // Collect for CSV output - extract relative path from raw_data_dir
        // e.g., /path/to/train/altitude01/longitude14/orientation06_light01.png -> altitude01/longitude14/orientation06_light01
        std::string gt_key;
        if (raw_img_path.find(args.raw_data_dir) == 0) {
            // Get path relative to raw_data_dir
            gt_key = raw_img_path.substr(args.raw_data_dir.length());
            if (!gt_key.empty() && gt_key[0] == '/') {
                gt_key = gt_key.substr(1);  // Remove leading slash
            }
            // Remove .png extension
            size_t ext_pos = gt_key.find_last_of('.');
            if (ext_pos != std::string::npos) {
                gt_key = gt_key.substr(0, ext_pos);
            }
        } else {
            gt_key = full_filename.substr(0, full_filename.find_last_of('.'));
        }
        all_predictions.push_back({gt_key, extracted});
        
        // ========================================================
        // EXPLICIT MEMORY CLEANUP - Release all temporary cv::Mat objects
        // This is CRITICAL to prevent memory accumulation across images.
        // cv::Mat uses reference counting, but local mats can hold 
        // references to intermediate FFT/DFT buffers.
        // ========================================================
        img_orig.release();
        img_resized.release();
        dem.release();
        grad.release();
        for (auto& plane : full_planes) {
            plane.release();
        }
        full_planes.clear();
        extracted.clear();
        extracted.shrink_to_fit();
        
        // DEBUG: Final memory check at end of image processing
        print_debug_memory(img_idx, full_filename, "end_of_loop", mem_stats.baseline_kb);
        
        // Update progress bar
        print_progress_bar(images_processed, total_images);
    }
    
    std::cout << std::endl;  // New line after progress bar

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_duration_s = std::chrono::duration<double>(total_end_time - total_start_time).count();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "INFERENCE SUMMARY (C++)" << std::endl;
    std::cout << "Images Processed: " << images_processed << std::endl;
    std::cout << "Total Craters Found: " << total_craters << std::endl;
    std::cout << "Average Craters/Image: " << std::fixed << std::setprecision(1) 
              << (float)total_craters / std::max(1, images_processed) << std::endl;
    std::cout << "TOTAL EXECUTION TIME: " << std::fixed << std::setprecision(3) << total_duration_s << " s" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // ============================================
    // DETAILED TIMING REPORT (matching Python)
    // ============================================
    double total_tracked = (time_image_loading.total + time_resizing.total + 
                           time_dem_gradient.total + time_model_inference.total + 
                           time_ellipse_extraction.total + time_ranker_scoring.total +
                           time_polar_fitting.total + time_classification.total) / 1000.0;  // to seconds
    double untracked = total_duration_s - total_tracked;
    
    auto printRow = [&](const char* name, const TimingStats& s, double total_s) {
        double pct = (s.total / 1000.0) / total_s * 100.0;
        std::cout << std::left << std::setw(30) << name << " | "
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << (s.total / 1000.0) << " | "
                  << std::setw(10) << std::setprecision(2) << s.mean() << " | "
                  << std::setw(10) << std::setprecision(2) << s.max << " | "
                  << std::setw(6) << std::setprecision(1) << pct << "%" << std::endl;
    };
    
    std::cout << "\n                    DETAILED TIMING REPORT" << std::endl;
    std::cout << "======================================================================\n" << std::endl;
    
    std::cout << "[ONE-TIME INITIALIZATION COSTS]" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "  config_loading                : " << std::setw(10) << std::fixed << std::setprecision(2) << time_config_loading << " ms" << std::endl;
    std::cout << "  onnx_session_init             : " << std::setw(10) << std::fixed << std::setprecision(2) << time_onnx_init << " ms" << std::endl;
    std::cout << "  TOTAL INIT                    : " << std::setw(10) << std::fixed << std::setprecision(2) << time_total_init << " ms" << std::endl;
    
    std::cout << "\n[PER-IMAGE PROCESSING COSTS]" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(30) << "  Stage" << " | "
              << std::right << std::setw(10) << "Total(s)" << " | "
              << std::setw(10) << "Mean(ms)" << " | "
              << std::setw(10) << "Max(ms)" << " | "
              << std::setw(6) << "    %" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    
    printRow("  model_inference", time_model_inference, total_duration_s);
    printRow("  ellipse_extraction", time_ellipse_extraction, total_duration_s);
    if (args.use_polar) {
        printRow("  polar_fitting", time_polar_fitting, total_duration_s);
    }
    printRow("  ranker_scoring", time_ranker_scoring, total_duration_s);
    if (args.use_classifier) {
        printRow("  classification", time_classification, total_duration_s);
    }
    printRow("  dem_gradient_creation", time_dem_gradient, total_duration_s);
    printRow("  image_loading", time_image_loading, total_duration_s);
    printRow("  resizing", time_resizing, total_duration_s);
    
    std::cout << std::left << std::setw(30) << "  (untracked overhead)" << " | "
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << untracked << " | "
              << std::setw(10) << std::setprecision(2) << (untracked / images_processed * 1000.0) << " | "
              << std::setw(10) << "" << " | "
              << std::setw(6) << std::setprecision(1) << (untracked / total_duration_s * 100.0) << "%" << std::endl;
    
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(30) << "  TOTAL (end-to-end)" << " | "
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) << total_duration_s << " | "
              << std::setw(10) << std::setprecision(2) << (total_duration_s / images_processed * 1000.0) << " | "
              << std::setw(10) << "" << " | "
              << "100.0%" << std::endl;
    
    std::cout << "\n  Images processed: " << images_processed << std::endl;
    std::cout << "  Average time per image: " << std::fixed << std::setprecision(2) 
              << (total_duration_s / images_processed * 1000.0) << " ms (" 
              << std::setprecision(2) << (total_duration_s / images_processed) << " s)" << std::endl;
    
    // Print memory report
    print_memory_report(mem_stats);
    
    // === WRITE CSV FILE ===
    // solution output (predictions)
    {
        std::ofstream sol_file(args.solution_out);
        sol_file << "ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification" << std::endl;
        
        for (const auto& [img_path, craters] : all_predictions) {
            if (craters.empty()) {
                // No craters detected - add a -1 row
                sol_file << "-1,-1,-1,-1,-1," << img_path << ",-1" << std::endl;
            } else {
                for (size_t i = 0; i < craters.size(); ++i) {
                    const auto& c = craters[i];
                    // Look up classification for this crater
                    int cls_label = 2;  // Default to B
                    auto cls_it = crater_classifications.find({img_path, static_cast<int>(i)});
                    if (cls_it != crater_classifications.end()) {
                        cls_label = cls_it->second;
                    }
                    sol_file << std::fixed << std::setprecision(2)
                             << c.x << "," << c.y << "," 
                             << c.a << "," << c.b << "," 
                             << c.angle << "," << img_path << "," << cls_label << std::endl;
                }
            }
        }
        sol_file.close();
        std::cout << "\nSaved " << args.solution_out << " with predictions for " << all_predictions.size() << " images." << std::endl;
    }
    
    // Close memory log file
    close_memory_log();
    std::cout << "\nMemory debug log saved to: memory_debug.log" << std::endl;
    
    return 0;
}
