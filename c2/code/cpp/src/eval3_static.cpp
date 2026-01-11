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
#include "../include/watershed_static.h"  // Static memory version
#include "../include/label.h"             // For skimage-compatible regionprops
#include "../include/lightgbm_ranker.h"   // Auto-generated LightGBM model (pure C)
#include "../include/ranking_features_multires.hpp"  // Feature extraction for ranker

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

// --- SCORING CONSTANTS ---
const float FIXED_SCALE = 1.20f;
const float FIXED_THRESH = 0.6f;
const float MIN_CONFIDENCE = 0.7f;      // Minimum confidence for crater extraction
const float MIN_CONFIDENCE_MIN = 0.6f;   // Floor for iterative confidence lowering

const float XI_2_THRESH = 13.277f;
const float NN_PIX_ERR_RATIO = 0.07f; 

// List of specific prefixes to process
const std::vector<std::string> TARGET_PREFIXES = {
    "altitude01_longitude10", "altitude01_longitude14",
    "altitude04_longitude13", "altitude06_longitude13",
    "altitude09_longitude08"
};

// --- DATA STRUCTURES ---
// Crater struct is defined in watershed_static.h

using GTMap = std::map<std::string, std::vector<Crater>>;


struct Args {
    std::string raw_data_dir; 
    std::string gt_csv_path;
    std::string model_path;
    int input_res = 1296 ;   // Input resolution (default 1024)
    int limit_craters = 12;  // 0 = no limit
    int limit_images = 0;   // 0 = no limit (for memory testing)
    float ranker_thresh = 0.1f;  // Ranker score threshold (default 0.1 as in Python)
    bool use_ranker = false;  // Enable/disable ranker (match eval3.cpp default)
    bool use_instance_norm = false;  // Apply per-image instance normalization
    
    // Tiling configuration (tile_w > 0 enables tiling)
    int tile_w = 0;         // Tile width (0 = no tiling, use full padded image)
    int tile_h = 0;         // Tile height
    int overlap_w = 0;      // Horizontal overlap between tiles
    int overlap_h = 0;      // Vertical overlap between tiles
};

// =================================================================
// HELPER FUNCTIONS 
// =================================================================

// --- MEMORY TRACKING (Linux /proc/self/status) ---
struct MemoryStats {
    long baseline_kb = 0;
    long peak_kb = 0;
    std::vector<long> samples_kb;
};

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
    stats.samples_kb.push_back(current);
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
    cv::Mat gx_planes[] = {gx.clone(), cv::Mat::zeros(H, W, CV_32F)};
    cv::Mat gy_planes[] = {gy.clone(), cv::Mat::zeros(H, W, CV_32F)};
    cv::merge(gx_planes, 2, gx_complex);
    cv::merge(gy_planes, 2, gy_complex);
    cv::dft(gx_complex, gx_complex);
    cv::dft(gy_complex, gy_complex);
    
    cv::Mat gx_fft[2], gy_fft[2];
    cv::split(gx_complex, gx_fft);
    cv::split(gy_complex, gy_fft);
    
    // Step 7: z_fft = -1j * (kx * gx_fft + ky * gy_fft) / denom
    cv::Mat A_real = kx.mul(gx_fft[0]) + ky.mul(gy_fft[0]);
    cv::Mat A_imag = kx.mul(gx_fft[1]) + ky.mul(gy_fft[1]);
    
    cv::Mat z_fft_real = A_imag / denom;
    cv::Mat z_fft_imag = -A_real / denom;
    
    // Step 8: Inverse FFT to get DEM
    cv::Mat z_complex;
    cv::Mat z_planes[] = {z_fft_real, z_fft_imag};
    cv::merge(z_planes, 2, z_complex);
    cv::idft(z_complex, z_complex, cv::DFT_SCALE);
    
    cv::Mat z_result[2];
    cv::split(z_complex, z_result);
    cv::Mat dem = z_result[0];  // Take real part
    
    // Step 9: Normalize DEM to [0, 1]
    double dem_min, dem_max;
    cv::minMaxLoc(dem, &dem_min, &dem_max);
    dem = dem - dem_min;
    cv::minMaxLoc(dem, nullptr, &dem_max);
    dem /= (dem_max + 1e-6);
    
    return {dem, grad};
}

// Legacy wrappers for backward compatibility
cv::Mat compute_pseudo_dem_cpp(const cv::Mat& img_gray) {
    auto [dem, grad] = compute_dem_and_gradient_cpp(img_gray);
    return dem;
}

cv::Mat compute_gradient_map_cpp(const cv::Mat& img_gray) {
    auto [dem, grad] = compute_dem_and_gradient_cpp(img_gray);
    return grad;
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
    
    fprintf(stderr, "[TILED] Image %dx%d | Tile %dx%d | Overlap %dx%d | Grid %dx%d = %d tiles | Static accumulators\n",
            W, H, tile_w, tile_h, overlap_w, overlap_h, n_cols, n_rows, n_rows * n_cols);
    
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
    
    return output;
}

cv::Mat calcYmat(float a, float b, float phi_rad) {
    cv::Mat R = (cv::Mat_<float>(2, 2) << std::cos(phi_rad), -std::sin(phi_rad), std::sin(phi_rad), std::cos(phi_rad));
    cv::Mat D = (cv::Mat_<float>(2, 2) << 1.0f/(a*a), 0.0f, 0.0f, 1.0f/(b*b));
    cv::Mat R_T = (cv::Mat_<float>(2, 2) << std::cos(phi_rad), std::sin(phi_rad), -std::sin(phi_rad), std::cos(phi_rad));
    return R * D * R_T;
}

std::pair<float, float> dGA_calc(const Crater& crater_A, const Crater& crater_B) {
    float phi_A = crater_A.angle * (float)M_PI / 180.0f;
    float phi_B = crater_B.angle * (float)M_PI / 180.0f;
    cv::Mat Yi = calcYmat(crater_A.a, crater_A.b, phi_A);
    cv::Mat Yj = calcYmat(crater_B.a, crater_B.b, phi_B);
    cv::Mat yi = (cv::Mat_<float>(2, 1) << crater_A.x, crater_A.y);
    cv::Mat yj = (cv::Mat_<float>(2, 1) << crater_B.x, crater_B.y);
    cv::Mat Sum = Yi + Yj;
    float det_Sum = cv::determinant(Sum);
    if (std::abs(det_Sum) < 1e-9f) return { (float)M_PI, 9999.0f };
    cv::Mat InvSum = Sum.inv();
    cv::Mat Diff = yi - yj; 
    cv::Mat T5 = Diff.t() * Yi * InvSum * Yj * Diff;
    float exponent = T5.at<float>(0, 0) * -0.5f;
    float det_Yi = cv::determinant(Yi);
    float det_Yj = cv::determinant(Yj);
    float multiplicand = 4.0f * std::sqrt(det_Yi * det_Yj) / det_Sum;
    float val = multiplicand * std::exp(exponent);
    float dGA = std::acos(std::min(1.0f, std::max(-1.0f, val)));
    float min_axis = std::min(crater_A.a, crater_A.b);
    float ref_sig = 0.85f / std::sqrt(crater_A.a * crater_A.b) * (NN_PIX_ERR_RATIO * min_axis);
    if (ref_sig < 1e-9f) ref_sig = 1e-6f;
    float xi_2 = (dGA * dGA) / (ref_sig * ref_sig);
    return { dGA, xi_2 };
}

float compute_full_scores(const std::vector<Crater>& gt_craters, std::vector<Crater>& pred_craters) {
    if (gt_craters.empty() || pred_craters.empty()) return 0.0f;
    std::vector<float> matches_dga;
    std::vector<bool> pred_matched(pred_craters.size(), false);
    
    for (const auto& t : gt_craters) {
        float best_xi = 1e9f;
        int best_p_idx = -1;
        float best_dGA = (float)M_PI;
        
        for (size_t p_idx = 0; p_idx < pred_craters.size(); ++p_idx) {
            if (pred_matched[p_idx]) continue;
            const auto& p = pred_craters[p_idx];
            
            // Scorer.py matching logic (same as Python)
            float rA = std::min(t.a, t.b);  // Semi-minor of GT
            float rB = std::min(p.a, p.b);  // Semi-minor of pred
            
            // Size check: semi-minor axes must be within 1.5x of each other
            if (rA > 1.5f * rB || rB > 1.5f * rA) continue;
            
            // Distance check (stricter): separate x/y checks with min radius
            float r = std::min(rA, rB);
            if (std::abs(t.x - p.x) > r) continue;
            if (std::abs(t.y - p.y) > r) continue;
            
            auto [dGA, xi_2] = dGA_calc(t, p);
            if (xi_2 < best_xi) { 
                best_xi = xi_2; 
                best_p_idx = (int)p_idx;
                best_dGA = dGA;
            }
        }
        
        if (best_xi < XI_2_THRESH && best_p_idx != -1) {
            pred_matched[best_p_idx] = true;
            pred_craters[best_p_idx].matched = true;
            matches_dga.push_back(1.0f - (best_dGA / (float)M_PI));
        }
    }
    
    if (matches_dga.empty()) return 0.0f;
    float sum_quality = 0.0f;
    for (float q : matches_dga) sum_quality += q;
    float avg_quality = sum_quality / pred_craters.size(); 
    float recall_factor = (float)matches_dga.size() / std::min(10, (int)gt_craters.size());
    if (recall_factor > 1.0f) recall_factor = 1.0f;
    return avg_quality * recall_factor;
}

GTMap load_gt_data(const std::string& gt_csv_path) {
    GTMap gt_map;
    std::ifstream file(gt_csv_path);
    if (!file.is_open()) return gt_map;
    std::string line;
    std::getline(file, line); 
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) row.push_back(cell);
        if (row.size() < 6) continue; 
        std::string input_image = row[5]; 
        // Temporarily process all images (disabled prefix filter for testing)
        bool is_target = false;  // Was: filtered by TARGET_PREFIXES
        
        for(const auto& prefix : TARGET_PREFIXES) { 
            size_t underscore_pos = prefix.find('_');
            if (underscore_pos == std::string::npos) continue; 
            std::string alt_part = prefix.substr(0, underscore_pos);
            std::string lon_part = prefix.substr(underscore_pos + 1);
            if (input_image.find(alt_part) != std::string::npos && input_image.find(lon_part) != std::string::npos) {
                is_target = true; break;
            }
        }
        
        if (!is_target) continue;
        try {
            Crater crater;
            crater.x = std::stof(row[0]); crater.y = std::stof(row[1]); crater.a = std::stof(row[2]); crater.b = std::stof(row[3]); crater.angle = std::stof(row[4]); 
            if (crater.b >= 5.0f) gt_map[input_image].push_back(crater);
        } catch (...) { continue; }
    }
    return gt_map;
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--raw-dir" && i + 1 < argc) args.raw_data_dir = argv[++i];
        else if (arg == "--gt" && i + 1 < argc) args.gt_csv_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.model_path = argv[++i];
        else if (arg == "--limit-craters" && i + 1 < argc) args.limit_craters = std::stoi(argv[++i]);
        else if (arg == "--limit-images" && i + 1 < argc) args.limit_images = std::stoi(argv[++i]);
        else if (arg == "--ranker-thresh" && i + 1 < argc) args.ranker_thresh = std::stof(argv[++i]);
        else if (arg == "--no-ranker") args.use_ranker = false;
        else if (arg == "--input-res" && i + 1 < argc) args.input_res = std::stoi(argv[++i]);
        else if (arg == "--use-instance-norm") args.use_instance_norm = true;
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
    double time_config_loading = 0, time_onnx_init = 0;
    
    auto t_init_start = std::chrono::high_resolution_clock::now();
    
    // -------------------------------------------------------------
    // SINGLE CORE MODE (deployment constraint)
    // -------------------------------------------------------------
    const int NUM_THREADS = 1;
    cv::setNumThreads(NUM_THREADS);
    
    // Initialize static memory buffers for watershed
    init_watershed_static_buffers(2048, 2592);  // Max expected image size
    // -------------------------------------------------------------

    auto t0 = std::chrono::high_resolution_clock::now();
    Args args = parse_args(argc, argv);
    if (args.raw_data_dir.empty()) return 1;
    time_config_loading = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // 1. Setup Environment and Load Model
    t0 = std::chrono::high_resolution_clock::now();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CraterFullPipeline");
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

    GTMap gt_map = load_gt_data(args.gt_csv_path);
    
    double time_total_init = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_init_start).count();
    
    std::cout << "\n--- STARTING EVALUATION (Single Threaded) ---\n" << std::endl;

    std::vector<float> all_nasa_scores;
    auto total_start_time = std::chrono::high_resolution_clock::now();
    int images_processed = 0;
    
    // Memory tracking
    MemoryStats mem_stats;
    mem_stats.baseline_kb = get_current_rss_kb();
    mem_stats.peak_kb = mem_stats.baseline_kb;
    
    // Data collection for CSV output
    std::vector<std::tuple<std::string, std::vector<Crater>>> all_predictions;  // (image_path, craters)
    std::set<std::string> tested_image_paths;

    // Count and display how many images will be processed
    int total_gt_images = gt_map.size();
    std::cout << "\nFound " << total_gt_images << " images in GT to process\n" << std::endl;

    for (const auto& pair : gt_map) {
        // Early exit for memory testing
        if (args.limit_images > 0 && images_processed >= args.limit_images) {
            std::cout << "Reached limit of " << args.limit_images << " images. Stopping." << std::endl;
            break;
        }
        
        const std::string& gt_key = pair.first; 
        const std::vector<Crater>& gt_craters = pair.second;
        std::string raw_img_path = args.raw_data_dir + "/" + gt_key + ".png";
        
        if (!fs::exists(raw_img_path)) continue;

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
            
            // Apply instance normalization if enabled (matches PyTorch: (x - mean) / clamp(std, min=0.1))
            if (args.use_instance_norm) {
                cv::Scalar mean_scalar, std_scalar;
                cv::meanStdDev(img_float, mean_scalar, std_scalar);
                float mean = (float)mean_scalar[0];
                float std_val = std::max((float)std_scalar[0], 0.1f);
                img_float = (img_float - mean) / std_val;
            }
            
            cv::Mat img_norm = (img_float - 0.5f) / 0.5f;
            
            // Build input planes based on CHANNELS configuration
            std::vector<cv::Mat> input_planes;
            if constexpr (CHANNELS == 1) {
                input_planes.push_back(img_norm);
            } else if constexpr (CHANNELS == 2) {
                // Normalize gradient (already in [0,1])
                cv::Mat grad_float = grad.clone();
                if (args.use_instance_norm) {
                    cv::Scalar mean_scalar, std_scalar;
                    cv::meanStdDev(grad_float, mean_scalar, std_scalar);
                    float mean = (float)mean_scalar[0];
                    float std_val = std::max((float)std_scalar[0], 0.1f);
                    grad_float = (grad_float - mean) / std_val;
                }
                cv::Mat grad_norm = (grad_float - 0.5f) / 0.5f;
                input_planes.push_back(img_norm);
                input_planes.push_back(grad_norm);
            } else {
                // 3 channels: img + DEM + gradient
                cv::Mat dem_float = dem.clone();
                cv::Mat grad_float = grad.clone();
                if (args.use_instance_norm) {
                    cv::Scalar mean_scalar, std_scalar;
                    cv::meanStdDev(dem_float, mean_scalar, std_scalar);
                    float mean = (float)mean_scalar[0];
                    float std_val = std::max((float)std_scalar[0], 0.1f);
                    dem_float = (dem_float - mean) / std_val;
                    
                    cv::meanStdDev(grad_float, mean_scalar, std_scalar);
                    mean = (float)mean_scalar[0];
                    std_val = std::max((float)std_scalar[0], 0.1f);
                    grad_float = (grad_float - mean) / std_val;
                }
                cv::Mat dem_norm = (dem_float - 0.5f) / 0.5f;
                cv::Mat grad_norm = (grad_float - 0.5f) / 0.5f;
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
            
            // Resize each channel to original resolution
            for (int c = 0; c < OUT_CLASSES; ++c) {
                int interp = (c == 2) ? cv::INTER_NEAREST : cv::INTER_LINEAR;
                cv::resize(tiled_output[c], full_planes[c], cv::Size(w_orig, h_orig), 0, 0, interp);
            }
        } else {
            // ===============================================
            // FULL IMAGE INFERENCE - DEPRECATED
            // Tiling is always required for memory-efficient inference.
            // ===============================================
            fprintf(stderr, "[ERROR] Full-image mode is disabled. Please use --tile-size to enable tiling.\\n");
            fprintf(stderr, "        Example: --tile-size 672x544 or --tile-size 544x416\\n");
            continue;  // Skip this image
        }
        
        cv::Mat preds_full_split;
        cv::merge(full_planes, preds_full_split);
        double t_inference = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_stage).count();
        time_model_inference.add(t_inference);

        // === ELLIPSE EXTRACTION ===
        t_stage = std::chrono::high_resolution_clock::now();
        // Use extraction function from watershed_static.cpp
        std::vector<Crater> extracted = extract_craters_cv2_adaptive_selection_rescue(
            full_planes,      // vector<cv::Mat> not merged Mat
            FIXED_THRESH,     // 0.75 threshold (matching eval3.cpp)
            0.25f,            // rim_thresh
            0.15f,            // ecc_floor
            1.0f, 1.4f, 30,   // scale_min, scale_max, scale_steps
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

        // === RANKER SCORING (Optional - uses LightGBM model converted to C) ===
        if (args.use_ranker && !extracted.empty()) {
            t_stage = std::chrono::high_resolution_clock::now();
            
            // Get rim probability channel for feature extraction (from full_planes vector)
            const cv::Mat& rim_prob = full_planes[2];  // Channel 2 is rim
            
            std::vector<Crater> ranked_craters;
            ranked_craters.reserve(extracted.size());
            
            for (auto& crater : extracted) {
                // Get rim points near this ellipse using fast method
                auto rim_pts = RankingFeatures::get_rim_points_for_candidate_fast(
                    rim_prob, crater.x, crater.y, crater.a, crater.b, crater.angle,
                    48,   // n_samples (matches Python eval3rect.py)
                    4,    // band_width (matches Python eval3rect.py)
                    0.2f  // prob_thresh
                );
                
                if (rim_pts.size() < 5) {
                    // Not enough rim points - use confidence as fallback
                    crater.ranker_score = crater.confidence;
                } else {
                    // Extract features using ultra-fast method (no SNR features)
                    auto feat_map = RankingFeatures::extract_crater_features_ultra_fast(
                        rim_pts,
                        crater.x, crater.y, crater.a, crater.b, crater.angle,
                        rim_prob,
                        h_orig, w_orig
                        // img_orig and illumination_level omitted for speed
                    );
                    
                    // Convert to ordered vector for model input
                    auto feat_vec = RankingFeatures::features_to_vector(feat_map);
                    
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

        // Sort by confidence (descending) and limit
        std::sort(extracted.begin(), extracted.end(), [](const Crater& a, const Crater& b){
            return a.confidence > b.confidence;
        });
        if (args.limit_craters > 0 && (int)extracted.size() > args.limit_craters) {
            extracted.resize(args.limit_craters);
        }
        
        auto scoring_start_time = std::chrono::high_resolution_clock::now();
        double pipeline_duration_s = std::chrono::duration<double>(scoring_start_time - pipeline_start_time).count();

        float nasa_score = 0.0f;
        if (!gt_craters.empty()) {
            nasa_score = compute_full_scores(gt_craters, extracted);
            all_nasa_scores.push_back(nasa_score);
        }

        images_processed++;
        
        // Track memory after inference
        update_memory_stats(mem_stats);
        
        // Collect for CSV output
        tested_image_paths.insert(gt_key);
        all_predictions.push_back({gt_key, extracted});
        
        std::cout << "Processed: " << full_filename 
                  << " | Time: " << std::fixed << std::setprecision(3) << pipeline_duration_s << " s"
                  << " | Craters Found: " << extracted.size() 
                  << " | GT Craters: " << gt_craters.size() 
                  << " | NASA Score: " << std::fixed << std::setprecision(4) << nasa_score
                  << std::endl;
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_duration_s = std::chrono::duration<double>(total_end_time - total_start_time).count();
    
    float mean_nasa_score = 0.0f;
    if (!all_nasa_scores.empty()) {
        for (float score : all_nasa_scores) mean_nasa_score += score;
        mean_nasa_score /= (float)all_nasa_scores.size();
    }

    std::cout << "\n==========================================" << std::endl;
    std::cout << "EVALUATION SUMMARY (C++)" << std::endl;
    std::cout << "Images Scored: " << all_nasa_scores.size() << " / " << images_processed << std::endl;
    std::cout << "FINAL MEAN NASA SCORE: " << std::fixed << std::setprecision(4) << mean_nasa_score << std::endl;
    std::cout << "TOTAL EXECUTION TIME: " << std::fixed << std::setprecision(3) << total_duration_s << " s" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // ============================================
    // DETAILED TIMING REPORT (matching Python)
    // ============================================
    double total_tracked = (time_image_loading.total + time_resizing.total + 
                           time_dem_gradient.total + time_model_inference.total + 
                           time_ellipse_extraction.total + time_ranker_scoring.total) / 1000.0;  // to seconds
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
    printRow("  ranker_scoring", time_ranker_scoring, total_duration_s);
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
    
    // === WRITE CSV FILES ===
    // 1. solution-val.csv (predictions)
    {
        std::ofstream sol_file("solution-val.csv");
        sol_file << "ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification" << std::endl;
        
        for (const auto& [img_path, craters] : all_predictions) {
            if (craters.empty()) {
                // No craters detected - add a -1 row
                sol_file << "-1,-1,-1,-1,-1," << img_path << ",-1" << std::endl;
            } else {
                for (const auto& c : craters) {
                    sol_file << std::fixed << std::setprecision(2)
                             << c.x << "," << c.y << "," 
                             << c.a << "," << c.b << "," 
                             << c.angle << "," << img_path << ",4" << std::endl;
                }
            }
        }
        sol_file.close();
        std::cout << "Saved solution-val.csv with predictions for " << all_predictions.size() << " images." << std::endl;
    }
    
    // 2. test-sub.csv (ground truth subset for tested images)
    {
        std::ofstream sub_file("test-sub.csv");
        sub_file << "ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage" << std::endl;
        
        int gt_rows = 0;
        for (const auto& img_path : tested_image_paths) {
            auto it = gt_map.find(img_path);
            if (it != gt_map.end()) {
                for (const auto& c : it->second) {
                    sub_file << std::fixed << std::setprecision(2)
                             << c.x << "," << c.y << "," 
                             << c.a << "," << c.b << "," 
                             << c.angle << "," << img_path << std::endl;
                    gt_rows++;
                }
            }
        }
        sub_file.close();
        std::cout << "Saved test-sub.csv with " << gt_rows << " GT rows for " << tested_image_paths.size() << " images." << std::endl;
    }
    
    return 0;
}
