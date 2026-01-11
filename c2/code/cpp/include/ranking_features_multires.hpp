#ifndef RANKING_FEATURES_MULTIRES_HPP
#define RANKING_FEATURES_MULTIRES_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <random>

namespace RankingFeatures {

// Feature names for consistent ordering - matches train_classifier.py FEATURE_COLS exactly
// Total: 45 features for classification
static const std::vector<std::string> FEATURE_NAMES = {
    // Geometry features (11)
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
    // Rim probability features (8)
    "rim_prob_mean",
    "rim_prob_std",
    "rim_prob_min",
    "rim_prob_p20",
    "rim_prob_p80",
    "rim_prob_frac_above_50",
    "rim_prob_frac_above_70",
    "rim_prob_entropy",
    // Morphology features (6)
    "morph_solidity",
    "morph_convexity",
    "morph_extent",
    "morph_perim_area_ratio",
    "morph_minor_axis",
    "morph_area_log",
    // Stability features (5)
    "stab_jitter_a",
    "stab_jitter_b",
    "stab_jitter_theta",
    "stab_center_shift",
    "stab_iou_subsample",
    // Mahanti degradation features (4)
    "mahanti_median_slope",
    "mahanti_slope_std",
    "mahanti_depth_ratio",
    "mahanti_rim_sharpness",
    // Context features (5)
    "ctx_inside_mean",
    "ctx_outside_mean",
    "ctx_diff",
    "ctx_grad_mean",
    "ctx_grad_std",
    // Meta features (2)
    "meta_log_radius",
    "meta_log_area",
    // SNR features (3)
    "snr_raw",
    "snr_x_illumination",
    "snr_illumination_level"
};

// ------------------------------------------------------------
// Helper utilities
// ------------------------------------------------------------

inline double safe_div(double a, double b) {
    return a / (b + 1e-6);
}

// Calculate ellipse distance for a set of points
inline std::vector<double> ellipse_distance(
    const std::vector<cv::Point2f>& pts,
    double cx, double cy, double a, double b, double theta_deg
) {
    double angle = theta_deg * M_PI / 180.0;
    double ct = std::cos(angle);
    double st = std::sin(angle);
    
    a = std::max(a, 1e-4);
    b = std::max(b, 1e-4);
    
    std::vector<double> distances(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        double x = pts[i].x - cx;
        double y = pts[i].y - cy;
        double xr = ct * x + st * y;
        double yr = -st * x + ct * y;
        distances[i] = std::abs((xr / a) * (xr / a) + (yr / b) * (yr / b) - 1.0);
    }
    return distances;
}

// Calculate percentile of a vector
inline double percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    double idx = (p / 100.0) * (data.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return data[lo];
    double frac = idx - lo;
    return data[lo] * (1.0 - frac) + data[hi] * frac;
}

// Calculate mean of a vector
inline double vec_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / v.size();
}

// Calculate std of a vector
inline double vec_std(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = vec_mean(v);
    double sum = 0.0;
    for (double x : v) sum += (x - m) * (x - m);
    return std::sqrt(sum / v.size());
}

// Calculate median of a vector
inline double vec_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
    return v[n/2];
}

// Compute gradient magnitude using simple finite differences (matches np.gradient)
// This is different from Sobel which uses a weighted kernel
inline cv::Mat compute_numpy_gradient(const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    
    cv::Mat grad_mag = cv::Mat::zeros(h, w, CV_32F);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float gx, gy;
            
            // X gradient (central difference, edge handling like np.gradient)
            if (x == 0) {
                gx = img.at<float>(y, 1) - img.at<float>(y, 0);
            } else if (x == w - 1) {
                gx = img.at<float>(y, w-1) - img.at<float>(y, w-2);
            } else {
                gx = (img.at<float>(y, x+1) - img.at<float>(y, x-1)) / 2.0f;
            }
            
            // Y gradient (central difference, edge handling like np.gradient)
            if (y == 0) {
                gy = img.at<float>(y+1, x) - img.at<float>(y, x);
            } else if (y == h - 1) {
                gy = img.at<float>(y, x) - img.at<float>(y-1, x);
            } else {
                gy = (img.at<float>(y+1, x) - img.at<float>(y-1, x)) / 2.0f;
            }
            
            grad_mag.at<float>(y, x) = std::sqrt(gx*gx + gy*gy);
        }
    }
    
    return grad_mag;
}

// ------------------------------------------------------------
// Geometry features (SCALE SAFE)
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> geometry_features(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta,
    int h, int w
) {
    std::unordered_map<std::string, double> feats;
    double img_area = static_cast<double>(h * w);
    
    // Ellipse distance residuals
    std::vector<double> d = ellipse_distance(rim_pts, cx, cy, a, b, theta);
    
    // Angular analysis
    std::vector<double> angles(rim_pts.size());
    for (size_t i = 0; i < rim_pts.size(); ++i) {
        double ang = std::atan2(rim_pts[i].y - cy, rim_pts[i].x - cx);
        angles[i] = std::fmod(ang + 2 * M_PI, 2 * M_PI);  // [0, 2*PI]
    }
    
    // Histogram with 36 bins
    const int num_bins = 36;
    std::vector<int> bins(num_bins, 0);
    for (double ang : angles) {
        int bin_idx = static_cast<int>(ang / (2 * M_PI) * num_bins);
        bin_idx = std::min(bin_idx, num_bins - 1);
        bins[bin_idx]++;
    }
    
    // Max gap calculation
    double max_gap = 1.0;
    if (angles.size() > 1) {
        std::sort(angles.begin(), angles.end());
        double largest_gap = 0.0;
        for (size_t i = 1; i < angles.size(); ++i) {
            largest_gap = std::max(largest_gap, angles[i] - angles[i-1]);
        }
        // Wrap-around gap
        largest_gap = std::max(largest_gap, angles[0] + 2 * M_PI - angles.back());
        max_gap = largest_gap / (2 * M_PI);
    }
    
    double ellipse_area = M_PI * a * b;
    
    // Count non-zero bins
    int nonzero_bins = 0;
    for (int c : bins) if (c > 0) nonzero_bins++;
    
    // Calculate bin std (matching np.std(bins) in Python)
    // np.std uses actual mean of the bins array, not theoretical mean
    double bins_sum = 0.0;
    for (int c : bins) bins_sum += c;
    double bins_mean = bins_sum / num_bins;  // Actual mean of bins (same as rim_pts.size() / num_bins)
    double bins_var = 0.0;
    for (int c : bins) bins_var += (c - bins_mean) * (c - bins_mean);
    double bins_std = std::sqrt(bins_var / num_bins);  // Population std (ddof=0, same as np.std)
    
    feats["geometry_eccentricity"] = std::sqrt(std::max(0.0, 1.0 - (b / (a + 1e-6)) * (b / (a + 1e-6))));
    feats["geometry_axis_ratio"] = b / (a + 1e-6);
    feats["geometry_ellipse_area"] = ellipse_area / img_area;
    feats["geometry_area_ratio"] = 1.0;  // Placeholder, needs prop.area
    
    // Residual RMS
    if (!d.empty()) {
        double sum_sq = 0.0;
        for (double x : d) sum_sq += x * x;
        feats["geometry_resid_rms"] = std::sqrt(sum_sq / d.size());
        feats["geometry_resid_p90"] = percentile(d, 90);
    } else {
        feats["geometry_resid_rms"] = 1.0;
        feats["geometry_resid_p90"] = 1.0;
    }
    
    double avg_radius = std::sqrt((a * a + b * b) / 2.0);
    feats["geometry_support_ratio"] = rim_pts.size() / (2 * M_PI * avg_radius + 1e-6);
    feats["geometry_angular_coverage"] = static_cast<double>(nonzero_bins) / num_bins;
    feats["geometry_angular_std"] = bins_std / (rim_pts.size() + 1e-6);
    feats["geometry_max_gap"] = max_gap;
    feats["geometry_condition"] = a / (b + 1e-6);
    
    return feats;
}

// ------------------------------------------------------------
// Rim probability features (SAFE)
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> rim_prob_features(
    const std::vector<cv::Point2f>& rim_pts,
    const cv::Mat& rim_prob  // CV_32F, values in [0, 1]
) {
    std::unordered_map<std::string, double> feats;
    int h = rim_prob.rows;
    int w = rim_prob.cols;
    
    std::vector<double> probs;
    probs.reserve(rim_pts.size());
    
    for (const auto& pt : rim_pts) {
        int x = std::min(std::max(static_cast<int>(std::round(pt.x)), 0), w - 1);
        int y = std::min(std::max(static_cast<int>(std::round(pt.y)), 0), h - 1);
        double p = rim_prob.at<float>(y, x);
        p = std::min(std::max(p, 1e-6), 1.0 - 1e-6);
        probs.push_back(p);
    }
    
    if (probs.empty()) {
        feats["rim_prob_mean"] = 0.0;
        feats["rim_prob_std"] = 0.0;
        feats["rim_prob_min"] = 0.0;
        feats["rim_prob_p20"] = 0.0;
        feats["rim_prob_p80"] = 0.0;
        feats["rim_prob_frac_above_50"] = 0.0;
        feats["rim_prob_frac_above_70"] = 0.0;
        feats["rim_prob_entropy"] = 0.0;
        return feats;
    }
    
    // Mean
    double sum = 0.0;
    for (double p : probs) sum += p;
    double mean = sum / probs.size();
    
    // Std
    double var = 0.0;
    for (double p : probs) var += (p - mean) * (p - mean);
    double std_val = std::sqrt(var / probs.size());
    
    // Min
    double min_val = *std::min_element(probs.begin(), probs.end());
    
    // Percentiles
    std::vector<double> probs_sorted = probs;
    double p20 = percentile(probs_sorted, 20);
    double p80 = percentile(probs_sorted, 80);
    
    // Fractions
    int above_50 = 0, above_70 = 0;
    for (double p : probs) {
        if (p > 0.5) above_50++;
        if (p > 0.7) above_70++;
    }
    
    // Entropy
    double entropy = 0.0;
    for (double p : probs) {
        entropy -= p * std::log(p);
    }
    entropy /= probs.size();
    
    feats["rim_prob_mean"] = mean;
    feats["rim_prob_std"] = std_val;
    feats["rim_prob_min"] = min_val;
    feats["rim_prob_p20"] = p20;
    feats["rim_prob_p80"] = p80;
    feats["rim_prob_frac_above_50"] = static_cast<double>(above_50) / probs.size();
    feats["rim_prob_frac_above_70"] = static_cast<double>(above_70) / probs.size();
    feats["rim_prob_entropy"] = entropy;
    
    return feats;
}

// ------------------------------------------------------------
// Polar-inspired features (NEW - high value)
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> polar_features(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta_deg,
    const cv::Mat& rim_prob
) {
    std::unordered_map<std::string, double> feats;
    const int num_bins = 36;
    
    // Convert rim points to polar coordinates relative to ellipse center
    std::vector<double> angles(rim_pts.size());
    std::vector<double> radii(rim_pts.size());
    
    for (size_t i = 0; i < rim_pts.size(); ++i) {
        double dx = rim_pts[i].x - cx;
        double dy = rim_pts[i].y - cy;
        angles[i] = std::atan2(dy, dx);  // [-PI, PI]
        radii[i] = std::sqrt(dx * dx + dy * dy);
    }
    
    // Bin into angular sectors
    std::vector<double> r_theta(num_bins, std::nan(""));
    double bin_width = 2 * M_PI / num_bins;
    
    for (size_t i = 0; i < rim_pts.size(); ++i) {
        int bin_idx = static_cast<int>((angles[i] + M_PI) / bin_width);
        bin_idx = std::min(std::max(bin_idx, 0), num_bins - 1);
        if (std::isnan(r_theta[bin_idx]) || radii[i] > r_theta[bin_idx]) {
            r_theta[bin_idx] = radii[i];
        }
    }
    
    // Count valid bins
    int valid_count = 0;
    for (int i = 0; i < num_bins; ++i) {
        if (!std::isnan(r_theta[i])) valid_count++;
    }
    
    // Feature 1: Angular support fraction
    double angular_support = static_cast<double>(valid_count) / num_bins;
    
    // Feature 2: Directional asymmetry
    int half = num_bins / 2;
    std::vector<double> asymmetries;
    
    for (int i = 0; i < half; ++i) {
        int opp_idx = i + half;
        if (!std::isnan(r_theta[i]) && !std::isnan(r_theta[opp_idx])) {
            double avg = (r_theta[i] + r_theta[opp_idx]) / 2.0;
            double diff = std::abs(r_theta[i] - r_theta[opp_idx]);
            asymmetries.push_back(diff / (avg + 1e-6));
        }
    }
    
    double mean_asymmetry = asymmetries.empty() ? 1.0 : vec_mean(asymmetries);
    double max_asymmetry = asymmetries.empty() ? 1.0 : *std::max_element(asymmetries.begin(), asymmetries.end());
    
    // Feature 3: Roughness
    double roughness_ratio = 1.0;
    if (valid_count >= 10) {
        double theta_rad = theta_deg * M_PI / 180.0;
        
        // Get valid r_theta values and their residuals from predicted ellipse radius
        std::vector<double> residuals;
        for (int i = 0; i < num_bins; ++i) {
            if (!std::isnan(r_theta[i])) {
                double bin_center = -M_PI + (i + 0.5) * bin_width;
                double ct = std::cos(bin_center - theta_rad);
                double st = std::sin(bin_center - theta_rad);
                double r_pred = (a * b) / (std::sqrt((b * ct) * (b * ct) + (a * st) * (a * st)) + 1e-9);
                residuals.push_back(r_theta[i] - r_pred);
            }
        }
        
        if (residuals.size() > 2) {
            // Gradient-based roughness
            double grad_sq_sum = 0.0;
            double res_sq_sum = 0.0;
            for (size_t i = 1; i < residuals.size(); ++i) {
                double grad = residuals[i] - residuals[i - 1];
                grad_sq_sum += grad * grad;
            }
            for (double r : residuals) res_sq_sum += r * r;
            roughness_ratio = grad_sq_sum / (res_sq_sum + 1e-10);
        } else {
            roughness_ratio = 0.0;
        }
    }
    
    feats["polar_angular_support"] = angular_support;
    feats["polar_mean_asymmetry"] = mean_asymmetry;
    feats["polar_max_asymmetry"] = max_asymmetry;
    feats["polar_roughness_ratio"] = roughness_ratio;
    
    return feats;
}

// ------------------------------------------------------------
// Mahanti degradation features
// These capture physical degradation cues from gradient analysis
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> mahanti_features(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta,
    const cv::Mat& img  // Grayscale image (CV_8U or CV_32F)
) {
    std::unordered_map<std::string, double> feats;
    
    // Default values if no valid image
    if (img.empty() || rim_pts.empty()) {
        feats["mahanti_median_slope"] = 0.0;
        feats["mahanti_slope_std"] = 0.0;
        feats["mahanti_depth_ratio"] = 0.0;
        feats["mahanti_rim_sharpness"] = 0.0;
        return feats;
    }
    
    int h = img.rows;
    int w = img.cols;
    
    // Convert to float WITHOUT normalizing to [0,1] - matches Python behavior
    // Python keeps original pixel values (0-255), not normalized
    cv::Mat img_f;
    if (img.type() == CV_8U) {
        img.convertTo(img_f, CV_32F);  // Keep original scale, no /255
    } else if (img.type() == CV_32F) {
        img_f = img;
    } else {
        feats["mahanti_median_slope"] = 0.0;
        feats["mahanti_slope_std"] = 0.0;
        feats["mahanti_depth_ratio"] = 0.0;
        feats["mahanti_rim_sharpness"] = 0.0;
        return feats;
    }
    
    // Compute gradient using np.gradient equivalent (NOT Sobel)
    cv::Mat grad_mag = compute_numpy_gradient(img_f);
    
    // Global mean for normalization
    double global_mean = cv::mean(grad_mag)[0] + 1e-6;
    
    // Sample gradient values at rim points
    std::vector<double> rim_grads;
    rim_grads.reserve(rim_pts.size());
    
    for (const auto& pt : rim_pts) {
        int x = std::min(std::max(static_cast<int>(std::round(pt.x)), 0), w - 1);
        int y = std::min(std::max(static_cast<int>(std::round(pt.y)), 0), h - 1);
        rim_grads.push_back(grad_mag.at<float>(y, x));
    }
    
    if (rim_grads.empty()) {
        feats["mahanti_median_slope"] = 0.0;
        feats["mahanti_slope_std"] = 0.0;
        feats["mahanti_depth_ratio"] = 0.0;
        feats["mahanti_rim_sharpness"] = 0.0;
        return feats;
    }
    
    // Compute statistics
    double median_slope = vec_median(rim_grads) / global_mean;
    double slope_std = vec_std(rim_grads) / global_mean;
    double min_grad = *std::min_element(rim_grads.begin(), rim_grads.end());
    double max_grad = *std::max_element(rim_grads.begin(), rim_grads.end());
    double depth_ratio = (max_grad - min_grad) / global_mean;
    double rim_sharpness = vec_mean(rim_grads) / global_mean;
    
    feats["mahanti_median_slope"] = median_slope;
    feats["mahanti_slope_std"] = slope_std;
    feats["mahanti_depth_ratio"] = depth_ratio;
    feats["mahanti_rim_sharpness"] = rim_sharpness;
    
    return feats;
}

// ------------------------------------------------------------
// Morphology features (region properties based)
// These require a binary mask of the crater region
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> morphology_features(
    const cv::Mat& mask,  // Binary mask (CV_8U)
    double a, double b,   // Ellipse axes
    int img_h, int img_w
) {
    std::unordered_map<std::string, double> feats;
    double img_area = static_cast<double>(img_h * img_w);
    double ellipse_area = M_PI * a * b;
    
    if (mask.empty()) {
        feats["morph_solidity"] = 1.0;
        feats["morph_convexity"] = 1.0;
        feats["morph_extent"] = 1.0;
        feats["morph_perim_area_ratio"] = 1.0;
        feats["morph_minor_axis"] = 1.0;
        feats["morph_area_log"] = std::log(ellipse_area / img_area + 1e-6);
        return feats;
    }
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        feats["morph_solidity"] = 1.0;
        feats["morph_convexity"] = 1.0;
        feats["morph_extent"] = 1.0;
        feats["morph_perim_area_ratio"] = 1.0;
        feats["morph_minor_axis"] = 1.0;
        feats["morph_area_log"] = std::log(ellipse_area / img_area + 1e-6);
        return feats;
    }
    
    // Find largest contour
    size_t max_idx = 0;
    double max_area = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }
    
    const auto& contour = contours[max_idx];
    double prop_area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    // Convex hull
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double convex_area = cv::contourArea(hull);
    
    // Bounding rectangle
    cv::Rect bbox = cv::boundingRect(contour);
    double bbox_area = static_cast<double>(bbox.width * bbox.height);
    
    // Fit ellipse for minor axis (if enough points)
    double minor_axis = std::min(a, b);
    if (contour.size() >= 5) {
        cv::RotatedRect fitted = cv::fitEllipse(contour);
        minor_axis = std::min(fitted.size.width, fitted.size.height) / 2.0;
    }
    
    feats["morph_solidity"] = safe_div(prop_area, convex_area);
    feats["morph_convexity"] = safe_div(prop_area, convex_area);
    feats["morph_extent"] = safe_div(prop_area, bbox_area);
    feats["morph_perim_area_ratio"] = safe_div(perimeter, std::sqrt(prop_area));
    feats["morph_minor_axis"] = safe_div(minor_axis, std::sqrt(prop_area));
    feats["morph_area_log"] = std::log(prop_area / img_area + 1e-6);
    
    return feats;
}

// Simplified version without mask - uses ellipse-based estimates
inline std::unordered_map<std::string, double> morphology_features_fast(
    double a, double b,
    int img_h, int img_w
) {
    std::unordered_map<std::string, double> feats;
    double img_area = static_cast<double>(img_h * img_w);
    double ellipse_area = M_PI * a * b;
    double perimeter = M_PI * (3 * (a + b) - std::sqrt((3 * a + b) * (a + 3 * b)));  // Ramanujan approx
    
    feats["morph_solidity"] = 1.0;       // Assume solid (no holes)
    feats["morph_convexity"] = 1.0;      // Ellipse is convex
    feats["morph_extent"] = M_PI / 4.0;  // Ellipse in bounding box
    feats["morph_perim_area_ratio"] = safe_div(perimeter, std::sqrt(ellipse_area));
    feats["morph_minor_axis"] = safe_div(std::min(a, b), std::sqrt(ellipse_area));
    feats["morph_area_log"] = std::log(ellipse_area / img_area + 1e-6);
    
    return feats;
}

// ------------------------------------------------------------
// Stability features (ellipse fit jitter under subsampling)
// These measure how stable the ellipse fit is
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> stability_features(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta_deg,
    int n_trials = 5
) {
    std::unordered_map<std::string, double> feats;
    
    if (rim_pts.size() < 10) {
        feats["stab_jitter_a"] = 1.0;
        feats["stab_jitter_b"] = 1.0;
        feats["stab_jitter_theta"] = 1.0;
        feats["stab_center_shift"] = 1.0;
        feats["stab_iou_subsample"] = 0.0;
        return feats;
    }
    
    std::vector<double> shifts_a, shifts_b, shifts_th, shifts_c;
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    
    double theta_rad = theta_deg * M_PI / 180.0;
    double theta_mod = std::fmod(theta_rad, M_PI);
    if (theta_mod < 0) theta_mod += M_PI;
    
    std::vector<cv::Point2f> pts_copy(rim_pts);
    
    for (int trial = 0; trial < n_trials; ++trial) {
        // Subsample 80% of points
        std::shuffle(pts_copy.begin(), pts_copy.end(), rng);
        size_t subsample_size = static_cast<size_t>(0.8 * rim_pts.size());
        
        if (subsample_size < 5) continue;
        
        std::vector<cv::Point2f> subset(pts_copy.begin(), pts_copy.begin() + subsample_size);
        
        // Fit ellipse using OpenCV
        if (subset.size() < 5) continue;
        
        cv::RotatedRect fitted = cv::fitEllipse(subset);
        
        double fitted_a = std::max(fitted.size.width, fitted.size.height) / 2.0;
        double fitted_b = std::min(fitted.size.width, fitted.size.height) / 2.0;
        double fitted_theta = fitted.angle * M_PI / 180.0;
        
        // Normalize angle to [0, PI)
        fitted_theta = std::fmod(fitted_theta, M_PI);
        if (fitted_theta < 0) fitted_theta += M_PI;
        
        shifts_a.push_back(std::abs(fitted_a - a) / (a + 1e-6));
        shifts_b.push_back(std::abs(fitted_b - b) / (b + 1e-6));
        
        // Angle difference (handle wraparound)
        double angle_diff = std::abs(fitted_theta - theta_mod);
        angle_diff = std::min(angle_diff, M_PI - angle_diff);
        shifts_th.push_back(angle_diff);
        
        // Normalized center shift
        double center_shift = std::hypot(fitted.center.x - cx, fitted.center.y - cy);
        shifts_c.push_back(center_shift / (std::sqrt(a * b) + 1e-6));
    }
    
    if (shifts_a.empty()) {
        feats["stab_jitter_a"] = 1.0;
        feats["stab_jitter_b"] = 1.0;
        feats["stab_jitter_theta"] = 1.0;
        feats["stab_center_shift"] = 1.0;
        feats["stab_iou_subsample"] = 0.0;
        return feats;
    }
    
    double ma = vec_mean(shifts_a);
    double mb = vec_mean(shifts_b);
    
    feats["stab_jitter_a"] = ma;
    feats["stab_jitter_b"] = mb;
    feats["stab_jitter_theta"] = vec_mean(shifts_th);
    feats["stab_center_shift"] = vec_mean(shifts_c);
    feats["stab_iou_subsample"] = 1.0 / (1.0 + ma + mb);
    
    return feats;
}

// ------------------------------------------------------------
// Context features (inside/outside intensity comparison)
// These capture intensity contrast between crater interior and exterior
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> context_features(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta_deg,
    const cv::Mat& img  // Grayscale image
) {
    std::unordered_map<std::string, double> feats;
    
    if (img.empty()) {
        feats["ctx_inside_mean"] = 0.0;
        feats["ctx_outside_mean"] = 0.0;
        feats["ctx_diff"] = 0.0;
        feats["ctx_grad_mean"] = 0.0;
        feats["ctx_grad_std"] = 0.0;
        return feats;
    }
    
    int h = img.rows;
    int w = img.cols;
    
    // Convert to float WITHOUT normalizing - matches Python behavior
    // Python keeps original pixel values (0-255)
    cv::Mat img_f;
    if (img.type() == CV_8U) {
        img.convertTo(img_f, CV_32F);  // Keep original scale, no /255
    } else if (img.type() == CV_32F) {
        img_f = img;
    } else {
        feats["ctx_inside_mean"] = 0.0;
        feats["ctx_outside_mean"] = 0.0;
        feats["ctx_diff"] = 0.0;
        feats["ctx_grad_mean"] = 0.0;
        feats["ctx_grad_std"] = 0.0;
        return feats;
    }
    
    // Create interior mask (shrunk ellipse)
    cv::Mat interior_mask = cv::Mat::zeros(h, w, CV_8U);
    cv::Point2f center(static_cast<float>(cx), static_cast<float>(cy));
    double shrink = 0.8;
    cv::ellipse(interior_mask, center, 
                cv::Size2f(static_cast<float>(a * shrink), static_cast<float>(b * shrink)), 
                theta_deg, 0, 360, cv::Scalar(255), -1);
    
    // Create exterior mask (dilated ellipse minus full ellipse)
    cv::Mat full_mask = cv::Mat::zeros(h, w, CV_8U);
    cv::ellipse(full_mask, center, 
                cv::Size2f(static_cast<float>(a), static_cast<float>(b)), 
                theta_deg, 0, 360, cv::Scalar(255), -1);
    
    cv::Mat expanded_mask = cv::Mat::zeros(h, w, CV_8U);
    cv::ellipse(expanded_mask, center, 
                cv::Size2f(static_cast<float>(a * 1.2), static_cast<float>(b * 1.2)), 
                theta_deg, 0, 360, cv::Scalar(255), -1);
    
    cv::Mat outside_mask;
    cv::subtract(expanded_mask, full_mask, outside_mask);
    
    // Compute means (in original pixel value scale, 0-255)
    cv::Scalar inside_mean = cv::mean(img_f, interior_mask);
    cv::Scalar outside_mean = cv::mean(img_f, outside_mask);
    
    // Compute gradient using np.gradient equivalent (NOT Sobel)
    cv::Mat grad_mag = compute_numpy_gradient(img_f);
    
    cv::Scalar global_grad_mean, global_grad_std;
    cv::meanStdDev(grad_mag, global_grad_mean, global_grad_std);
    double g_mean = global_grad_mean[0] + 1e-6;
    double g_std = global_grad_std[0] + 1e-6;
    
    // Sample gradient at rim points
    std::vector<double> rim_grads;
    for (const auto& pt : rim_pts) {
        int x = std::min(std::max(static_cast<int>(std::round(pt.x)), 0), w - 1);
        int y = std::min(std::max(static_cast<int>(std::round(pt.y)), 0), h - 1);
        rim_grads.push_back(grad_mag.at<float>(y, x));
    }
    
    double rim_grad_mean = rim_grads.empty() ? 0.0 : vec_mean(rim_grads);
    double rim_grad_std = rim_grads.empty() ? 0.0 : vec_std(rim_grads);
    
    feats["ctx_inside_mean"] = inside_mean[0];
    feats["ctx_outside_mean"] = outside_mean[0];
    feats["ctx_diff"] = inside_mean[0] - outside_mean[0];
    feats["ctx_grad_mean"] = rim_grad_mean / g_mean;
    feats["ctx_grad_std"] = rim_grad_std / g_std;
    
    return feats;
}

// ------------------------------------------------------------
// Get rim points for a candidate (used by eval3.cpp)
// ------------------------------------------------------------

inline std::vector<cv::Point2f> get_rim_points_for_candidate_fast(
    const cv::Mat& rim_prob,  // CV_32F
    double cx, double cy, double a, double b, double theta_deg,
    int n_samples = 48,       // Matches Python eval3rect.py
    int band_width = 4,       // Matches Python eval3rect.py
    float prob_thresh = 0.2f
) {
    int h = rim_prob.rows;
    int w = rim_prob.cols;
    
    double theta_rad = theta_deg * M_PI / 180.0;
    double cos_t = std::cos(theta_rad);
    double sin_t = std::sin(theta_rad);
    
    std::set<std::pair<int, int>> collected;
    
    for (int s = 0; s < n_samples; ++s) {
        double angle = 2.0 * M_PI * s / n_samples;
        
        // Ellipse point before rotation
        double ex = a * std::cos(angle);
        double ey = b * std::sin(angle);
        
        // Rotate and translate
        double px = cx + cos_t * ex - sin_t * ey;
        double py = cy + sin_t * ex + cos_t * ey;
        
        int x0 = static_cast<int>(std::round(px));
        int y0 = static_cast<int>(std::round(py));
        
        // Search in a small window
        for (int dx = -band_width; dx <= band_width; ++dx) {
            for (int dy = -band_width; dy <= band_width; ++dy) {
                int x = x0 + dx;
                int y = y0 + dy;
                if (x >= 0 && x < w && y >= 0 && y < h) {
                    if (rim_prob.at<float>(y, x) > prob_thresh) {
                        collected.insert({x, y});
                    }
                }
            }
        }
    }
    
    std::vector<cv::Point2f> result;
    result.reserve(collected.size());
    for (const auto& p : collected) {
        result.emplace_back(static_cast<float>(p.first), static_cast<float>(p.second));
    }
    
    return result;
}

// ------------------------------------------------------------
// SNR (Signal-to-Noise Ratio) features
// Helps the ranker identify trustworthy detections in dark images
// ------------------------------------------------------------

struct SNRInfo {
    double snr_raw;              // (mean_rim - mean_interior) / std_background
    double snr_x_illumination;   // snr_raw * (illumination_level / 255.0)
    double snr_illumination_level; // Global image brightness normalized
};

inline SNRInfo compute_snr_features(
    const cv::Mat& img,       // Grayscale image (CV_8U or CV_32F)
    double cx, double cy, double a, double b, double angle_deg,
    double illumination_level  // Mean brightness of image [0-255]
) {
    SNRInfo info = {0.0, 0.0, illumination_level / 255.0};
    
    int h = img.rows;
    int w = img.cols;
    
    // Convert to float if needed
    cv::Mat img_f;
    if (img.type() == CV_8U) {
        img.convertTo(img_f, CV_32F, 1.0 / 255.0);
    } else if (img.type() == CV_32F) {
        img_f = img;
        // Normalize if values are in [0, 255]
        double max_val;
        cv::minMaxLoc(img_f, nullptr, &max_val);
        if (max_val > 1.0) {
            img_f = img_f / 255.0;
        }
    } else {
        return info;  // Unsupported type
    }
    
    // Create ellipse masks
    cv::Point2f center(static_cast<float>(cx), static_cast<float>(cy));
    
    // Interior mask (shrunk by 20%)
    cv::Mat interior_mask = cv::Mat::zeros(h, w, CV_8U);
    double shrink = 0.8;
    cv::ellipse(interior_mask, center, 
                cv::Size2f(static_cast<float>(a * shrink), static_cast<float>(b * shrink)), 
                angle_deg, 0, 360, cv::Scalar(255), -1);
    
    // Rim annulus mask (between 80% and 120%)
    cv::Mat outer_mask = cv::Mat::zeros(h, w, CV_8U);
    double expand = 1.2;
    cv::ellipse(outer_mask, center, 
                cv::Size2f(static_cast<float>(a * expand), static_cast<float>(b * expand)), 
                angle_deg, 0, 360, cv::Scalar(255), -1);
    
    cv::Mat rim_mask = cv::Mat::zeros(h, w, CV_8U);
    cv::subtract(outer_mask, interior_mask, rim_mask);
    
    // Background mask (1.5x ellipse, excluding the full ellipse)
    cv::Mat bg_outer_mask = cv::Mat::zeros(h, w, CV_8U);
    double bg_expand = 1.5;
    cv::ellipse(bg_outer_mask, center, 
                cv::Size2f(static_cast<float>(a * bg_expand), static_cast<float>(b * bg_expand)), 
                angle_deg, 0, 360, cv::Scalar(255), -1);
    
    cv::Mat background_mask = cv::Mat::zeros(h, w, CV_8U);
    cv::subtract(bg_outer_mask, outer_mask, background_mask);
    
    // Compute statistics
    int interior_count = cv::countNonZero(interior_mask);
    int rim_count = cv::countNonZero(rim_mask);
    int bg_count = cv::countNonZero(background_mask);
    
    if (interior_count < 10 || rim_count < 10 || bg_count < 10) {
        return info;  // Not enough pixels
    }
    
    cv::Scalar mean_interior, mean_rim, mean_bg, std_bg;
    cv::meanStdDev(img_f, mean_interior, cv::noArray(), interior_mask);
    cv::meanStdDev(img_f, mean_rim, cv::noArray(), rim_mask);
    cv::meanStdDev(img_f, std_bg, std_bg, background_mask);
    
    double rim_val = mean_rim[0];
    double interior_val = mean_interior[0];
    double bg_std = std_bg[0];
    
    if (bg_std < 1e-6) bg_std = 1e-6;
    
    // SNR = contrast / noise
    info.snr_raw = (rim_val - interior_val) / bg_std;
    
    // Illumination-scaled SNR
    double illum_norm = illumination_level / 255.0;
    info.snr_x_illumination = info.snr_raw * illum_norm;
    info.snr_illumination_level = illum_norm;
    
    return info;
}

// ------------------------------------------------------------
// Main feature extractor (ULTRA FAST mode - for ranking)
// Matches train_ranker.py FEATURE_COLS (21 features)
// Use for ranking where speed is critical
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> extract_crater_features_ultra_fast(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta,
    const cv::Mat& rim_prob,
    int img_h, int img_w,
    const cv::Mat& img_orig = cv::Mat(),  // Optional: used if full_mode
    double illumination_level = 128.0     // Optional: used if full_mode
) {
    (void)img_orig;          // Suppress unused parameter warning
    (void)illumination_level; // Suppress unused parameter warning
    
    std::unordered_map<std::string, double> feats;
    
    // Geometry features (11)
    auto geom = geometry_features(rim_pts, cx, cy, a, b, theta, img_h, img_w);
    feats.insert(geom.begin(), geom.end());
    
    // Rim probability features (8)
    auto rim = rim_prob_features(rim_pts, rim_prob);
    feats.insert(rim.begin(), rim.end());
    
    // Meta features (2)
    feats["meta_log_radius"] = std::log(std::sqrt(a * b) / std::sqrt(img_h * img_w) + 1e-6);
    feats["meta_log_area"] = std::log(M_PI * a * b / (img_h * img_w) + 1e-6);
    
    return feats;
}

// ------------------------------------------------------------
// Full feature extractor (for CLASSIFICATION)
// Matches train_classifier.py FEATURE_COLS (45 features)
// Includes Mahanti, morphology, stability, context, SNR
// Use for classification where accuracy is critical
// ------------------------------------------------------------

inline std::unordered_map<std::string, double> extract_crater_features_full(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta,
    const cv::Mat& rim_prob,
    const cv::Mat& img,                  // Grayscale image (required for full features)
    int img_h, int img_w,
    double illumination_level = 128.0,   // Mean brightness [0-255]
    bool compute_stability = true        // Stability is slow, make optional
) {
    std::unordered_map<std::string, double> feats;
    
    // === GEOMETRY FEATURES (11) ===
    auto geom = geometry_features(rim_pts, cx, cy, a, b, theta, img_h, img_w);
    feats.insert(geom.begin(), geom.end());
    
    // === RIM PROBABILITY FEATURES (8) ===
    auto rim = rim_prob_features(rim_pts, rim_prob);
    feats.insert(rim.begin(), rim.end());
    
    // === MORPHOLOGY FEATURES (6) ===
    // Use fast version (no mask required)
    auto morph = morphology_features_fast(a, b, img_h, img_w);
    feats.insert(morph.begin(), morph.end());
    
    // === STABILITY FEATURES (5) ===
    if (compute_stability) {
        auto stab = stability_features(rim_pts, cx, cy, a, b, theta, 5);
        feats.insert(stab.begin(), stab.end());
    } else {
        feats["stab_jitter_a"] = 0.0;
        feats["stab_jitter_b"] = 0.0;
        feats["stab_jitter_theta"] = 0.0;
        feats["stab_center_shift"] = 0.0;
        feats["stab_iou_subsample"] = 1.0;
    }
    
    // === MAHANTI DEGRADATION FEATURES (4) ===
    auto mahanti = mahanti_features(rim_pts, cx, cy, a, b, theta, img);
    feats.insert(mahanti.begin(), mahanti.end());
    
    // === CONTEXT FEATURES (5) ===
    auto ctx = context_features(rim_pts, cx, cy, a, b, theta, img);
    feats.insert(ctx.begin(), ctx.end());
    
    // === META FEATURES (2) ===
    feats["meta_log_radius"] = std::log(std::sqrt(a * b) / std::sqrt(img_h * img_w) + 1e-6);
    feats["meta_log_area"] = feats["morph_area_log"];  // Same as morph_area_log
    
    // === SNR FEATURES (3) ===
    auto snr = compute_snr_features(img, cx, cy, a, b, theta, illumination_level);
    feats["snr_raw"] = snr.snr_raw;
    feats["snr_x_illumination"] = snr.snr_x_illumination;
    feats["snr_illumination_level"] = snr.snr_illumination_level;
    
    return feats;
}

// Convert feature map to ordered vector for model input
inline std::vector<double> features_to_vector(const std::unordered_map<std::string, double>& feats) {
    std::vector<double> result;
    result.reserve(FEATURE_NAMES.size());
    for (const auto& name : FEATURE_NAMES) {
        auto it = feats.find(name);
        if (it != feats.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(0.0);  // Default for missing features
        }
    }
    return result;
}

// Classifier-specific feature names (31 features) - matches classifier_ensemble.h exactly
// Note: This is a SUBSET of FEATURE_NAMES, excluding morphology, stability, and meta features
static const std::vector<std::string> CLASSIFIER_FEATURE_NAMES = {
    // Geometry features (11)
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
    // Rim probability features (8)
    "rim_prob_mean",
    "rim_prob_std",
    "rim_prob_min",
    "rim_prob_p20",
    "rim_prob_p80",
    "rim_prob_frac_above_50",
    "rim_prob_frac_above_70",
    "rim_prob_entropy",
    // Mahanti degradation features (4)
    "mahanti_median_slope",
    "mahanti_slope_std",
    "mahanti_depth_ratio",
    "mahanti_rim_sharpness",
    // Context features (5)
    "ctx_inside_mean",
    "ctx_outside_mean",
    "ctx_diff",
    "ctx_grad_mean",
    "ctx_grad_std",
    // SNR features (3)
    "snr_raw",
    "snr_x_illumination",
    "snr_illumination_level"
};

// ------------------------------------------------------------
// LEAN Classifier-specific feature extraction (31 features ONLY)
// Skips morphology, stability, and meta features that aren't used
// Much faster than extract_crater_features_full
// 
// IMPORTANT: Applies the same feature scaling as Python training:
//   - Geometry features × 0.85
//   - Mahanti features × 1.8
// ------------------------------------------------------------

// Feature scaling constants (must match train_classifier.py)
constexpr double GEOMETRY_SCALE = 0.85;
constexpr double MAHANTI_SCALE = 1.8;

inline std::vector<double> extract_classifier_features(
    const std::vector<cv::Point2f>& rim_pts,
    double cx, double cy, double a, double b, double theta,
    const cv::Mat& rim_prob,
    const cv::Mat& img,                  // Grayscale image (CV_8U or CV_32F)
    int img_h, int img_w,
    double illumination_level = 128.0
) {
    std::vector<double> result(31, 0.0);
    
    if (rim_pts.size() < 5) {
        return result;  // Return zeros if not enough points
    }
    
    // === GEOMETRY FEATURES (11) - indices 0-10 ===
    // Apply GEOMETRY_SCALE (0.85) to match Python training
    auto geom = geometry_features(rim_pts, cx, cy, a, b, theta, img_h, img_w);
    result[0] = geom["geometry_eccentricity"] * GEOMETRY_SCALE;
    result[1] = geom["geometry_axis_ratio"] * GEOMETRY_SCALE;
    result[2] = geom["geometry_ellipse_area"] * GEOMETRY_SCALE;
    result[3] = geom["geometry_area_ratio"] * GEOMETRY_SCALE;
    result[4] = geom["geometry_resid_rms"] * GEOMETRY_SCALE;
    result[5] = geom["geometry_resid_p90"] * GEOMETRY_SCALE;
    result[6] = geom["geometry_support_ratio"] * GEOMETRY_SCALE;
    result[7] = geom["geometry_angular_coverage"] * GEOMETRY_SCALE;
    result[8] = geom["geometry_angular_std"] * GEOMETRY_SCALE;
    result[9] = geom["geometry_max_gap"] * GEOMETRY_SCALE;
    result[10] = geom["geometry_condition"] * GEOMETRY_SCALE;
    
    // === RIM PROBABILITY FEATURES (8) - indices 11-18 ===
    // No scaling for rim probability features
    auto rim = rim_prob_features(rim_pts, rim_prob);
    result[11] = rim["rim_prob_mean"];
    result[12] = rim["rim_prob_std"];
    result[13] = rim["rim_prob_min"];
    result[14] = rim["rim_prob_p20"];
    result[15] = rim["rim_prob_p80"];
    result[16] = rim["rim_prob_frac_above_50"];
    result[17] = rim["rim_prob_frac_above_70"];
    result[18] = rim["rim_prob_entropy"];
    
    // === MAHANTI DEGRADATION FEATURES (4) - indices 19-22 ===
    // Apply MAHANTI_SCALE (1.8) to match Python training
    auto mahanti = mahanti_features(rim_pts, cx, cy, a, b, theta, img);
    result[19] = mahanti["mahanti_median_slope"] * MAHANTI_SCALE;
    result[20] = mahanti["mahanti_slope_std"] * MAHANTI_SCALE;
    result[21] = mahanti["mahanti_depth_ratio"] * MAHANTI_SCALE;
    result[22] = mahanti["mahanti_rim_sharpness"] * MAHANTI_SCALE;
    
    // === CONTEXT FEATURES (5) - indices 23-27 ===
    // No scaling for context features
    auto ctx = context_features(rim_pts, cx, cy, a, b, theta, img);
    result[23] = ctx["ctx_inside_mean"];
    result[24] = ctx["ctx_outside_mean"];
    result[25] = ctx["ctx_diff"];
    result[26] = ctx["ctx_grad_mean"];
    result[27] = ctx["ctx_grad_std"];
    
    // === SNR FEATURES (3) - indices 28-30 ===
    // No scaling for SNR features
    auto snr = compute_snr_features(img, cx, cy, a, b, theta, illumination_level);
    result[28] = snr.snr_raw;
    result[29] = snr.snr_x_illumination;
    result[30] = snr.snr_illumination_level;
    
    return result;
}

// Convert feature map to classifier vector (31 features matching classifier_ensemble.h)
inline std::vector<double> features_to_classifier_vector(const std::unordered_map<std::string, double>& feats) {
    std::vector<double> result;
    result.reserve(CLASSIFIER_FEATURE_NAMES.size());
    for (const auto& name : CLASSIFIER_FEATURE_NAMES) {
        auto it = feats.find(name);
        if (it != feats.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(0.0);  // Default for missing features
        }
    }
    return result;
}

}  // namespace RankingFeatures

#endif  // RANKING_FEATURES_MULTIRES_HPP
