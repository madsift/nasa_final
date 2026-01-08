#ifndef POLAR_HPP
#define POLAR_HPP

/**
 * C++ port of polar.py - Polar ellipse fitting and ranking features for crater detection.
 * 
 * This module provides:
 * - Angular representation of rim points
 * - Polar ellipse fitting with optional center refinement
 * - Ranking features based on polar analysis
 * 
 * NOTE: This implementation uses a simplified least-squares solver for the polar fitting
 * since scipy.optimize.least_squares is not available in C++. For production use,
 * consider integrating Ceres Solver or similar for more robust optimization.
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <random>
#include <numeric>

namespace Polar {

// ============================================================
// Constants
// ============================================================
static constexpr double PI = 3.14159265358979323846;
static constexpr double EPS = 1e-9;

// ============================================================
// Step (2): Angular-domain representation
// ============================================================

struct AngularRepresentation {
    std::vector<double> theta_centers;  // Angular bin centers
    std::vector<double> r_theta;        // Radius per bin (NaN for missing)
};

/**
 * Converts rim points into r(theta) representation.
 * 
 * @param xs X coordinates of rim points
 * @param ys Y coordinates of rim points
 * @param cx, cy Center of ellipse
 * @param num_bins Number of angular bins (default 360)
 * @param mode "max" for outer rim crest, "min" for inner, "median" for robust
 * @return AngularRepresentation with theta_centers and r_theta
 */
inline AngularRepresentation angular_representation(
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    double cx, double cy,
    int num_bins = 360,
    const std::string& mode = "max"
) {
    AngularRepresentation result;
    result.theta_centers.resize(num_bins);
    result.r_theta.resize(num_bins, std::nan(""));
    
    // Compute bin edges: [-PI, PI) divided into num_bins
    double bin_width = 2.0 * PI / num_bins;
    
    // Compute theta_centers
    for (int i = 0; i < num_bins; ++i) {
        result.theta_centers[i] = -PI + (i + 0.5) * bin_width;
    }
    
    // Aggregate points into bins
    std::vector<std::vector<double>> bin_radii(num_bins);
    
    for (size_t i = 0; i < xs.size(); ++i) {
        double dx = xs[i] - cx;
        double dy = ys[i] - cy;
        double theta = std::atan2(dy, dx);  // [-PI, PI]
        double r = std::sqrt(dx * dx + dy * dy);
        
        int bin_idx = static_cast<int>((theta + PI) / bin_width);
        bin_idx = std::min(std::max(bin_idx, 0), num_bins - 1);
        bin_radii[bin_idx].push_back(r);
    }
    
    // Compute summary statistic for each bin
    for (int i = 0; i < num_bins; ++i) {
        if (!bin_radii[i].empty()) {
            if (mode == "max") {
                result.r_theta[i] = *std::max_element(bin_radii[i].begin(), bin_radii[i].end());
            } else if (mode == "min") {
                result.r_theta[i] = *std::min_element(bin_radii[i].begin(), bin_radii[i].end());
            } else {  // median
                std::sort(bin_radii[i].begin(), bin_radii[i].end());
                size_t n = bin_radii[i].size();
                if (n % 2 == 0) {
                    result.r_theta[i] = (bin_radii[i][n/2 - 1] + bin_radii[i][n/2]) / 2.0;
                } else {
                    result.r_theta[i] = bin_radii[i][n/2];
                }
            }
        }
    }
    
    return result;
}

// ============================================================
// Step (5): Angular continuity
// ============================================================

/**
 * Check if angular support is sufficient.
 * 
 * @param r_theta Radius per angular bin
 * @param min_support Minimum fraction of valid bins required
 * @return Pointer to valid mask, or nullptr if insufficient support
 */
inline std::vector<bool> angular_continuity_mask(
    const std::vector<double>& r_theta,
    double min_support = 0.5
) {
    std::vector<bool> valid(r_theta.size());
    int valid_count = 0;
    
    for (size_t i = 0; i < r_theta.size(); ++i) {
        valid[i] = !std::isnan(r_theta[i]);
        if (valid[i]) valid_count++;
    }
    
    if (static_cast<double>(valid_count) / r_theta.size() < min_support) {
        return {};  // Empty vector indicates rejection
    }
    
    return valid;
}

// ============================================================
// Step (6): Inner–outer rim coupling
// ============================================================

/**
 * Enforces physical ordering: outer rim must be outside core.
 */
inline std::vector<bool> inner_outer_coupling(
    const std::vector<double>& r_outer,
    const std::vector<double>& r_inner,
    double margin = 1.0
) {
    std::vector<bool> valid(r_outer.size(), false);
    
    for (size_t i = 0; i < r_outer.size(); ++i) {
        if (!std::isnan(r_outer[i]) && !std::isnan(r_inner[i])) {
            valid[i] = (r_outer[i] > r_inner[i] + margin);
        }
    }
    
    return valid;
}

// ============================================================
// Polar ellipse model
// ============================================================

/**
 * Compute ellipse radius at given angle.
 * r(θ) = (a*b) / sqrt((b*cos(θ-φ))² + (a*sin(θ-φ))²)
 */
inline double ellipse_radius(double theta, double a, double b, double phi) {
    double ct = std::cos(theta - phi);
    double st = std::sin(theta - phi);
    double denom = std::sqrt((b * ct) * (b * ct) + (a * st) * (a * st));
    return (a * b) / (denom + EPS);
}

// ============================================================
// Levenberg-Marquardt solver for polar ellipse fitting
// Matches scipy.optimize.least_squares with Huber loss
// ============================================================

struct PolarFitResult {
    double a, b, phi;       // Semi-axes and orientation
    double cx, cy;          // Center (may be refined)
    bool success;
    double final_cost;
    int iterations;
};

/**
 * Compute Huber-weighted residual and cost.
 * Matches Python: polar_residual function with Huber loss.
 */
inline void compute_polar_residuals(
    const std::vector<double>& theta,
    const std::vector<double>& r_obs,
    const std::vector<bool>& mask,
    double log_a, double log_b, double phi,
    double f_scale,  // Huber threshold (default 0.05 in Python)
    std::vector<double>& residuals,
    std::vector<double>& weights,
    double& cost
) {
    double a = std::exp(log_a);
    double b = std::exp(log_b);
    
    residuals.clear();
    weights.clear();
    cost = 0.0;
    
    for (size_t i = 0; i < theta.size(); ++i) {
        if (!mask[i]) continue;
        
        double r_pred = ellipse_radius(theta[i], a, b, phi);
        double rel_err = (r_pred - r_obs[i]) / (r_obs[i] + EPS);
        
        residuals.push_back(rel_err);
        
        // Huber weighting (soft L1)
        double abs_err = std::abs(rel_err);
        double rho;  // sqrt(2 * huber_loss)
        double w;    // weight = d(rho)/d(residual)
        
        if (abs_err <= f_scale) {
            // Quadratic region
            rho = abs_err;
            w = 1.0;
        } else {
            // Linear region
            rho = std::sqrt(2.0 * f_scale * abs_err - f_scale * f_scale);
            w = f_scale / abs_err;
        }
        
        weights.push_back(w);
        cost += (abs_err <= f_scale) ? 
                0.5 * rel_err * rel_err : 
                f_scale * abs_err - 0.5 * f_scale * f_scale;
    }
    
    // Add axis ratio regularization penalty (same as Python: 0.05 * (log(a/b))^2)
    double ratio_term = 0.05 * (log_a - log_b);
    residuals.push_back(ratio_term);
    weights.push_back(1.0);
    cost += 0.5 * ratio_term * ratio_term;
}

/**
 * Compute Jacobian of weighted residuals w.r.t. [log_a, log_b, phi].
 * Uses numerical differentiation for robustness.
 */
inline void compute_jacobian(
    const std::vector<double>& theta,
    const std::vector<double>& r_obs,
    const std::vector<bool>& mask,
    double log_a, double log_b, double phi,
    double f_scale,
    const std::vector<double>& weights,
    std::vector<std::vector<double>>& J
) {
    const double eps = 1e-7;
    int n_residuals = static_cast<int>(weights.size());
    
    J.assign(n_residuals, std::vector<double>(3, 0.0));
    
    // Compute residuals at current point
    std::vector<double> res0, w0;
    double cost0;
    compute_polar_residuals(theta, r_obs, mask, log_a, log_b, phi, f_scale, res0, w0, cost0);
    
    // Numerical derivatives for each parameter
    double params[3] = {log_a, log_b, phi};
    
    for (int p = 0; p < 3; ++p) {
        double orig = params[p];
        params[p] = orig + eps;
        
        std::vector<double> res_p, w_p;
        double cost_p;
        compute_polar_residuals(theta, r_obs, mask, params[0], params[1], params[2], f_scale, res_p, w_p, cost_p);
        
        for (int i = 0; i < n_residuals; ++i) {
            // Weighted residual derivative
            J[i][p] = std::sqrt(weights[i]) * (res_p[i] - res0[i]) / eps;
        }
        
        params[p] = orig;
    }
}

/**
 * Solve 3x3 linear system using Cholesky decomposition.
 * Solves (J^T W J + lambda*I) * delta = -J^T W r
 */
inline bool solve_3x3(
    const std::vector<std::vector<double>>& JtWJ,  // 3x3
    const std::vector<double>& JtWr,                // 3x1
    double lambda,
    std::vector<double>& delta                      // 3x1 output
) {
    // Add damping
    double A[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            A[i][j] = JtWJ[i][j];
        }
        A[i][i] += lambda;
    }
    
    double b[3] = {-JtWr[0], -JtWr[1], -JtWr[2]};
    
    // Gaussian elimination with partial pivoting
    for (int k = 0; k < 3; ++k) {
        // Find pivot
        int max_row = k;
        for (int i = k + 1; i < 3; ++i) {
            if (std::abs(A[i][k]) > std::abs(A[max_row][k])) {
                max_row = i;
            }
        }
        
        // Swap rows
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(b[k], b[max_row]);
        }
        
        if (std::abs(A[k][k]) < 1e-12) {
            return false;  // Singular
        }
        
        // Eliminate column
        for (int i = k + 1; i < 3; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < 3; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    
    // Back substitution
    delta.resize(3);
    for (int i = 2; i >= 0; --i) {
        delta[i] = b[i];
        for (int j = i + 1; j < 3; ++j) {
            delta[i] -= A[i][j] * delta[j];
        }
        delta[i] /= A[i][i];
    }
    
    return true;
}

/**
 * Fit ellipse to angular representation using Levenberg-Marquardt.
 * Matches scipy.optimize.least_squares behavior closely.
 */
inline PolarFitResult fit_polar_ellipse_lm(
    const std::vector<double>& theta,
    const std::vector<double>& r_obs,
    const std::vector<bool>& mask,
    double initial_r,
    double f_scale = 0.05,  // Same as Python default
    int max_iterations = 200,
    double tol = 1e-8
) {
    PolarFitResult result;
    result.success = false;
    result.iterations = 0;
    
    // Count valid points
    int valid_count = 0;
    for (bool v : mask) if (v) valid_count++;
    
    if (valid_count < 10) {
        return result;
    }
    
    // Initial guess: circle
    double log_a = std::log(initial_r);
    double log_b = std::log(initial_r);
    double phi = 0.0;
    
    // Initial cost
    std::vector<double> residuals, weights;
    double cost;
    compute_polar_residuals(theta, r_obs, mask, log_a, log_b, phi, f_scale, residuals, weights, cost);
    
    double lambda = 1e-3;  // LM damping factor
    double prev_cost = cost;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        result.iterations = iter + 1;
        
        // Compute Jacobian
        std::vector<std::vector<double>> J;
        compute_jacobian(theta, r_obs, mask, log_a, log_b, phi, f_scale, weights, J);
        
        int n = static_cast<int>(residuals.size());
        
        // Compute J^T W J and J^T W r
        std::vector<std::vector<double>> JtWJ(3, std::vector<double>(3, 0.0));
        std::vector<double> JtWr(3, 0.0);
        
        for (int i = 0; i < n; ++i) {
            double w = weights[i];
            for (int j = 0; j < 3; ++j) {
                JtWr[j] += J[i][j] * std::sqrt(w) * residuals[i] * std::sqrt(w);
                for (int k = 0; k < 3; ++k) {
                    JtWJ[j][k] += J[i][j] * w * J[i][k];
                }
            }
        }
        
        // Solve for step
        std::vector<double> delta;
        if (!solve_3x3(JtWJ, JtWr, lambda, delta)) {
            lambda *= 10;
            continue;
        }
        
        // Try step
        double new_log_a = log_a + delta[0];
        double new_log_b = log_b + delta[1];
        double new_phi = phi + delta[2];
        
        // Clamp to reasonable bounds
        new_log_a = std::max(std::log(3.0), std::min(new_log_a, std::log(500.0)));
        new_log_b = std::max(std::log(3.0), std::min(new_log_b, std::log(500.0)));
        
        std::vector<double> new_residuals, new_weights;
        double new_cost;
        compute_polar_residuals(theta, r_obs, mask, new_log_a, new_log_b, new_phi, f_scale, 
                               new_residuals, new_weights, new_cost);
        
        if (new_cost < cost) {
            // Accept step
            log_a = new_log_a;
            log_b = new_log_b;
            phi = new_phi;
            residuals = new_residuals;
            weights = new_weights;
            cost = new_cost;
            lambda = std::max(lambda / 3.0, 1e-10);
            
            // Check convergence
            if (std::abs(prev_cost - cost) < tol * cost && iter > 3) {
                result.success = true;
                break;
            }
            prev_cost = cost;
        } else {
            // Reject step, increase damping
            lambda = std::min(lambda * 2.0, 1e7);
        }
        
        // Also check gradient convergence
        double grad_norm = std::sqrt(JtWr[0]*JtWr[0] + JtWr[1]*JtWr[1] + JtWr[2]*JtWr[2]);
        if (grad_norm < tol) {
            result.success = true;
            break;
        }
    }
    
    result.a = std::exp(log_a);
    result.b = std::exp(log_b);
    result.phi = phi;
    result.final_cost = cost;
    
    // Ensure a >= b
    if (result.b > result.a) {
        std::swap(result.a, result.b);
        result.phi += PI / 2;
    }
    
    // Normalize phi to [0, PI)
    result.phi = std::fmod(result.phi + PI, PI);
    if (result.phi < 0) result.phi += PI;
    
    result.success = true;
    return result;
}

// ============================================================
// Main scorer-aware ellipse fitting entry point
// ============================================================

struct EllipseFitResult {
    double cx, cy, a, b, phi;
    bool valid;
    double fit_cost;
};

/**
 * Fit crater ellipse using polar representation.
 * 
 * @param rim_pts Rim point coordinates
 * @param core_pts Core point coordinates (for inner-outer coupling)
 * @param center Initial center estimate
 * @param num_bins Number of angular bins
 * @param min_support Minimum fraction of bins with valid data
 * @return EllipseFitResult or invalid result if fitting fails
 */
inline EllipseFitResult fit_crater_ellipse_polar(
    const std::vector<cv::Point2f>& rim_pts,
    const std::vector<cv::Point2f>& core_pts,
    cv::Point2f center,
    int num_bins = 360,
    double min_support = 0.5
) {
    EllipseFitResult result;
    result.valid = false;
    result.cx = center.x;
    result.cy = center.y;
    
    // Convert to vectors
    std::vector<float> rim_xs(rim_pts.size()), rim_ys(rim_pts.size());
    std::vector<float> core_xs(core_pts.size()), core_ys(core_pts.size());
    
    for (size_t i = 0; i < rim_pts.size(); ++i) {
        rim_xs[i] = rim_pts[i].x;
        rim_ys[i] = rim_pts[i].y;
    }
    for (size_t i = 0; i < core_pts.size(); ++i) {
        core_xs[i] = core_pts[i].x;
        core_ys[i] = core_pts[i].y;
    }
    
    // Angular representations
    auto angular_outer = angular_representation(rim_xs, rim_ys, center.x, center.y, num_bins, "max");
    auto angular_inner = angular_representation(core_xs, core_ys, center.x, center.y, num_bins, "max");
    
    // Angular continuity check
    auto mask_cont = angular_continuity_mask(angular_outer.r_theta, min_support);
    if (mask_cont.empty()) {
        return result;
    }
    
    // Inner-outer coupling
    auto mask_phys = inner_outer_coupling(angular_outer.r_theta, angular_inner.r_theta);
    
    // Combine masks
    std::vector<bool> mask(num_bins);
    int valid_count = 0;
    for (int i = 0; i < num_bins; ++i) {
        mask[i] = mask_cont[i] && (core_pts.empty() || mask_phys[i]);
        if (mask[i]) valid_count++;
    }
    
    if (valid_count < static_cast<int>(0.3 * num_bins)) {
        return result;
    }
    
    // Compute median radius
    std::vector<double> valid_radii;
    for (int i = 0; i < num_bins; ++i) {
        if (mask[i]) {
            valid_radii.push_back(angular_outer.r_theta[i]);
        }
    }
    std::sort(valid_radii.begin(), valid_radii.end());
    double r_med = valid_radii[valid_radii.size() / 2];
    
    if (!std::isfinite(r_med) || r_med < 3.0) {
        return result;
    }
    
    // Fit ellipse using Levenberg-Marquardt (matches scipy.optimize.least_squares)
    auto fit = fit_polar_ellipse_lm(
        angular_outer.theta_centers,
        angular_outer.r_theta,
        mask,
        r_med,
        0.05  // f_scale = same as Python default
    );
    
    if (!fit.success || fit.a < 3 || fit.b < 3) {
        return result;
    }
    
    result.cx = center.x;
    result.cy = center.y;
    result.a = fit.a;
    result.b = fit.b;
    result.phi = fit.phi;
    result.fit_cost = fit.final_cost;
    result.valid = true;
    
    return result;
}

// ============================================================
// RANKING FEATURES - Tier 1 (MUST-HAVE, highest impact)
// ============================================================

/**
 * Tier 1 - Feature 1: Angular support fraction (rim completeness).
 */
inline double compute_angular_support_fraction(const std::vector<double>& r_theta) {
    int valid_count = 0;
    for (double r : r_theta) {
        if (!std::isnan(r)) valid_count++;
    }
    return static_cast<double>(valid_count) / r_theta.size();
}

/**
 * Tier 1 - Feature 2: Radial residual energy (ellipse goodness-of-fit).
 */
struct RadialResidualInfo {
    double rmse;
    double mae;
    double relative_rmse;
};

inline RadialResidualInfo compute_radial_residual_energy(
    const std::vector<double>& theta,
    const std::vector<double>& r_obs,
    double a, double b, double phi,
    const std::vector<bool>& mask
) {
    RadialResidualInfo info = {0.0, 0.0, 0.0};
    
    std::vector<double> residuals;
    std::vector<double> actuals;
    
    for (size_t i = 0; i < theta.size(); ++i) {
        if (!mask.empty() && !mask[i]) continue;
        if (std::isnan(r_obs[i])) continue;
        
        double r_pred = ellipse_radius(theta[i], a, b, phi);
        residuals.push_back(r_pred - r_obs[i]);
        actuals.push_back(r_obs[i]);
    }
    
    if (residuals.empty()) return info;
    
    double sum_sq = 0.0, sum_abs = 0.0, sum_actual = 0.0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        sum_sq += residuals[i] * residuals[i];
        sum_abs += std::abs(residuals[i]);
        sum_actual += actuals[i];
    }
    
    info.rmse = std::sqrt(sum_sq / residuals.size());
    info.mae = sum_abs / residuals.size();
    info.relative_rmse = info.rmse / (sum_actual / residuals.size() + EPS);
    
    return info;
}

/**
 * Tier 1 - Feature 3: Inner-outer separation stability (rim sharpness consistency).
 */
struct SeparationInfo {
    double mean_separation;
    double std_separation;
    double cv_separation;
};

inline SeparationInfo compute_separation_stability(
    const std::vector<double>& r_outer,
    const std::vector<double>& r_inner
) {
    SeparationInfo info = {std::nan(""), std::nan(""), std::nan("")};
    
    std::vector<double> separations;
    for (size_t i = 0; i < r_outer.size() && i < r_inner.size(); ++i) {
        if (!std::isnan(r_outer[i]) && !std::isnan(r_inner[i])) {
            separations.push_back(r_outer[i] - r_inner[i]);
        }
    }
    
    if (separations.size() < 10) return info;
    
    double sum = 0.0;
    for (double s : separations) sum += s;
    double mean = sum / separations.size();
    
    double var = 0.0;
    for (double s : separations) var += (s - mean) * (s - mean);
    double std = std::sqrt(var / separations.size());
    
    info.mean_separation = mean;
    info.std_separation = std;
    info.cv_separation = std / (mean + EPS);
    
    return info;
}

// ============================================================
// RANKING FEATURES - Tier 2 (STRONG secondary features)
// ============================================================

/**
 * Tier 2 - Feature 4: Largest angular gap.
 */
struct GapInfo {
    int largest_gap_bins;
    double largest_gap_degrees;
    double largest_gap_fraction;
};

inline GapInfo compute_largest_angular_gap(const std::vector<double>& r_theta) {
    int num_bins = static_cast<int>(r_theta.size());
    
    // Find longest run of invalid (NaN) bins, handling wrap-around
    int max_gap = 0;
    int current_gap = 0;
    
    // Two passes for circular handling
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < num_bins; ++i) {
            if (std::isnan(r_theta[i])) {
                current_gap++;
                max_gap = std::max(max_gap, current_gap);
            } else {
                current_gap = 0;
            }
        }
    }
    
    // Cap at num_bins
    max_gap = std::min(max_gap, num_bins);
    
    GapInfo info;
    info.largest_gap_bins = max_gap;
    info.largest_gap_degrees = (static_cast<double>(max_gap) / num_bins) * 360.0;
    info.largest_gap_fraction = static_cast<double>(max_gap) / num_bins;
    
    return info;
}

// ============================================================
// RANKING FEATURES - Tier 3 (USEFUL tie-breakers)
// ============================================================

/**
 * Tier 3 - Feature 7: Directional asymmetry score.
 */
struct AsymmetryInfo {
    double mean_asymmetry;
    double max_asymmetry;
    double asymmetry_fraction;
};

inline AsymmetryInfo compute_directional_asymmetry(const std::vector<double>& r_theta) {
    AsymmetryInfo info = {std::nan(""), std::nan(""), std::nan("")};
    
    int num_bins = static_cast<int>(r_theta.size());
    int half = num_bins / 2;
    
    std::vector<double> asymmetries;
    
    for (int i = 0; i < half; ++i) {
        int opp_idx = i + half;
        if (!std::isnan(r_theta[i]) && !std::isnan(r_theta[opp_idx])) {
            double avg = (r_theta[i] + r_theta[opp_idx]) / 2.0;
            double diff = std::abs(r_theta[i] - r_theta[opp_idx]);
            asymmetries.push_back(diff / (avg + EPS));
        }
    }
    
    if (asymmetries.size() < 5) return info;
    
    double sum = 0.0;
    double max_val = 0.0;
    int above_20pct = 0;
    
    for (double a : asymmetries) {
        sum += a;
        max_val = std::max(max_val, a);
        if (a > 0.2) above_20pct++;
    }
    
    info.mean_asymmetry = sum / asymmetries.size();
    info.max_asymmetry = max_val;
    info.asymmetry_fraction = static_cast<double>(above_20pct) / asymmetries.size();
    
    return info;
}

/**
 * Tier 3 - Feature 8: Polar roughness spectrum.
 */
struct RoughnessInfo {
    double total_roughness;
    double high_freq_energy;
    double roughness_ratio;
};

inline RoughnessInfo compute_polar_roughness_spectrum(
    const std::vector<double>& theta,
    const std::vector<double>& r_obs,
    double a, double b, double phi,
    const std::vector<bool>& mask
) {
    RoughnessInfo info = {std::nan(""), std::nan(""), std::nan("")};
    
    // Get valid residuals
    std::vector<double> residuals;
    for (size_t i = 0; i < theta.size(); ++i) {
        if (!mask.empty() && !mask[i]) continue;
        if (std::isnan(r_obs[i])) continue;
        
        double r_pred = ellipse_radius(theta[i], a, b, phi);
        residuals.push_back(r_obs[i] - r_pred);
    }
    
    if (residuals.size() < 20) return info;
    
    // Gradient-based roughness (simpler than FFT)
    double total_energy = 0.0;
    double grad_energy = 0.0;
    
    for (double r : residuals) {
        total_energy += r * r;
    }
    
    for (size_t i = 1; i < residuals.size(); ++i) {
        double grad = residuals[i] - residuals[i-1];
        grad_energy += grad * grad;
    }
    
    info.total_roughness = total_energy;
    info.high_freq_energy = grad_energy;
    info.roughness_ratio = grad_energy / (total_energy + 1e-10);
    
    return info;
}

// ============================================================
// COMPOSITE RANKING FUNCTIONS
// ============================================================

/**
 * All ranking features for a single crater candidate.
 */
inline std::unordered_map<std::string, double> extract_all_ranking_features(
    const std::vector<float>& rim_xs, const std::vector<float>& rim_ys,
    const std::vector<float>& core_xs, const std::vector<float>& core_ys,
    double cx, double cy, double a, double b, double phi,
    int num_bins = 360
) {
    std::unordered_map<std::string, double> features;
    
    // Compute angular representations
    auto angular_outer = angular_representation(rim_xs, rim_ys, cx, cy, num_bins, "max");
    auto angular_inner = angular_representation(core_xs, core_ys, cx, cy, num_bins, "max");
    
    std::vector<bool> mask(num_bins);
    for (int i = 0; i < num_bins; ++i) {
        mask[i] = !std::isnan(angular_outer.r_theta[i]);
    }
    
    // Tier 1 features
    features["angular_support"] = compute_angular_support_fraction(angular_outer.r_theta);
    
    auto residual_info = compute_radial_residual_energy(
        angular_outer.theta_centers, angular_outer.r_theta, a, b, phi, mask);
    features["residual_rmse"] = residual_info.rmse;
    features["residual_mae"] = residual_info.mae;
    features["residual_relative_rmse"] = residual_info.relative_rmse;
    
    auto sep_info = compute_separation_stability(angular_outer.r_theta, angular_inner.r_theta);
    features["separation_mean"] = sep_info.mean_separation;
    features["separation_std"] = sep_info.std_separation;
    features["separation_cv"] = sep_info.cv_separation;
    
    // Tier 2 features
    auto gap_info = compute_largest_angular_gap(angular_outer.r_theta);
    features["largest_gap_degrees"] = gap_info.largest_gap_degrees;
    features["largest_gap_fraction"] = gap_info.largest_gap_fraction;
    
    features["axis_ratio"] = (b > 0) ? (a / b) : std::nan("");
    
    // Tier 3 features
    auto asym_info = compute_directional_asymmetry(angular_outer.r_theta);
    features["mean_asymmetry"] = asym_info.mean_asymmetry;
    features["max_asymmetry"] = asym_info.max_asymmetry;
    features["asymmetry_fraction"] = asym_info.asymmetry_fraction;
    
    auto roughness_info = compute_polar_roughness_spectrum(
        angular_outer.theta_centers, angular_outer.r_theta, a, b, phi, mask);
    features["roughness_ratio"] = roughness_info.roughness_ratio;
    features["high_freq_energy"] = roughness_info.high_freq_energy;
    
    return features;
}

/**
 * Minimal 4-feature set for fast ranking.
 */
inline std::unordered_map<std::string, double> extract_minimal_ranking_features(
    const std::vector<float>& rim_xs, const std::vector<float>& rim_ys,
    const std::vector<float>& core_xs, const std::vector<float>& core_ys,
    double cx, double cy, double a, double b, double phi,
    int num_bins = 360
) {
    std::unordered_map<std::string, double> features;
    
    auto angular_outer = angular_representation(rim_xs, rim_ys, cx, cy, num_bins, "max");
    auto angular_inner = angular_representation(core_xs, core_ys, cx, cy, num_bins, "max");
    
    std::vector<bool> mask(num_bins);
    for (int i = 0; i < num_bins; ++i) {
        mask[i] = !std::isnan(angular_outer.r_theta[i]);
    }
    
    // Feature 1: Angular support
    features["angular_support"] = compute_angular_support_fraction(angular_outer.r_theta);
    
    // Feature 2: Radial residual energy
    auto residual_info = compute_radial_residual_energy(
        angular_outer.theta_centers, angular_outer.r_theta, a, b, phi, mask);
    features["residual_relative_rmse"] = residual_info.relative_rmse;
    
    // Feature 3: Separation variance
    auto sep_info = compute_separation_stability(angular_outer.r_theta, angular_inner.r_theta);
    features["separation_cv"] = sep_info.cv_separation;
    
    // Feature 4: Largest gap
    auto gap_info = compute_largest_angular_gap(angular_outer.r_theta);
    features["largest_gap_fraction"] = gap_info.largest_gap_fraction;
    
    return features;
}

/**
 * Compute composite ranking score from features.
 */
inline double compute_ranking_score(
    const std::unordered_map<std::string, double>& features,
    double support_weight = 0.35,
    double residual_weight = 0.30,
    double separation_weight = 0.20,
    double gap_weight = 0.15,
    double support_threshold = 0.6,
    double residual_threshold = 0.5,
    double gap_threshold = 0.25
) {
    auto get_feature = [&](const std::string& name, double default_val) {
        auto it = features.find(name);
        if (it != features.end() && !std::isnan(it->second)) {
            return it->second;
        }
        return default_val;
    };
    
    double support = get_feature("angular_support", 0.0);
    double residual = get_feature("residual_relative_rmse", 1.0);
    double sep_cv = get_feature("separation_cv", 1.0);
    double gap_frac = get_feature("largest_gap_fraction", 1.0);
    
    // Hard filters
    if (support < support_threshold) return 0.0;
    if (residual > residual_threshold) return 0.0;
    if (gap_frac > gap_threshold) return 0.0;
    
    // Normalize features to [0, 1] where higher is better
    double support_norm = std::min(std::max(support, 0.0), 1.0);
    double residual_norm = std::min(std::max(1.0 - residual / residual_threshold, 0.0), 1.0);
    double sep_norm = std::isnan(sep_cv) ? 0.5 : std::min(std::max(1.0 - sep_cv, 0.0), 1.0);
    double gap_norm = std::min(std::max(1.0 - gap_frac / gap_threshold, 0.0), 1.0);
    
    // Weighted combination
    double score = support_weight * support_norm + 
                   residual_weight * residual_norm +
                   separation_weight * sep_norm +
                   gap_weight * gap_norm;
    
    return score;
}

/**
 * Filter and rank crater candidates based on computed features.
 * Returns indices of top candidates, sorted by ranking score (descending).
 */
inline std::vector<int> filter_and_rank_candidates(
    const std::vector<std::unordered_map<std::string, double>>& features_list,
    int top_n = 10,
    double support_threshold = 0.6,
    double residual_threshold = 0.5,
    double gap_threshold = 0.25
) {
    std::vector<std::pair<int, double>> scores;
    
    for (size_t i = 0; i < features_list.size(); ++i) {
        double score = compute_ranking_score(
            features_list[i],
            0.35, 0.30, 0.20, 0.15,
            support_threshold, residual_threshold, gap_threshold
        );
        scores.push_back({static_cast<int>(i), score});
    }
    
    // Sort by score descending
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top N indices that pass filters
    std::vector<int> result;
    for (const auto& s : scores) {
        if (s.second > 0 && result.size() < static_cast<size_t>(top_n)) {
            result.push_back(s.first);
        }
    }
    
    return result;
}

}  // namespace Polar

#endif  // POLAR_HPP
