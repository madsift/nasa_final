#include "watershed_static.h"
#include <queue>
#include <map>
#include <iostream>
#include <cmath>
#include <algorithm>

// ============================================================================
// STATIC MEMORY CONFIGURATION
// ============================================================================
// NOTE: Static buffers were REMOVED as analysis showed they were allocated
// but never used. The watershed function creates local buffers.
// The init function is kept as a no-op for API compatibility.

// Initialize static buffers (NO-OP - kept for API compatibility)
void init_watershed_static_buffers(int max_h, int max_w) {
    // Previously allocated ~100MB of cv::Mat buffers here that were never used.
    // Removed to save memory. The watershed_skimage function creates local buffers.
    (void)max_h;
    (void)max_w;
}

// Internal struct for watershed priority queue
// Matches skimage watershed: primary=value, secondary=time-of-entry (age)
struct WatershedPixel {
    float val;
    uint64_t age;  // Time of entry - lower = earlier = higher priority
    int r, c;
    int label;
    
    bool operator>(const WatershedPixel& other) const {
        if (val != other.val) return val > other.val;
        return age > other.age;  // Earlier entries have priority
    }
};


void watershed_skimage(const cv::Mat& image, cv::Mat& markers, const cv::Mat& mask) {
    /*
     * This implements the skimage watershed algorithm:
     * - Priority queue based flooding from marker seeds
     * - Floods in order of image intensity (ascending for -distance)
     * - Uses 8-connectivity
     * - mask defines the valid region to flood
     */
    const int rows = image.rows;
    const int cols = image.cols;
    
    // Track pixel states: 0=unvisited, 1=in_queue, 2=processed
    cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8U);
    
    // Min-heap: process lowest values first (like -distance ascending)
    std::priority_queue<WatershedPixel, std::vector<WatershedPixel>, std::greater<WatershedPixel>> pq;
    
    // Age counter for tie-breaking (matches skimage "time of entry")
    uint64_t age_counter = 0;
    
    // 8-connectivity (matches skimage default)
    constexpr int kNumNeighbors = 8;
    constexpr int dr[8] = {-1, -1, -1,  0,  0,  1, 1, 1};
    constexpr int dc[8] = {-1,  0,  1, -1,  1, -1, 0, 1};
    
    // Initialize: seed pixels with labels > 0 are starting points
    for (int r = 0; r < rows; ++r) {
        const int* mk_ptr = markers.ptr<int>(r);
        const uchar* mask_ptr = mask.ptr<uchar>(r);
        uchar* vis_ptr = visited.ptr<uchar>(r);
        
        for (int c = 0; c < cols; ++c) {
            if (mask_ptr[c] == 0) continue;
            
            if (mk_ptr[c] > 0) {
                vis_ptr[c] = 2; // Seed is already processed
                
                // Add unlabeled neighbors within mask to queue
                for (int i = 0; i < kNumNeighbors; ++i) {
                    const int nr = r + dr[i];
                    const int nc = c + dc[i];
                    
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                        if (markers.at<int>(nr, nc) == 0 && 
                            mask.at<uchar>(nr, nc) > 0 && 
                            visited.at<uchar>(nr, nc) == 0) {
                            const float val = image.at<float>(nr, nc);
                            pq.push({val, age_counter++, nr, nc, mk_ptr[c]});
                            visited.at<uchar>(nr, nc) = 1;
                        }
                    }
                }
            }
        }
    }
    
    // Flood from seeds with PLATEAU HANDLING (matches skimage behavior)
    // Key insight: Process all pixels with same value as a batch before expanding neighbors
    while (!pq.empty()) {
        const float current_val = pq.top().val;
        
        // 1) Collect all pixels with same value (the plateau)
        std::vector<WatershedPixel> plateau;
        while (!pq.empty() && pq.top().val == current_val) {
            plateau.push_back(pq.top());
            pq.pop();
        }
        
        // 2) Assign labels to plateau pixels (without pushing neighbors yet)
        for (const auto& curr : plateau) {
            const int r = curr.r;
            const int c = curr.c;
            
            // Skip if already labeled
            if (markers.at<int>(r, c) != 0) continue;
            
            uchar& vis = visited.at<uchar>(r, c);
            if (vis == 2) continue;
            
            // Assign label
            markers.at<int>(r, c) = curr.label;
            vis = 2;
        }
        
        // 3) Expand neighbors AFTER entire plateau is resolved
        for (const auto& curr : plateau) {
            const int r = curr.r;
            const int c = curr.c;
            
            // Only expand from pixels we actually labeled
            if (markers.at<int>(r, c) != curr.label) continue;
            
            for (int i = 0; i < kNumNeighbors; ++i) {
                const int nr = r + dr[i];
                const int nc = c + dc[i];
                
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    if (markers.at<int>(nr, nc) == 0 && 
                        mask.at<uchar>(nr, nc) > 0 && 
                        visited.at<uchar>(nr, nc) == 0) {
                        const float val = image.at<float>(nr, nc);
                        pq.push({val, age_counter++, nr, nc, curr.label});
                        visited.at<uchar>(nr, nc) = 1;
                    }
                }
            }
        }
    }
}

// ==========================================
// REGION PROPERTIES (matches skimage.measure.regionprops)
// ==========================================


std::vector<RegionProps> compute_regionprops(const cv::Mat& labels, int num_labels, int min_area) {
    const int rows = labels.rows;
    const int cols = labels.cols;
    
    // Collect pixels per label
    std::vector<std::vector<cv::Point>> label_pixels(num_labels + 1);
    
    for (int r = 0; r < rows; ++r) {
        const int* row_ptr = labels.ptr<int>(r);
        for (int c = 0; c < cols; ++c) {
            int lbl = row_ptr[c];
            if (lbl > 0 && lbl <= num_labels) {
                label_pixels[lbl].emplace_back(c, r);  // Point(x, y) = (col, row)
            }
        }
    }
    
    std::vector<RegionProps> regions;
    
    for (int lbl = 1; lbl <= num_labels; ++lbl) {
        auto& pixels = label_pixels[lbl];
        if (static_cast<int>(pixels.size()) < min_area) continue;
        
        RegionProps prop;
        prop.label = lbl;
        prop.pixels = std::move(pixels);
        prop.area = static_cast<double>(prop.pixels.size());
        
        // Bounding box
        prop.bbox = cv::boundingRect(prop.pixels);
        
        // Centroid
        double sum_x = 0, sum_y = 0;
        for (const auto& p : prop.pixels) {
            sum_x += p.x;
            sum_y += p.y;
        }
        prop.centroid = cv::Point2f(static_cast<float>(sum_x / prop.area), 
                                     static_cast<float>(sum_y / prop.area));
        
        // Create binary image (local mask) - use 255 for compatibility with OpenCV ops
        prop.image = cv::Mat::zeros(prop.bbox.height, prop.bbox.width, CV_8U);
        for (const auto& p : prop.pixels) {
            prop.image.at<uchar>(p.y - prop.bbox.y, p.x - prop.bbox.x) = 255;
        }
        
        // Compute moments for axis lengths and eccentricity
        cv::Moments m = cv::moments(prop.image, true);
        
        // Central moments (normalized)
        double mu20 = m.mu20 / m.m00;
        double mu02 = m.mu02 / m.m00;
        double mu11 = m.mu11 / m.m00;
        
        // Eigenvalues of inertia tensor give axis lengths
        double common = std::sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11);
        double lambda1 = (mu20 + mu02 + common) / 2.0;
        double lambda2 = (mu20 + mu02 - common) / 2.0;
        
        // Major/minor axis: 4 * sqrt(eigenvalue)
        prop.major_axis_length = 4.0 * std::sqrt(std::max(0.0, lambda1));
        prop.minor_axis_length = 4.0 * std::sqrt(std::max(0.0, lambda2));
        
        // Eccentricity
        if (prop.major_axis_length > 1e-6) {
            double ratio = prop.minor_axis_length / prop.major_axis_length;
            prop.eccentricity = std::sqrt(1.0 - ratio * ratio);
        } else {
            prop.eccentricity = 0.0;
        }
        
        // Solidity = area / convex_hull_area
        std::vector<cv::Point> hull;
        cv::convexHull(prop.pixels, hull);
        double hull_area = cv::contourArea(hull);
        prop.solidity = (hull_area > 0) ? prop.area / hull_area : 0.0;
        
        regions.push_back(std::move(prop));
    }
    
    return regions;
}

// ==========================================
// OUTER RIM POINTS (matches Python outer_rim_points)
// Uses distance transform: rim = (dist > 0) & (dist <= thickness)
// ==========================================

std::vector<cv::Point> outer_rim_points(const cv::Mat& binary_mask, int thickness) {
    cv::Mat dist;
    cv::distanceTransform(binary_mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    
    std::vector<cv::Point> rim_pts;
    
    for (int r = 0; r < dist.rows; ++r) {
        const float* dist_ptr = dist.ptr<float>(r);
        for (int c = 0; c < dist.cols; ++c) {
            float d = dist_ptr[c];
            if (d > 0 && d <= thickness) {
                rim_pts.emplace_back(c, r);  // (x, y) = (col, row)
            }
        }
    }
    
    return rim_pts;
}

// ==========================================
// ANGULAR SAMPLE (matches Python angular_sample)
// Select one point per angular bin (the one with max radius)
// ==========================================

std::vector<cv::Point2f> angular_sample(const std::vector<cv::Point>& points, 
                                         cv::Point2f center, int bins) {
    if (points.empty()) return {};
    
    std::map<int, std::pair<cv::Point2f, float>> selected;  // bin -> (point, radius)
    
    float cx = center.x;
    float cy = center.y;
    
    for (const auto& p : points) {
        float dx = p.x - cx;
        float dy = p.y - cy;
        float angle = std::atan2(dy, dx);  // [-pi, pi]
        float radius = std::sqrt(dx * dx + dy * dy);
        
        // Map angle to bin: (angle + pi) / (2*pi) * bins
        int bin_id = static_cast<int>((angle + CV_PI) / (2.0 * CV_PI) * bins);
        bin_id = std::clamp(bin_id, 0, bins - 1);
        
        auto it = selected.find(bin_id);
        if (it == selected.end() || radius > it->second.second) {
            selected[bin_id] = {cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)), radius};
        }
    }
    
    std::vector<cv::Point2f> result;
    result.reserve(selected.size());
    for (const auto& kv : selected) {
        result.push_back(kv.second.first);
    }
    
    return result;
}

// ==========================================
// GEOMETRIC HELPERS (match Python)
// ==========================================

inline bool valid_ellipse(float w, float h) {
    if (w < 5.0f || h < 5.0f) return false;
    float ratio = std::max(w, h) / (std::min(w, h) + 1e-6f);
    return ratio < 4.0f;
}

std::pair<float, float> enforce_eccentricity_floor(float w, float h, float min_e) {
    float a = std::max(w, h) / 2.0f;
    float b = std::min(w, h) / 2.0f;
    
    if (a < 1e-5f) return {w, h};
    
    float e = std::sqrt(1.0f - (b * b) / (a * a));
    
    if (e >= min_e) return {w, h};
    
    // Stretch minor axis
    float target_b = a * std::sqrt(1.0f - min_e * min_e);
    float scale = target_b / std::max(b, 1e-6f);
    
    // Python: returns w, h * scale (stretches h if h is minor)
    if (w < h) {
        return {w * scale, h};
    } else {
        return {w, h * scale};
    }
}

// ==========================================
// FIND OPTIMAL SCALE (matches Python find_optimal_scale_from_rim)
// ==========================================

std::pair<float, float> find_optimal_scale_from_rim(
    float cx, float cy, float w_base, float h_base, float angle,
    const cv::Mat& rim_prob,
    float scale_min, float scale_max, int steps
) {
    const int H = rim_prob.rows;
    const int W = rim_prob.cols;
    
    constexpr int kNumSamples = 72;
    const float angle_step = 2.0f * CV_PI / kNumSamples;
    
    float cos_ang = std::cos(angle * CV_PI / 180.0f);
    float sin_ang = std::sin(angle * CV_PI / 180.0f);
    
    float a_base = w_base / 2.0f;
    float b_base = h_base / 2.0f;
    
    float best_scale = 1.14f;  // Fallback default
    float best_prob = 0.0f;
    
    for (int s = 0; s < steps; ++s) {
        float scale = scale_min + s * (scale_max - scale_min) / std::max(1, steps - 1);
        float a = a_base * scale;
        float b = b_base * scale;
        
        float sum_prob = 0.0f;
        int valid_count = 0;
        
        for (int i = 0; i < kNumSamples; ++i) {
            float theta = i * angle_step;
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);
            
            float x_p = cx + a * cos_theta * cos_ang - b * sin_theta * sin_ang;
            float y_p = cy + a * cos_theta * sin_ang + b * sin_theta * cos_ang;
            
            int ix = static_cast<int>(x_p);
            int iy = static_cast<int>(y_p);
            
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                sum_prob += rim_prob.at<float>(iy, ix);
                valid_count++;
            }
        }
        
        // Python: if np.sum(valid) < 36: continue (at least half the points)
        if (valid_count >= kNumSamples / 2) {
            float avg_prob = sum_prob / valid_count;
            if (avg_prob > best_prob) {
                best_prob = avg_prob;
                best_scale = scale;
            }
        }
    }
    
    return {best_scale, best_prob};
}

// ==========================================
// COMPUTE ELLIPSE CONFIDENCE (matches Python compute_ellipse_confidence)
// ==========================================

std::pair<float, ConfidenceDetails> compute_ellipse_confidence(
    float cx, float cy, float a, float b, float angle,
    const cv::Mat& rim_prob, const cv::Mat& global_prob,
    int n_samples
) {
    const int H = rim_prob.rows;
    const int W = rim_prob.cols;
    
    float cos_ang = std::cos(angle * CV_PI / 180.0f);
    float sin_ang = std::sin(angle * CV_PI / 180.0f);
    
    const float angle_step = 2.0f * CV_PI / n_samples;
    
    // Sample perimeter
    std::vector<float> rim_probs;
    rim_probs.reserve(n_samples);
    int n_valid = 0;
    
    for (int i = 0; i < n_samples; ++i) {
        float theta = i * angle_step;
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);
        
        float x_p = cx + a * cos_theta * cos_ang - b * sin_theta * sin_ang;
        float y_p = cy + a * cos_theta * sin_ang + b * sin_theta * cos_ang;
        
        int ix = static_cast<int>(x_p);
        int iy = static_cast<int>(y_p);
        
        if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
            rim_probs.push_back(rim_prob.at<float>(iy, ix));
            n_valid++;
        }
    }
    
    if (n_valid < n_samples / 2) {
        return {0.0f, {0.0f, 0.0f, 0.0f}};
    }
    
    // 1. Average rim probability
    float rim_score = 0.0f;
    for (float p : rim_probs) rim_score += p;
    rim_score /= n_valid;
    
    // 2. Angular completeness: fraction with rim_prob > 0.2
    int high_count = 0;
    for (float p : rim_probs) {
        if (p > 0.2f) high_count++;
    }
    float angular_completeness = static_cast<float>(high_count) / n_valid;
    
    // 3. Global coverage: sample interior points (matching Python thetas[::3])
    float inner_scales[] = {0.3f, 0.5f, 0.7f};
    int global_hits = 0;
    int global_total = 0;
    
    for (float inner_scale : inner_scales) {
        float a_scaled = a * inner_scale;
        float b_scaled = b * inner_scale;
        
        // Python: thetas[::3] means every 3rd theta
        for (int i = 0; i < n_samples; i += 3) {
            float theta = i * angle_step;
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);
            
            float x_i = cx + a_scaled * cos_theta * cos_ang - b_scaled * sin_theta * sin_ang;
            float y_i = cy + a_scaled * cos_theta * sin_ang + b_scaled * sin_theta * cos_ang;
            
            int ix = static_cast<int>(x_i);
            int iy = static_cast<int>(y_i);
            
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                if (global_prob.at<float>(iy, ix) > 0.5f) {
                    global_hits++;
                }
                global_total++;
            }
        }
    }
    
    float global_coverage = (global_total > 0) ? static_cast<float>(global_hits) / global_total : 0.0f;
    
    // Combined confidence (matches Python weights)
    float confidence = 0.5f * rim_score + 0.3f * angular_completeness + 0.2f * global_coverage;
    
    return {confidence, {rim_score, angular_completeness, global_coverage}};
}

// ==========================================
// MAIN EXTRACTION (matches Python extract_craters_cv2_adaptive_selection_rescue)
// ==========================================

std::vector<Crater> extract_craters_cv2_adaptive_selection_rescue(
    const std::vector<cv::Mat>& pred_tensor,
    float threshold,
    float rim_thresh,
    float ecc_floor,
    float scale_min,
    float scale_max,
    int scale_steps,
    float min_confidence,
    float min_confidence_min,
    bool nms_filter,
    bool enable_rescue,
    bool enable_completeness,
    int min_semi_axis,
    float max_size_ratio,
    bool require_fully_visible,
    cv::Size image_shape
) {
    const cv::Mat& core_prob = pred_tensor[0];
    const cv::Mat& global_prob = pred_tensor[1];
    const cv::Mat& rim_prob = pred_tensor[2];
    const int H = core_prob.rows;
    const int W = core_prob.cols;

    // Binary masks
    cv::Mat mask_core, mask_global;
    cv::threshold(core_prob, mask_core, threshold, 1, cv::THRESH_BINARY);
    cv::threshold(global_prob, mask_global, threshold, 1, cv::THRESH_BINARY);
    mask_core.convertTo(mask_core, CV_8U);
    mask_global.convertTo(mask_global, CV_8U);

    // Connected components for markers (matches skimage.measure.label)
    cv::Mat markers;
    int num_labels = cv::connectedComponents(mask_core, markers, 8, CV_32S);

    // Distance transform
    cv::Mat distance;
    cv::distanceTransform(mask_global, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    
    // Negate for watershed (matches Python: -distance)
    cv::Mat neg_distance;
    cv::subtract(cv::Scalar(0.0f), distance, neg_distance);

    // Watershed
    cv::Mat labels = markers.clone();
    watershed_skimage(neg_distance, labels, mask_global);

    // Compute region properties (matches skimage.measure.regionprops)
    auto regions = compute_regionprops(labels, num_labels);

    std::vector<Crater> candidates;
    candidates.reserve(regions.size());

    for (const auto& prop : regions) {
        // Python filters:
        // if prop.area < 40: continue
        if (prop.area < 40) continue;
        // if (prop.area > 100000) continue;
        // if prop.minor_axis_length < max(3, 0.02 * np.sqrt(prop.area)): continue
        double min_axis_thresh = std::max(3.0, 0.02 * std::sqrt(prop.area));
        if (prop.minor_axis_length < min_axis_thresh) continue;
        
        // if prop.eccentricity > 0.95 and prop.area < 200: continue
        if (prop.eccentricity > 0.95 && prop.area < 200) continue;
        
        // if prop.solidity < 0.7 and prop.area < 150: continue
        if (prop.solidity < 0.7 && prop.area < 150) continue;

        // Get outer rim points with adaptive thickness
        int thickness = std::max(2, static_cast<int>(0.03 * std::sqrt(prop.area)));
        auto local_rim_pts = outer_rim_points(prop.image, thickness);

        if (local_rim_pts.size() < 20) continue;

        // Map to global coordinates
        std::vector<cv::Point> global_rim_pts;
        global_rim_pts.reserve(local_rim_pts.size());
        for (const auto& p : local_rim_pts) {
            global_rim_pts.emplace_back(p.x + prop.bbox.x, p.y + prop.bbox.y);
        }

        // Filter by rim probability
        std::vector<cv::Point> filtered_rim_pts;
        filtered_rim_pts.reserve(global_rim_pts.size());
        for (const auto& p : global_rim_pts) {
            if (rim_prob.at<float>(p.y, p.x) > rim_thresh) {
                filtered_rim_pts.push_back(p);
            }
        }

        if (filtered_rim_pts.size() < 20) continue;

        // Angular sampling
        cv::Point2f center(prop.centroid.x, prop.centroid.y);
        auto sampled_pts = angular_sample(filtered_rim_pts, center, 64);

        if (sampled_pts.size() < 16) continue;

        // Fit ellipse
        std::vector<cv::Point2f> pts_for_fit(sampled_pts.begin(), sampled_pts.end());
        float cx, cy, w, h, angle;
        
        try {
            cv::RotatedRect fit = cv::fitEllipse(pts_for_fit);
            cx = fit.center.x;
            cy = fit.center.y;
            w = fit.size.width;
            h = fit.size.height;
            angle = fit.angle;
        } catch (...) {
            // Fallback to bbox
            cx = prop.centroid.x;
            cy = prop.centroid.y;
            w = static_cast<float>(prop.bbox.width);
            h = static_cast<float>(prop.bbox.height);
            angle = 0.0f;
        }

        if (!valid_ellipse(w, h)) {
            w = static_cast<float>(prop.bbox.width);
            h = static_cast<float>(prop.bbox.height);
        }

        // Find optimal scale
        auto [opt_scale, rim_conf] = find_optimal_scale_from_rim(
            cx, cy, w, h, angle, rim_prob, scale_min, scale_max, scale_steps
        );

        float w_scaled = w * opt_scale;
        float h_scaled = h * opt_scale;
        
        // Enforce eccentricity floor
        auto [w_final, h_final] = enforce_eccentricity_floor(w_scaled, h_scaled, ecc_floor);

        // Compute confidence
        auto [confidence, details] = compute_ellipse_confidence(
            cx, cy, w_final / 2.0f, h_final / 2.0f, angle, rim_prob, global_prob
        );

        // Angular completeness filter
        if (details.angular_completeness < 0.25f) continue;

        // Rescue confidence
        float confidence_rescue = confidence;
        
        if (enable_rescue) {
            float radius = (w_final + h_final) / 4.0f;
            if (radius > 15.0f && details.global_coverage > 0.6f) {
                confidence_rescue = std::max(confidence_rescue, details.global_coverage * 0.9f);
            }
        }
        
        if (enable_completeness) {
            if (details.angular_completeness > 0.7f) {
                confidence_rescue = std::max(confidence_rescue, 
                    (confidence + details.angular_completeness) / 2.0f);
            }
        }

        Crater cand;
        cand.x = cx;
        cand.y = cy;
        cand.a = w_final / 2.0f;
        cand.b = h_final / 2.0f;
        cand.angle = angle;
        cand.confidence = confidence;
        cand.confidence_rescue = confidence_rescue;
        cand.scale = opt_scale;
        cand.rim_score = details.rim_score;
        cand.angular_completeness = details.angular_completeness;
        cand.global_coverage = details.global_coverage;

        candidates.push_back(cand);
    }

    // Sort by confidence (descending)
    std::sort(candidates.begin(), candidates.end(), [](const Crater& a, const Crater& b) {
        return a.confidence > b.confidence;
    });

    // Size/visibility filter
    if (min_semi_axis > 0 || max_size_ratio < 1.0f || require_fully_visible) {
        int h_img = (image_shape.height > 0) ? image_shape.height : H;
        int w_img = (image_shape.width > 0) ? image_shape.width : W;
        int S = std::min(w_img, h_img);
        
        std::vector<Crater> filtered;
        filtered.reserve(candidates.size());
        
        for (const auto& c : candidates) {
            // Min semi-minor axis
            float semi_minor = std::min(c.a, c.b);
            if (semi_minor < min_semi_axis) continue;
            
            // Calculate bounding box
            cv::RotatedRect rect(cv::Point2f(c.x, c.y), cv::Size2f(c.a * 2, c.b * 2), c.angle);
            cv::Point2f box_pts[4];
            rect.points(box_pts);
            
            float x_min = std::min({box_pts[0].x, box_pts[1].x, box_pts[2].x, box_pts[3].x});
            float y_min = std::min({box_pts[0].y, box_pts[1].y, box_pts[2].y, box_pts[3].y});
            float x_max = std::max({box_pts[0].x, box_pts[1].x, box_pts[2].x, box_pts[3].x});
            float y_max = std::max({box_pts[0].y, box_pts[1].y, box_pts[2].y, box_pts[3].y});
            
            float bbox_w = x_max - x_min;
            float bbox_h = y_max - y_min;
            
            // Max size ratio
            if (max_size_ratio < 1.0f) {
                if ((bbox_w + bbox_h) >= (max_size_ratio * S)) continue;
            }
            
            // Fully visible
            if (require_fully_visible) {
                if (x_min < 0 || y_min < 0 || x_max > w_img || y_max > h_img) continue;
            }
            
            filtered.push_back(c);
        }
        
        candidates = std::move(filtered);
    }

    // NMS filter
    if (nms_filter) {
        std::vector<Crater> accepted;
        accepted.reserve(candidates.size());
        
        for (const auto& cand : candidates) {
            bool is_bad = false;
            float r_cand = std::sqrt(cand.a * cand.b);
            
            for (const auto& kept : accepted) {
                float dx = cand.x - kept.x;
                float dy = cand.y - kept.y;
                float dist = std::sqrt(dx * dx + dy * dy);
                float r_kept = std::sqrt(kept.a * kept.b);
                
                // Check nested
                if (dist < std::abs(r_cand - r_kept)) {
                    is_bad = true;
                    break;
                }
                
                // Check overlap
                if (dist < (r_cand + r_kept) * 0.80f) {
                    is_bad = true;
                    break;
                }
            }
            
            if (!is_bad) {
                accepted.push_back(cand);
            }
        }
        
        candidates = std::move(accepted);
    }

    // Selection: lower threshold until we get at least 10 craters
    float step = 0.01f;
    float current_thresh = min_confidence;
    
    std::vector<Crater> final_predictions;
    
    while (current_thresh >= min_confidence_min - 1e-6f) {
        final_predictions.clear();
        for (const auto& c : candidates) {
            if (c.confidence >= current_thresh) {
                final_predictions.push_back(c);
            }
        }
        
        if (final_predictions.size() >= 10) break;
        current_thresh -= step;
    }

    // Rescue fallback
    if (final_predictions.size() < 10 && (enable_rescue || enable_completeness)) {
        // Get candidates not in final list
        std::vector<Crater> remaining;
        for (const auto& c : candidates) {
            bool in_final = false;
            for (const auto& f : final_predictions) {
                if (std::abs(c.x - f.x) < 1e-3f && std::abs(c.y - f.y) < 1e-3f) {
                    in_final = true;
                    break;
                }
            }
            if (!in_final) {
                remaining.push_back(c);
            }
        }
        
        // Sort by rescue confidence
        std::sort(remaining.begin(), remaining.end(), [](const Crater& a, const Crater& b) {
            return a.confidence_rescue > b.confidence_rescue;
        });
        
        for (const auto& cand : remaining) {
            if (final_predictions.size() >= 10) break;
            
            if (cand.confidence_rescue >= 0.85f) {
                final_predictions.push_back(cand);
            }
        }
    }

    return final_predictions;
}