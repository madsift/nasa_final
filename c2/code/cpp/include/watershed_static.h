#ifndef WATERSHED_STATIC_H
#define WATERSHED_STATIC_H

#include <opencv2/opencv.hpp>
#include <vector>

// ==========================================
// DATA STRUCTURES
// ==========================================

struct Crater {
    float x, y;
    float a, b;
    float angle;
    float confidence;
    float confidence_rescue;
    float scale;
    float rim_score;
    float angular_completeness;
    float global_coverage;
    float ranker_score = 0.0f;  // LightGBM ranker probability
    bool matched = false;  // For scoring
    
    // Alias for angle (used by ranking features)
    float theta() const { return angle; }
    
    cv::Rect2f getBBox() const {
        float rad = angle * CV_PI / 180.0f;
        float cos_t = std::cos(rad);
        float sin_t = std::sin(rad);
        float ux = std::sqrt(a * a * cos_t * cos_t + b * b * sin_t * sin_t);
        float uy = std::sqrt(a * a * sin_t * sin_t + b * b * cos_t * cos_t);
        return cv::Rect2f(x - ux, y - uy, 2 * ux, 2 * uy);
    }
};


struct RegionProps {
    int label;
    std::vector<cv::Point> pixels;
    cv::Rect bbox;          // (x, y, width, height)
    cv::Point2f centroid;   // (x, y) = (col, row)
    double area;
    double minor_axis_length;
    double major_axis_length;
    double eccentricity;
    double solidity;
    cv::Mat image;          // Boolean mask of region (local coords)
};

struct ConfidenceDetails {
    float rim_score;
    float angular_completeness;
    float global_coverage;
};

// ==========================================
// STATIC MEMORY INITIALIZATION
// ==========================================
void init_watershed_static_buffers(int max_h, int max_w);

// ==========================================
// FUNCTION DECLARATIONS
// ==========================================

// Watershed (matches skimage.segmentation.watershed)
void watershed_skimage(const cv::Mat& image, cv::Mat& markers, const cv::Mat& mask);

// Region properties (matches skimage.measure.regionprops)
std::vector<RegionProps> compute_regionprops(const cv::Mat& labels, int num_labels, int min_area = 1);

// Outer rim points (matches Python outer_rim_points)
std::vector<cv::Point> outer_rim_points(const cv::Mat& binary_mask, int thickness = 1);

// Angular sampling (matches Python angular_sample)
std::vector<cv::Point2f> angular_sample(const std::vector<cv::Point>& points, 
                                         cv::Point2f center, int bins = 64);

// Geometric helpers
inline bool valid_ellipse(float w, float h);
std::pair<float, float> enforce_eccentricity_floor(float w, float h, float min_e = 0.15f);

// Scale optimization
std::pair<float, float> find_optimal_scale_from_rim(
    float cx, float cy, float w_base, float h_base, float angle,
    const cv::Mat& rim_prob,
    float scale_min = 1.0f, float scale_max = 1.35f, int steps = 15
);

// Confidence computation
std::pair<float, ConfidenceDetails> compute_ellipse_confidence(
    float cx, float cy, float a, float b, float angle,
    const cv::Mat& rim_prob, const cv::Mat& global_prob,
    int n_samples = 120
);

// Main extraction function
std::vector<Crater> extract_craters_cv2_adaptive_selection_rescue(
    const std::vector<cv::Mat>& pred_tensor,
    float threshold = 0.85f,
    float rim_thresh = 0.25f,
    float ecc_floor = 0.15f,
    float scale_min = 1.0f,
    float scale_max = 1.4f,
    int scale_steps = 30,
    float min_confidence = 0.97f,
    float min_confidence_min = 0.90f,
    bool nms_filter = false,
    bool enable_rescue = false,
    bool enable_completeness = false,
    int min_semi_axis = 0,
    float max_size_ratio = 1.0f,
    bool require_fully_visible = false,
    cv::Size image_shape = cv::Size(0, 0)
);

#endif // WATERSHED_STATIC_H
