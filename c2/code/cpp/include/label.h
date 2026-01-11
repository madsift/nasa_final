#ifndef LABEL_H
#define LABEL_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <cmath>
#include <climits>
#include <opencv2/opencv.hpp>

/* ============================================================
   Union–Find (for connected components labeling)
   ============================================================ */

struct UnionFind {
    std::vector<int> parent;

    UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[b] = a;
    }
};

/* ============================================================
   label_skimage - Connected components matching skimage.measure.label
   
   Uses Union-Find for exact skimage semantics:
   - connectivity=2 means 8-connectivity (default)
   - connectivity=1 means 4-connectivity
   - background pixels (value 0) are not labeled
   - Returns labeled cv::Mat and number of labels
   ============================================================ */

inline std::pair<cv::Mat, int> label_skimage(
    const cv::Mat& input,
    int connectivity = 2,
    int background = 0
) {
    const int rows = input.rows;
    const int cols = input.cols;
    
    cv::Mat tmp = cv::Mat::zeros(rows, cols, CV_32S);
    UnionFind uf(rows * cols / 2 + 1);

    int next_label = 1;

    // -------- First pass --------
    for (int r = 0; r < rows; ++r) {
        const uchar* in_ptr = input.ptr<uchar>(r);
        int* tmp_ptr = tmp.ptr<int>(r);
        
        for (int c = 0; c < cols; ++c) {
            if (in_ptr[c] == background) continue;

            int best = 0;

            // Check top neighbor
            if (r > 0 && input.at<uchar>(r-1, c) != background) {
                best = tmp.at<int>(r-1, c);
            }

            // Check left neighbor
            if (c > 0 && input.at<uchar>(r, c-1) != background) {
                int l = tmp.at<int>(r, c-1);
                best = (best == 0) ? l : std::min(best, l);
            }

            // 8-connectivity: check diagonal neighbors
            if (connectivity == 2 && r > 0) {
                // Top-left
                if (c > 0 && input.at<uchar>(r-1, c-1) != background) {
                    int d = tmp.at<int>(r-1, c-1);
                    best = (best == 0) ? d : std::min(best, d);
                }
                // Top-right
                if (c < cols-1 && input.at<uchar>(r-1, c+1) != background) {
                    int d = tmp.at<int>(r-1, c+1);
                    best = (best == 0) ? d : std::min(best, d);
                }
            }

            if (best == 0) {
                tmp_ptr[c] = next_label++;
            } else {
                tmp_ptr[c] = best;
                // Unite with all connected neighbors
                if (r > 0 && input.at<uchar>(r-1, c) != background)
                    uf.unite(best, tmp.at<int>(r-1, c));
                if (c > 0 && input.at<uchar>(r, c-1) != background)
                    uf.unite(best, tmp.at<int>(r, c-1));
                if (connectivity == 2 && r > 0) {
                    if (c > 0 && input.at<uchar>(r-1, c-1) != background)
                        uf.unite(best, tmp.at<int>(r-1, c-1));
                    if (c < cols-1 && input.at<uchar>(r-1, c+1) != background)
                        uf.unite(best, tmp.at<int>(r-1, c+1));
                }
            }
        }
    }

    // -------- Second pass: remap labels --------
    cv::Mat output = cv::Mat::zeros(rows, cols, CV_32S);
    std::unordered_map<int, int> remap;
    int out_label = 1;

    for (int r = 0; r < rows; ++r) {
        const int* tmp_ptr = tmp.ptr<int>(r);
        int* out_ptr = output.ptr<int>(r);
        
        for (int c = 0; c < cols; ++c) {
            if (tmp_ptr[c] == 0) continue;
            int root = uf.find(tmp_ptr[c]);
            if (remap.find(root) == remap.end()) {
                remap[root] = out_label++;
            }
            out_ptr[c] = remap[root];
        }
    }

    return {output, out_label - 1};
}


/* ============================================================
   RegionProps - skimage-compatible region properties
   ============================================================ */

struct SkimageRegionProps {
    int label = 0;
    int area = 0;

    // Bounding box (min_r, min_c, max_r, max_c)
    int min_r = INT_MAX, min_c = INT_MAX;
    int max_r = -1, max_c = -1;

    // Centroid
    double sum_r = 0, sum_c = 0;
    double centroid_r = 0, centroid_c = 0;

    // Central moments (for axis computation)
    double mu20 = 0, mu02 = 0, mu11 = 0;

    // Derived properties (computed after moments)
    double major_axis_length = 0;
    double minor_axis_length = 0;
    double eccentricity = 0;

    // Boundary/perimeter
    int boundary_pixels = 0;
    
    // Convex hull area (for solidity)
    double convex_area = 0;
    double solidity = 0;
    
    // Boundary points (for ellipse fitting)
    std::vector<cv::Point> boundary_pts;
    
    // OpenCV-style bounding rect
    cv::Rect bbox() const {
        return cv::Rect(min_c, min_r, max_c - min_c + 1, max_r - min_r + 1);
    }
    
    cv::Point2f centroid() const {
        return cv::Point2f((float)centroid_c, (float)centroid_r);
    }
};

/* ============================================================
   compute_regionprops_skimage
   Computes region properties matching skimage.measure.regionprops
   ============================================================ */

inline std::unordered_map<int, SkimageRegionProps>
compute_regionprops_skimage(const cv::Mat& labels, int num_labels) {
    const int H = labels.rows;
    const int W = labels.cols;
    
    std::unordered_map<int, SkimageRegionProps> props;
    
    // Pre-allocate for all labels
    for (int l = 1; l < num_labels; ++l) {
        props[l].label = l;
    }

    // ---- First pass: area, bbox, centroid sums ----
    for (int r = 0; r < H; ++r) {
        const int* row_ptr = labels.ptr<int>(r);
        for (int c = 0; c < W; ++c) {
            int l = row_ptr[c];
            if (l == 0) continue;

            auto& p = props[l];
            p.area++;

            p.min_r = std::min(p.min_r, r);
            p.min_c = std::min(p.min_c, c);
            p.max_r = std::max(p.max_r, r);
            p.max_c = std::max(p.max_c, c);

            p.sum_r += r;
            p.sum_c += c;
        }
    }

    // Compute centroids
    for (auto& [l, p] : props) {
        if (p.area > 0) {
            p.centroid_r = p.sum_r / p.area;
            p.centroid_c = p.sum_c / p.area;
        }
    }

    // ---- Second pass: moments + boundary detection ----
    // 4-connectivity for boundary detection (matches skimage perimeter)
    const int dr4[4] = {-1, 1, 0, 0};
    const int dc4[4] = {0, 0, -1, 1};

    for (int r = 0; r < H; ++r) {
        const int* row_ptr = labels.ptr<int>(r);
        for (int c = 0; c < W; ++c) {
            int l = row_ptr[c];
            if (l == 0) continue;

            auto& p = props[l];
            double dr = r - p.centroid_r;
            double dc = c - p.centroid_c;

            p.mu20 += dr * dr;
            p.mu02 += dc * dc;
            p.mu11 += dr * dc;

            // Check if boundary pixel
            bool is_boundary = false;
            for (int k = 0; k < 4; ++k) {
                int nr = r + dr4[k], nc = c + dc4[k];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W ||
                    labels.at<int>(nr, nc) != l) {
                    is_boundary = true;
                    break;
                }
            }
            if (is_boundary) {
                p.boundary_pixels++;
                p.boundary_pts.emplace_back(c, r);  // (x, y) = (col, row)
            }
        }
    }

    // ---- Compute derived properties ----
    for (auto& [l, p] : props) {
        if (p.area < 1) continue;
        
        // Normalize moments
        double mu20 = p.mu20 / p.area;
        double mu02 = p.mu02 / p.area;
        double mu11 = p.mu11 / p.area;

        // Eigenvalues of inertia tensor -> axis lengths
        // Matches skimage: major/minor = 4 * sqrt(eigenvalue)
        double common = std::sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11);
        double lambda1 = (mu20 + mu02 + common) / 2.0;
        double lambda2 = (mu20 + mu02 - common) / 2.0;

        p.major_axis_length = 4.0 * std::sqrt(std::max(0.0, lambda1));
        p.minor_axis_length = 4.0 * std::sqrt(std::max(0.0, lambda2));

        // Eccentricity
        if (p.major_axis_length > 1e-6) {
            double ratio = p.minor_axis_length / p.major_axis_length;
            p.eccentricity = std::sqrt(1.0 - ratio * ratio);
        }

        // Solidity = area / convex_hull_area
        if (p.boundary_pts.size() >= 3) {
            std::vector<cv::Point> hull;
            cv::convexHull(p.boundary_pts, hull);
            p.convex_area = cv::contourArea(hull);
            p.solidity = (p.convex_area > 0) ? p.area / p.convex_area : 0.0;
        }
    }

    return props;
}

/* ============================================================
   Circularity metric: 4π * area / perimeter²
   ============================================================ */

inline double circularity(const SkimageRegionProps& p) {
    if (p.boundary_pixels == 0) return 0.0;
    return 4.0 * M_PI * p.area / (p.boundary_pixels * p.boundary_pixels);
}

#endif // LABEL_H
