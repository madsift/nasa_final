#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <queue>
#include <cmath>
#include <climits>

/* ============================================================
   Union–Find
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
   skimage.measure.label (exact semantics)
   ============================================================ */

std::pair<std::vector<int>, int> label(
    const std::vector<int>& input,
    int rows,
    int cols,
    int connectivity = 2,
    int background = 0
) {
    if ((int)input.size() != rows * cols)
        throw std::invalid_argument("Input size mismatch");

    std::vector<int> tmp(rows * cols, 0);
    UnionFind uf(rows * cols / 2 + 1);

    int next_label = 1;
    auto idx = [&](int r, int c) { return r * cols + c; };

    // -------- First pass --------
    for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
        int i = idx(r,c);
        if (input[i] == background) continue;

        int best = 0;

        if (r > 0 && input[idx(r-1,c)] != background)
            best = tmp[idx(r-1,c)];

        if (c > 0 && input[idx(r,c-1)] != background) {
            int l = tmp[idx(r,c-1)];
            best = (best == 0) ? l : std::min(best, l);
        }

        if (connectivity == 2 && r > 0) {
            if (c > 0 && input[idx(r-1,c-1)] != background) {
                int d = tmp[idx(r-1,c-1)];
                best = (best == 0) ? d : std::min(best, d);
            }
            if (c < cols-1 && input[idx(r-1,c+1)] != background) {
                int d = tmp[idx(r-1,c+1)];
                best = (best == 0) ? d : std::min(best, d);
            }
        }

        if (best == 0) {
            tmp[i] = next_label++;
        } else {
            tmp[i] = best;
            if (r > 0 && input[idx(r-1,c)] != background)
                uf.unite(best, tmp[idx(r-1,c)]);
            if (c > 0 && input[idx(r,c-1)] != background)
                uf.unite(best, tmp[idx(r,c-1)]);
            if (connectivity == 2 && r > 0) {
                if (c > 0 && input[idx(r-1,c-1)] != background)
                    uf.unite(best, tmp[idx(r-1,c-1)]);
                if (c < cols-1 && input[idx(r-1,c+1)] != background)
                    uf.unite(best, tmp[idx(r-1,c+1)]);
            }
        }
    }

    // -------- Second pass --------
    std::vector<int> output(rows * cols, 0);
    std::unordered_map<int,int> remap;
    int out_label = 1;

    for (int i = 0; i < rows * cols; ++i) {
        if (tmp[i] == 0) continue;
        int root = uf.find(tmp[i]);
        if (!remap.count(root))
            remap[root] = out_label++;
        output[i] = remap[root];
    }

    return {output, out_label - 1};
}

/* ============================================================
   RegionProps
   ============================================================ */

struct RegionProps {
    int label = 0;
    int area = 0;

    int min_r = INT_MAX, min_c = INT_MAX;
    int max_r = -1, max_c = -1;

    double sum_r = 0, sum_c = 0;
    double centroid_r = 0, centroid_c = 0;

    double mu20 = 0, mu02 = 0, mu11 = 0;

    int boundary_pixels = 0;
};

/* ============================================================
   regionprops computation
   ============================================================ */

std::unordered_map<int, RegionProps>
compute_regionprops(
    const std::vector<int>& labels,
    int H, int W
) {
    std::unordered_map<int, RegionProps> props;
    auto idx = [&](int r, int c) { return r * W + c; };

    // ---- area, bbox, centroid sums ----
    for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
        int l = labels[idx(r,c)];
        if (l == 0) continue;

        auto& p = props[l];
        p.label = l;
        p.area++;

        p.min_r = std::min(p.min_r, r);
        p.min_c = std::min(p.min_c, c);
        p.max_r = std::max(p.max_r, r);
        p.max_c = std::max(p.max_c, c);

        p.sum_r += r;
        p.sum_c += c;
    }

    for (auto& [l,p] : props) {
        p.centroid_r = p.sum_r / p.area;
        p.centroid_c = p.sum_c / p.area;
    }

    // ---- moments + perimeter ----
    const int dr4[4] = {-1,1,0,0};
    const int dc4[4] = {0,0,-1,1};

    for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
        int l = labels[idx(r,c)];
        if (l == 0) continue;

        auto& p = props[l];
        double dr = r - p.centroid_r;
        double dc = c - p.centroid_c;

        p.mu20 += dr*dr;
        p.mu02 += dc*dc;
        p.mu11 += dr*dc;

        bool boundary = false;
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr4[k], nc = c + dc4[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W ||
                labels[idx(nr,nc)] != l) {
                boundary = true;
                break;
            }
        }
        if (boundary) p.boundary_pixels++;
    }

    return props;
}

/* ============================================================
   Rim thickness (distance based)
   ============================================================ */

double rim_thickness(
    const std::vector<int>& labels,
    int H, int W,
    int target
) {
    std::vector<int> dist(H * W, -1);
    std::queue<int> q;
    auto idx = [&](int r, int c) { return r * W + c; };

    const int dr4[4] = {-1,1,0,0};
    const int dc4[4] = {0,0,-1,1};

    for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
        if (labels[idx(r,c)] != target) continue;

        bool boundary = false;
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr4[k], nc = c + dc4[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W ||
                labels[idx(nr,nc)] != target) {
                boundary = true;
                break;
            }
        }

        if (boundary) {
            dist[idx(r,c)] = 0;
            q.push(idx(r,c));
        }
    }

    int maxd = 0;
    while (!q.empty()) {
        int i = q.front(); q.pop();
        int r = i / W, c = i % W;

        for (int k = 0; k < 4; ++k) {
            int nr = r + dr4[k], nc = c + dc4[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int ni = idx(nr,nc);

            if (labels[ni] == target && dist[ni] < 0) {
                dist[ni] = dist[i] + 1;
                maxd = std::max(maxd, dist[ni]);
                q.push(ni);
            }
        }
    }

    return 2.0 * maxd;
}

/* ============================================================
   Shape metrics
   ============================================================ */

double circularity(const RegionProps& p) {
    if (p.boundary_pixels == 0) return 0.0;
    return 4.0 * M_PI * p.area /
           (p.boundary_pixels * p.boundary_pixels);
}

double continuity(const RegionProps& p) {
    return double(p.boundary_pixels) / p.area;
}

/* ============================================================
   Demo
   ============================================================ */

int main() {
    std::vector<std::vector<int>> img = {
        {0,1,1,1,0},
        {1,1,0,1,1},
        {1,0,0,0,1},
        {1,1,0,1,1},
        {0,1,1,1,0}
    };

    int H = img.size(), W = img[0].size();
    std::vector<int> flat(H*W);
    for (int r=0;r<H;++r)
    for (int c=0;c<W;++c)
        flat[r*W+c] = img[r][c];

    auto [labels, n] = label(flat, H, W, 2, 0);
    auto props = compute_regionprops(labels, H, W);

    for (auto& [l,p] : props) {
        std::cout << "Label " << l << "\n";
        std::cout << "  area        = " << p.area << "\n";
        std::cout << "  centroid    = (" << p.centroid_r << ", " << p.centroid_c << ")\n";
        std::cout << "  perimeter   = " << p.boundary_pixels << "\n";
        std::cout << "  circularity = " << circularity(p) << "\n";
        std::cout << "  continuity  = " << continuity(p) << "\n";
        std::cout << "  thickness   = " << rim_thickness(labels, H, W, l) << "\n";
    }

    return 0;
}
