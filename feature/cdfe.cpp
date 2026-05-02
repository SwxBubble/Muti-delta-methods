#include "feature/cdfe.h"

#include "chunk/chunk.h"
#include "utils/gear.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <iomanip>
#include <iostream>

namespace Delta {
namespace {

// 为了先把逻辑写稳，窗口 hash 这里直接对窗口内容做一次 gear-style hash。
// 这样不依赖复杂 rolling 状态，也不会跨子块污染。
// 后续如果你要优化速度，再把这里换成真正 rolling hash 就行。
uint64_t HashWindow(const uint8_t *data, int len) {
  uint64_t h = 0;
  for (int i = 0; i < len; ++i) {
    h = (h << 1) + GEAR_TABLE[data[i]];
  }
  return h;
}

static inline uint64_t GearWindowInit(const uint8_t *data, int window_size) {
    uint64_t h = 0;

    for (int i = 0; i < window_size; ++i) {
        h = (h << 1) + GEAR_TABLE[data[i]];
    }

    return h;
}

static inline uint64_t GearWindowRoll(uint64_t prev,
                                      uint8_t out_byte,
                                      uint8_t in_byte,
                                      int window_size) {
    // HashWindow 的定义是：
    // h = (((gear[b0] << 1) + gear[b1]) << 1 + ...)
    // 所以 b0 的权重是 2^(window_size - 1)
    const uint64_t out_contrib =
        GEAR_TABLE[out_byte] << (window_size - 1);

    uint64_t h = prev - out_contrib;
    h = (h << 1) + GEAR_TABLE[in_byte];

    return h;
}

// 子块太短时的回退 hash
uint64_t HashBytes(const uint8_t *data, int len) {
  uint64_t h = 1469598103934665603ULL; // FNV offset basis
  for (int i = 0; i < len; ++i) {
    h ^= static_cast<uint64_t>(data[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

static inline uint64_t Mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t GearUpdate(uint64_t fp, uint8_t byte) {
    return (fp << 1) + GEAR_TABLE[byte];
}

static inline bool IsBoundary(uint64_t fp, uint64_t mask) {
    if (mask == 0) {
        return false;
    }
    return (Mix64(fp) & mask) == 0;
}

} // namespace

Feature CDFESetOrderV2Feature::operator()(std::shared_ptr<Chunk> chunk) {
  return BuildFeatureSet(chunk->buf(), chunk->len());
}


CDFESetOrderV2Feature::~CDFESetOrderV2Feature() {
  std::cout << "\n[CDFE-SetOrder-v2 Subblock Split Stats]" << std::endl;
  std::cout << "Total chunks processed: " << total_chunks_ << std::endl;
  std::cout << "Total subblocks: " << total_subblocks_ << std::endl;

  if (total_subblocks_ == 0) {
    std::cout << "Boundary-cut subblocks: 0" << std::endl;
    std::cout << "Forced-cut subblocks: 0" << std::endl;
    return;
  }

  auto print_ratio = [](size_t a, size_t b) {
    double ratio = static_cast<double>(a) / static_cast<double>(b);
    std::cout << std::fixed << std::setprecision(1)
              << ratio * 100.0 << "%";
    std::cout << std::defaultfloat;
  };

  std::cout << "Boundary-cut subblocks: " << boundary_cut_subblocks_ << " (";
  print_ratio(boundary_cut_subblocks_, total_subblocks_);
  std::cout << ")" << std::endl;

  std::cout << "Forced-cut subblocks: " << forced_cut_subblocks_ << " (";
  print_ratio(forced_cut_subblocks_, total_subblocks_);
  std::cout << ")" << std::endl;

  if (total_chunks_ > 0) {
    std::cout << "Average subblocks per chunk: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_subblocks_) /
                     static_cast<double>(total_chunks_)
              << std::defaultfloat << std::endl;
  }

  std::cout << std::endl;
}


std::vector<SubblockSpan>
CDFESetOrderV2Feature::SplitIntoSubblocks(const uint8_t *buf,int chunk_len) const {
    std::vector<SubblockSpan> res;
    if (chunk_len <= 0) {
        return res;
    }

    int start = 0;
    int rank = 0;

    while (start < chunk_len) {
        const int remain = chunk_len - start;

        // 最后剩余部分不大，直接作为最后一个子块
        if (remain <= params_.max_subblock_size) {
            res.push_back({start, remain, rank});
            total_subblocks_++;
            forced_cut_subblocks_++;
            break;
        }

        const int min_pos = std::min(start + params_.min_subblock_size, chunk_len);
        const int avg_pos = std::min(start + params_.avg_subblock_size, chunk_len);
        const int max_pos = std::min(start + params_.max_subblock_size, chunk_len);

        int cut = -1;
        bool hit_boundary = false;

        // 预先计算 [min_pos, max_pos] 每个切点对应的 Gear fingerprint。
        // fingerprint 从 start 开始递推，64-bit 溢出自然丢弃旧信息。
        std::vector<uint64_t> fps(max_pos - min_pos + 1);

        uint64_t fp = 0;
        for (int pos = start; pos < min_pos; ++pos) {
            fp = GearUpdate(fp, buf[pos]);
        }
        fps[0] = fp;

        for (int pos = min_pos + 1; pos <= max_pos; ++pos) {
            fp = GearUpdate(fp, buf[pos - 1]);
            fps[pos - min_pos] = fp;
        }

        auto boundary_at = [&](int pos) -> bool {
            if (pos < min_pos || pos > max_pos) {
                return false;
            }
            return IsBoundary(fps[pos - min_pos], params_.boundary_mask);
        };

        // 关键：从 avg 附近向左右搜索，而不是从 min 开始找第一个
        int left = avg_pos;
        int right = avg_pos + 1;

        while (left >= min_pos || right <= max_pos) {
            if (left >= min_pos && boundary_at(left)) {
                cut = left;
                hit_boundary = true;
                break;
            }

            if (right <= max_pos && boundary_at(right)) {
                cut = right;
                hit_boundary = true;
                break;
            }

            --left;
            ++right;
        }

        if (cut == -1) {
            cut = max_pos;
            hit_boundary = false;
        }

        if (cut <= start) {
            cut = std::min(start + params_.max_subblock_size, chunk_len);
            hit_boundary = false;
        }

        total_subblocks_++;
        if (hit_boundary) {
            boundary_cut_subblocks_++;
        } else {
            forced_cut_subblocks_++;
        }

        res.push_back({start, cut - start, rank});
        start = cut;
        rank++;
    }

    return res;
}


// uint64_t CDFESetOrderV2Feature::ExtractOneLocalFeature(const uint8_t *sbuf,int slen) const {
//     if (slen <= 0) {
//         return 0;
//     }

//     // 子块太短，退化成整段 hash
//     if (slen < params_.feature_window_size) {
//         return Mix64(HashBytes(sbuf, slen));
//     }

//     const int m = slen - params_.feature_window_size + 1;
//     uint64_t min_h = std::numeric_limits<uint64_t>::max();

//     for (int off = 0; off < m; ++off) {
//         uint64_t h = HashWindow(sbuf + off, params_.feature_window_size);
//         h = Mix64(h);
//         if (h < min_h) {
//             min_h = h;
//         }
//     }

//     if (min_h == std::numeric_limits<uint64_t>::max()) {
//         min_h = Mix64(HashBytes(sbuf, slen));
//     }

//     return min_h;
// }
uint64_t CDFESetOrderV2Feature::ExtractOneLocalFeature(
    const uint8_t *sbuf,
    int slen) const {
    if (slen <= 0) {
        return 0;
    }

    int w = params_.feature_window_size;

    if (w <= 0) {
        return Mix64(HashBytes(sbuf, slen));
    }

    // 当前 rolling window 递推写法要求 window_size < 64
    // 你现在常用 16/24/48，都满足。
    if (w >= 64) {
        w = 63;
    }

    // 子块太短时，退化成整段 hash
    if (slen < w) {
        return Mix64(HashBytes(sbuf, slen));
    }

    const int window_count = slen - w + 1;

    uint64_t fp = GearWindowInit(sbuf, w);
    uint64_t min_h = Mix64(fp);

    for (int off = 1; off < window_count; ++off) {
        fp = GearWindowRoll(
            fp,
            sbuf[off - 1],
            sbuf[off + w - 1],
            w
        );

        uint64_t h = Mix64(fp);

        if (h < min_h) {
            min_h = h;
        }
    }

    return min_h;
}


CDFESetOrderFeature CDFESetOrderV2Feature::BuildFeatureSet(const uint8_t *buf,int chunk_len) const {
    CDFESetOrderFeature feats;
    total_chunks_++;

    auto subblocks = SplitIntoSubblocks(buf, chunk_len);

    // 每个子块只产出 1 个 feature
    feats.reserve(subblocks.size());

    for (const auto &sb : subblocks) {
        const uint8_t *sbuf = buf + sb.start;
        const int slen = sb.len;

        const uint64_t one_feature = ExtractOneLocalFeature(sbuf, slen);

        const float norm_pos =
            static_cast<float>(sb.start + sb.len * 0.5f) /
            static_cast<float>(chunk_len);

        feats.push_back(CDFELocalFeature{
            one_feature,
            static_cast<uint16_t>(sb.rank),
            norm_pos
        });
    }

    return feats;
    }

// uint64_t CDFESuperFeature::ExtractOneLocalFeature(const uint8_t *sbuf,int slen) const {
//     // 直接复制 CDFESetOrderV2Feature::ExtractOneLocalFeature 的实现
//     if (slen <= 0) {
//         return 0;
//     }

//     // 子块太短，退化成整段 hash
//     if (slen < params_.feature_window_size) {
//         return Mix64(HashBytes(sbuf, slen));
//     }

//     const int m = slen - params_.feature_window_size + 1;
//     uint64_t min_h = std::numeric_limits<uint64_t>::max();

//     for (int off = 0; off < m; ++off) {
//         uint64_t h = HashWindow(sbuf + off, params_.feature_window_size);
//         h = Mix64(h);
//         if (h < min_h) {
//             min_h = h;
//         }
//     }

//     if (min_h == std::numeric_limits<uint64_t>::max()) {
//         min_h = Mix64(HashBytes(sbuf, slen));
//     }

//     return min_h;
// }

uint64_t CDFESuperFeature::ExtractOneLocalFeature(const uint8_t *sbuf,
                                                  int slen) const {
    if (slen <= 0) {
        return 0;
    }

    int w = params_.feature_window_size;

    if (w <= 0) {
        return Mix64(HashBytes(sbuf, slen));
    }

    // GearWindowRoll 里面有 << (w - 1)，避免位移超过 63。
    // 你现在常用 16 / 24 / 48，都是安全的。
    if (w >= 64) {
        w = 63;
    }

    // 子块太短，退化成整段 hash
    if (slen < w) {
        return Mix64(HashBytes(sbuf, slen));
    }

    const int window_count = slen - w + 1;

    // 第一个窗口正常初始化
    uint64_t fp = GearWindowInit(sbuf, w);
    uint64_t min_h = Mix64(fp);

    // 后续窗口递推更新，不再每次重新 HashWindow()
    for (int off = 1; off < window_count; ++off) {
        fp = GearWindowRoll(
            fp,
            sbuf[off - 1],       // 滑出窗口的字节
            sbuf[off + w - 1],   // 滑入窗口的字节
            w
        );

        uint64_t h = Mix64(fp);

        if (h < min_h) {
            min_h = h;
        }
    }

    return min_h;
}

    // CDFESuperFeature 的实现先直接复用 CDFESetOrderV2Feature 的逻辑，后续再根据需要调整。
std::vector<SubblockSpan>CDFESuperFeature::SplitIntoSubblocks(const uint8_t *buf, int chunk_len) const {
    // 直接复制 CDFESetOrderV2Feature::SplitIntoSubblocks 的实现
    std::vector<SubblockSpan> res;
    if (chunk_len <= 0) {
        return res;
    }

    int start = 0;
    int rank = 0;

    while (start < chunk_len) {
        const int remain = chunk_len - start;

        // 最后剩余部分不大，直接作为最后一个子块
        if (remain <= params_.max_subblock_size) {
            res.push_back({start, remain, rank});
            total_subblocks_++;
            forced_cut_subblocks_++;
            break;
        }

        const int min_pos = std::min(start + params_.min_subblock_size, chunk_len);
        const int avg_pos = std::min(start + params_.avg_subblock_size, chunk_len);
        const int max_pos = std::min(start + params_.max_subblock_size, chunk_len);

        int cut = -1;
        bool hit_boundary = false;

        // 预先计算 [min_pos, max_pos] 每个切点对应的 Gear fingerprint。
        // fingerprint 从 start 开始递推，64-bit 溢出自然丢弃旧信息。
        std::vector<uint64_t> fps(max_pos - min_pos + 1);

        uint64_t fp = 0;
        for (int pos = start; pos < min_pos; ++pos) {
            fp = GearUpdate(fp, buf[pos]);
        }
        fps[0] = fp;

        for (int pos = min_pos + 1; pos <= max_pos; ++pos) {
            fp = GearUpdate(fp, buf[pos - 1]);
            fps[pos - min_pos] = fp;
        }

        auto boundary_at = [&](int pos) -> bool {
            if (pos < min_pos || pos > max_pos) {
                return false;
            }
            return IsBoundary(fps[pos - min_pos], params_.boundary_mask);
        };

        // 关键：从 avg 附近向左右搜索，而不是从 min 开始找第一个
        int left = avg_pos;
        int right = avg_pos + 1;

        while (left >= min_pos || right <= max_pos) {
            if (left >= min_pos && boundary_at(left)) {
                cut = left;
                hit_boundary = true;
                break;
            }

            if (right <= max_pos && boundary_at(right)) {
                cut = right;
                hit_boundary = true;
                break;
            }

            --left;
            ++right;
        }

        if (cut == -1) {
            cut = max_pos;
            hit_boundary = false;
        }

        if (cut <= start) {
            cut = std::min(start + params_.max_subblock_size, chunk_len);
            hit_boundary = false;
        }

        total_subblocks_++;
        if (hit_boundary) {
            boundary_cut_subblocks_++;
        } else {
            forced_cut_subblocks_++;
        }

        res.push_back({start, cut - start, rank});
        start = cut;
        rank++;
    }

    return res;
}

// uint64_t CDFESuperFeature::ExtractOneLocalFeature(const uint8_t *sbuf,int slen) const {
//     // 直接复制 CDFESetOrderV2Feature::ExtractOneLocalFeature 的实现
//     if (slen <= 0) {
//         return 0;
//     }

//     // 子块太短，退化成整段 hash
//     if (slen < params_.feature_window_size) {
//         return Mix64(HashBytes(sbuf, slen));
//     }

//     const int m = slen - params_.feature_window_size + 1;
//     uint64_t min_h = std::numeric_limits<uint64_t>::max();

//     for (int off = 0; off < m; ++off) {
//         uint64_t h = HashWindow(sbuf + off, params_.feature_window_size);
//         h = Mix64(h);
//         if (h < min_h) {
//             min_h = h;
//         }
//     }

//     if (min_h == std::numeric_limits<uint64_t>::max()) {
//         min_h = Mix64(HashBytes(sbuf, slen));
//     }

//     return min_h;
// }



std::vector<uint64_t>CDFESuperFeature::BuildCDFEValues(const uint8_t *buf, int chunk_len) const {
    std::vector<uint64_t> values;

    auto subblocks = SplitIntoSubblocks(buf, chunk_len);
    values.reserve(subblocks.size());

    for (const auto &sb : subblocks) {
        const uint8_t *sbuf = buf + sb.start;
        const int slen = sb.len;

        uint64_t v = ExtractOneLocalFeature(sbuf, slen);
        values.push_back(v);
    }

    if (values.empty()) {
        values.push_back(0);
    }

    return values;
}

static uint32_t CDFE_M[] = {
    0x5b49898a, 0xe4f94e27, 0x95f658b2, 0x8f9c99fc,
    0xeba8d4d8, 0xba2c8e92, 0xa868aeb4, 0xd767df82,
    0x843606a4, 0xc1e70129, 0x32d9d1b0, 0xeb91e53c,
};

static uint32_t CDFE_A[] = {
    0x0ff4be8c, 0x6f485986, 0x012843ff, 0x5b47dc4d,
    0x7faa9b8a, 0xd547b8ba, 0xf9979921, 0x4f5400da,
    0x725f79a9, 0x3c9321ac, 0x0032716d, 0x3f5adf5d,
};

std::vector<uint64_t>
CDFESuperFeature::BuildSuperFeatures(const std::vector<uint64_t> &values) const {
    const int features_num = sf_cnt_ * sf_subf_;

    std::vector<uint32_t> sub_features(features_num, 0);
    std::vector<uint64_t> super_features(sf_cnt_, 0);

    for (uint64_t x64 : values) {
        // 和 N-transform 一样，最终 transform 截成 uint32_t
        uint32_t x = static_cast<uint32_t>(x64);

        for (int j = 0; j < features_num; ++j) {
            uint32_t transform = CDFE_M[j] * x + CDFE_A[j];

            if (sub_features[j] >= transform || sub_features[j] == 0) {
                sub_features[j] = transform;
            }
        }
    }

    // 和 NTransformFeature 一样，用 GEAR_TABLE hash 每组 uint32_t sub-features
    auto hash_buf = reinterpret_cast<const uint8_t *>(sub_features.data());

    for (int i = 0; i < sf_cnt_; ++i) {
        uint64_t hash_value = 0;
        auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);

        for (int j = 0; j < sf_subf_ * static_cast<int>(sizeof(uint32_t)); ++j) {
            hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
        }

        super_features[i] = hash_value;
    }

    return super_features;
}
Feature CDFESuperFeature::operator()(std::shared_ptr<Chunk> chunk) {
    const int chunk_len = chunk->len();
    total_chunks_++;
    const uint8_t *buf = chunk->buf();

    auto values = BuildCDFEValues(buf, chunk_len);
    auto super_features = BuildSuperFeatures(values);

    return super_features;
}
CDFESuperFeature::~CDFESuperFeature() {
  std::cout << "\n[CDFE-SetOrder-v2 Subblock Split Stats]" << std::endl;
  std::cout << "Total chunks processed: " << total_chunks_ << std::endl;
  std::cout << "Total subblocks: " << total_subblocks_ << std::endl;

  if (total_subblocks_ == 0) {
    std::cout << "Boundary-cut subblocks: 0" << std::endl;
    std::cout << "Forced-cut subblocks: 0" << std::endl;
    return;
  }

  auto print_ratio = [](size_t a, size_t b) {
    double ratio = static_cast<double>(a) / static_cast<double>(b);
    std::cout << std::fixed << std::setprecision(1)
              << ratio * 100.0 << "%";
    std::cout << std::defaultfloat;
  };

  std::cout << "Boundary-cut subblocks: " << boundary_cut_subblocks_ << " (";
  print_ratio(boundary_cut_subblocks_, total_subblocks_);
  std::cout << ")" << std::endl;

  std::cout << "Forced-cut subblocks: " << forced_cut_subblocks_ << " (";
  print_ratio(forced_cut_subblocks_, total_subblocks_);
  std::cout << ")" << std::endl;

  if (total_chunks_ > 0) {
    std::cout << "Average subblocks per chunk: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_subblocks_) /
                     static_cast<double>(total_chunks_)
              << std::defaultfloat << std::endl;
  }

  std::cout << std::endl;
}
} // namespace Delta