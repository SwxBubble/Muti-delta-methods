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



// std::vector<SubblockSpan>
// CDFESetOrderV2Feature::SplitIntoSubblocks(const uint8_t *buf,
//                                           int chunk_len) const {
//   std::vector<SubblockSpan> res;
//   if (chunk_len <= 0) {
//     return res;
//   }

//   int start = 0;
//   int rank = 0;

//   while (start < chunk_len) {
//     const int end_limit =
//         std::min(start + params_.max_subblock_size, chunk_len);

//     int cut = -1;
//     bool hit_boundary = false;

//     for (int end = start + 1; end <= end_limit; ++end) {
//       const int cur_len = end - start;

//       if (cur_len < params_.min_subblock_size) {
//         continue;
//       }

//       if (cur_len < params_.split_window_size) {
//         continue;
//       }

//       const int window_start = end - params_.split_window_size;
//       const uint64_t fp =
//           HashWindow(buf + window_start, params_.split_window_size);

//       if (params_.boundary_divisor > 0 &&
//           (fp % params_.boundary_divisor) == 0) {
//         cut = end;
//         hit_boundary = true;
//         break;
//       }
//     }

//     if (cut == -1) {
//       cut = end_limit;
//       hit_boundary = false;
//     }

//     // 统计
//     total_subblocks_++;
//     if (hit_boundary) {
//       boundary_cut_subblocks_++;
//     } else {
//       forced_cut_subblocks_++;
//     }

//     res.push_back({start, cut - start, rank});
//     start = cut;
//     rank++;
//   }

//   return res;
// }

std::vector<SubblockSpan>
CDFESetOrderV2Feature::SplitIntoSubblocks(const uint8_t *buf,
                                          int chunk_len) const {
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


// std::array<uint64_t, 2>
// CDFESetOrderV2Feature::ExtractLocalRobustFeatures(const uint8_t *sbuf,
//                                                   int slen) const {
//   // 子块太短时，退化成整块 hash 的双副本
//   if (slen <= params_.feature_window_size ||
//       slen < 2 * params_.feature_window_size) {
//     const uint64_t g = HashBytes(sbuf, slen);
//     return {g, g};
//   }

//   const int m = slen - params_.feature_window_size + 1;
//   const int mid = m / 2;

//   uint64_t min1 = std::numeric_limits<uint64_t>::max();
//   uint64_t min2 = std::numeric_limits<uint64_t>::max();

//   for (int off = 0; off < m; ++off) {
//     const uint64_t h =
//         HashWindow(sbuf + off, params_.feature_window_size);
//     if (off < mid) {
//       min1 = std::min(min1, h);
//     } else {
//       min2 = std::min(min2, h);
//     }
//   }

//   if (min1 == std::numeric_limits<uint64_t>::max()) {
//     min1 = HashBytes(sbuf, slen);
//   }
//   if (min2 == std::numeric_limits<uint64_t>::max()) {
//     min2 = min1;
//   }

//   return {min1, min2};
// }


uint64_t CDFESetOrderV2Feature::ExtractOneLocalFeature(
    const uint8_t *sbuf,
    int slen) const {
    if (slen <= 0) {
        return 0;
    }

    // 子块太短，退化成整段 hash
    if (slen < params_.feature_window_size) {
        return Mix64(HashBytes(sbuf, slen));
    }

    const int m = slen - params_.feature_window_size + 1;
    uint64_t min_h = std::numeric_limits<uint64_t>::max();

    for (int off = 0; off < m; ++off) {
        uint64_t h = HashWindow(sbuf + off, params_.feature_window_size);
        h = Mix64(h);
        if (h < min_h) {
            min_h = h;
        }
    }

    if (min_h == std::numeric_limits<uint64_t>::max()) {
        min_h = Mix64(HashBytes(sbuf, slen));
    }

    return min_h;
}




// CDFESetOrderFeature
// CDFESetOrderV2Feature::BuildFeatureSet(const uint8_t *buf,
//                                        int chunk_len) const {
//   CDFESetOrderFeature feats;

//   total_chunks_++;    //计数

//   auto subblocks = SplitIntoSubblocks(buf, chunk_len);

//   feats.reserve(subblocks.size() * 2);

//   for (const auto &sb : subblocks) {
//     const uint8_t *sbuf = buf + sb.start;
//     const int slen = sb.len;

//     auto local = ExtractLocalRobustFeatures(sbuf, slen);
//     const float norm_pos =
//         static_cast<float>(sb.start + sb.len * 0.5f) /
//         static_cast<float>(chunk_len);

//     // 至少保证 1，最多不超过 2
//     int k = params_.local_features_per_subblock;
//     if (k < 1) k = 1;
//     if (k > 2) k = 2;

//     for (int i = 0; i < k; ++i) {
//       feats.push_back(
//           CDFELocalFeature{local[i],
//                           static_cast<uint16_t>(sb.rank),
//                           norm_pos});
//     }
//   }

//   return feats;
// }

CDFESetOrderFeature CDFESetOrderV2Feature::BuildFeatureSet(
    const uint8_t *buf,
    int chunk_len) const {
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

} // namespace Delta