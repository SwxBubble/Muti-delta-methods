#include "feature/cdfe.h"

#include "chunk/chunk.h"
#include "utils/gear.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

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

} // namespace

Feature CDFESetOrderV2Feature::operator()(std::shared_ptr<Chunk> chunk) {
  return BuildFeatureSet(chunk->buf(), chunk->len());
}

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
    const int end_limit = std::min(start + params_.max_subblock_size, chunk_len);
    int cut = -1;

    // 从左往右扫描，找到第一个合法边界
    for (int end = start + 1; end <= end_limit; ++end) {
      const int cur_len = end - start;

      if (cur_len < params_.min_subblock_size) {
        continue;
      }

      // 只有当前子块内至少有一个完整 split window 时才检查边界
      if (cur_len < params_.split_window_size) {
        continue;
      }

      const int window_start = end - params_.split_window_size;
      const uint64_t fp =
          HashWindow(buf + window_start, params_.split_window_size);

      if (params_.boundary_divisor > 0 &&
          (fp % params_.boundary_divisor) == 0) {
        cut = end;
        break;
      }
    }

    // 找不到合法边界则强切
    if (cut == -1) {
      cut = end_limit;
    }

    res.push_back({start, cut - start, rank});
    start = cut;
    rank++;
  }

  return res;
}

std::array<uint64_t, 2>
CDFESetOrderV2Feature::ExtractLocalRobustFeatures(const uint8_t *sbuf,
                                                  int slen) const {
  // 子块太短时，退化成整块 hash 的双副本
  if (slen <= params_.feature_window_size ||
      slen < 2 * params_.feature_window_size) {
    const uint64_t g = HashBytes(sbuf, slen);
    return {g, g};
  }

  const int m = slen - params_.feature_window_size + 1;
  const int mid = m / 2;

  uint64_t min1 = std::numeric_limits<uint64_t>::max();
  uint64_t min2 = std::numeric_limits<uint64_t>::max();

  for (int off = 0; off < m; ++off) {
    const uint64_t h =
        HashWindow(sbuf + off, params_.feature_window_size);
    if (off < mid) {
      min1 = std::min(min1, h);
    } else {
      min2 = std::min(min2, h);
    }
  }

  if (min1 == std::numeric_limits<uint64_t>::max()) {
    min1 = HashBytes(sbuf, slen);
  }
  if (min2 == std::numeric_limits<uint64_t>::max()) {
    min2 = min1;
  }

  return {min1, min2};
}

CDFESetOrderFeature
CDFESetOrderV2Feature::BuildFeatureSet(const uint8_t *buf,
                                       int chunk_len) const {
  CDFESetOrderFeature feats;
  auto subblocks = SplitIntoSubblocks(buf, chunk_len);

  feats.reserve(subblocks.size() * 2);

  for (const auto &sb : subblocks) {
    const uint8_t *sbuf = buf + sb.start;
    const int slen = sb.len;

    auto local = ExtractLocalRobustFeatures(sbuf, slen);
    const float norm_pos =
        static_cast<float>(sb.start + sb.len * 0.5f) /
        static_cast<float>(chunk_len);

    feats.push_back(
        CDFELocalFeature{local[0],
                         static_cast<uint16_t>(sb.rank),
                         norm_pos});
    feats.push_back(
        CDFELocalFeature{local[1],
                         static_cast<uint16_t>(sb.rank),
                         norm_pos});
  }

  return feats;
}

} // namespace Delta