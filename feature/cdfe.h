#pragma once

#include "feature/features.h"

#include <array>
#include <memory>
#include <vector>

namespace Delta {

struct CDFEParams {
  int min_subblock_size = 64;
  int max_subblock_size = 512;
  uint64_t boundary_divisor = 16;

  int split_window_size = 16;    // 子块划分窗口
  int feature_window_size = 16;  // 子块特征窗口

  int local_features_per_subblock = 2; // 先固定为 2
};

class Chunk;

class CDFESetOrderV2Feature : public FeatureCalculator {
public:
  explicit CDFESetOrderV2Feature(const CDFEParams &params) : params_(params) {}
  ~CDFESetOrderV2Feature() override;

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  CDFEParams params_;

    // 统计信息
  mutable size_t total_chunks_ = 0;
  mutable size_t total_subblocks_ = 0;
  mutable size_t boundary_cut_subblocks_ = 0;
  mutable size_t forced_cut_subblocks_ = 0;


  std::vector<SubblockSpan> SplitIntoSubblocks(const uint8_t *buf,int chunk_len) const;

  std::array<uint64_t, 2> ExtractLocalRobustFeatures(const uint8_t *sbuf,
                                                     int slen) const;

  CDFESetOrderFeature BuildFeatureSet(const uint8_t *buf, int chunk_len) const;
};

} // namespace Delta