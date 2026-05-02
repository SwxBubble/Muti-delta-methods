#pragma once

#include "feature/features.h"

#include <array>
#include <memory>
#include <vector>

namespace Delta {

struct CDFEParams {
    int min_subblock_size = 256;
    int avg_subblock_size = 512;
    int max_subblock_size = 1024;

    // 单 mask：要求形如 2^k - 1，例如 0x1FF = 511
    uint64_t boundary_mask = 0x1FF;

    int split_window_size = 32;
    int feature_window_size = 16;

    // 现在固定每个 subblock 只产出 1 个 feature
    int local_features_per_subblock = 1;
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

  // std::array<uint64_t, 2> ExtractLocalRobustFeatures(const uint8_t *sbuf,
  //                                                    int slen) const;
  uint64_t ExtractOneLocalFeature(const uint8_t *sbuf, int slen) const;

  CDFESetOrderFeature BuildFeatureSet(const uint8_t *buf, int chunk_len) const;
};

// CDFE-SuperFeature 先作为一个独立的 FeatureCalculator 实现，后续根据需要再考虑和 CDFE-SetOrderV2Feature 的关系。  
class CDFESuperFeature : public FeatureCalculator {
public:
  explicit CDFESuperFeature(const CDFEParams &params,
                            int sf_cnt = 3,
                            int sf_subf = 4)
      : params_(params), sf_cnt_(sf_cnt), sf_subf_(sf_subf) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;
  ~CDFESuperFeature() override;

private:
  CDFEParams params_;
  int sf_cnt_;
  int sf_subf_;
  mutable size_t total_chunks_ = 0; 
  mutable size_t total_subblocks_ = 0;
  mutable size_t boundary_cut_subblocks_ = 0;
  mutable size_t forced_cut_subblocks_ = 0;

  std::vector<SubblockSpan> SplitIntoSubblocks(const uint8_t *buf,
                                               int chunk_len) const;

  uint64_t ExtractOneLocalFeature(const uint8_t *sbuf, int slen) const;

  std::vector<uint64_t> BuildCDFEValues(const uint8_t *buf,
                                        int chunk_len) const;

  std::vector<uint64_t> BuildSuperFeatures(const std::vector<uint64_t> &values) const;
};

} // namespace Delta