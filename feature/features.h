#pragma once
#include <memory>
#include <variant>
#include <vector>

#include <cstdint>

namespace Delta {
constexpr int default_finesse_sf_cnt = 3;
// every super feature is grouped with 4 sub-features by default
constexpr int default_finesse_sf_subf = 4;

constexpr int default_odess_sf_cnt = 3;
constexpr int default_odess_sf_subf = 4;
constexpr uint64_t default_odess_mask = (1 << 7) - 1;
class Chunk;
// 原版
// using Feature = std::variant<std::vector<std::vector<uint64_t>>,
//                              std::vector<uint64_t>
//                              >;

// using Feature = std::variant<std::vector<std::vector<uint64_t>>,
//                              std::vector<uint32_t>,
//                              std::vector<uint64_t>
//                              >;

//cdfe-v2
struct SubblockSpan {
  int start;
  int len;
  int rank;
};

struct CDFELocalFeature {
  uint64_t value;
  uint16_t subblock_rank;
  float norm_pos;
};

using CDFESetOrderFeature = std::vector<CDFELocalFeature>;

using Feature = std::variant<std::vector<std::vector<uint64_t>>,
                             std::vector<uint32_t>,
                             std::vector<uint64_t>,
                             CDFESetOrderFeature>;

class FeatureCalculator {
public:
  virtual ~FeatureCalculator() = default;
  virtual Feature operator()(std::shared_ptr<Chunk> chunk) = 0;
};

class FinesseFeature : public FeatureCalculator {
public:
  FinesseFeature(const int sf_cnt = default_finesse_sf_cnt,
                 const int sf_subf = default_finesse_sf_subf)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  // grouped super features count
  const int sf_cnt_;
  // how much sub feature does a one super feature contain
  const int sf_subf_;
};

class NTransformFeature : public FeatureCalculator {
public:
  NTransformFeature(const int sf_cnt = 3, const int sf_subf = 4)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  // grouped super features count
  const int sf_cnt_;
  // how much sub feature does a one super feature contain
  const int sf_subf_;
};

class OdessFeature : public FeatureCalculator {
public:
  OdessFeature(const int sf_cnt = default_odess_sf_cnt,
               const int sf_subf = default_odess_sf_subf,
               const int mask = default_odess_mask)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf), mask_(mask) {}

  ~OdessFeature() override;


  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  // grouped super features count
  const int sf_cnt_;
  // how much sub feature does a one super feature contain
  const int sf_subf_;

  const int mask_;
  
  // ===== Odess sampling statistics =====
  uint64_t total_chunks_ = 0;
  uint64_t total_chunk_bytes_ = 0;

  // 满足 (finger_print & mask_) == 0 的采样点总数
  uint64_t total_sampled_points_ = 0;

  // 一个采样点都没有的 chunk 数
  uint64_t zero_sample_chunks_ = 0;

  // 每个 chunk 理论生成 sf_cnt_ * sf_subf_ 个 sub-features
  uint64_t total_generated_subfeatures_ = 0;

  // 最终 sub-features 来自多少个不同采样点，用于观察 Odess useless feature 问题
  uint64_t total_unique_source_points_ = 0;
  uint64_t total_duplicate_source_features_ = 0;
};

class OdessSubfeatures : public FeatureCalculator {
public:
  Feature operator()(std::shared_ptr<Chunk> chunk);
};

class PalantirFeature : public FeatureCalculator {
public:
  Feature operator()(std::shared_ptr<Chunk> chunk);
private:
  OdessSubfeatures get_sub_features_;
};
} // namespace Delta