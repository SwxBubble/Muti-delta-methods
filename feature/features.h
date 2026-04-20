#pragma once
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace Delta {
constexpr int default_finesse_sf_cnt = 3;
// every super feature is grouped with 4 sub-features by default
constexpr int default_finesse_sf_subf = 4;

constexpr int default_odess_sf_cnt = 3;
constexpr int default_odess_sf_subf = 4;
constexpr uint64_t default_odess_mask = (1 << 7) - 1;

constexpr int default_cdfe_ordered_feature_count = 12;
constexpr int default_cdfe_ordered_min_subblock = 64;
constexpr int default_cdfe_ordered_avg_subblock = 256;
constexpr int default_cdfe_ordered_max_subblock = 1024;
// Boundary is searched after avg_subblock_size; low bits equal zero means cut.
// mask = 31 means a hit probability of roughly 1/32.
constexpr uint64_t default_cdfe_ordered_boundary_mask = (1 << 5) - 1;

class Chunk;

// using Feature = std::variant<std::vector<std::vector<uint64_t>>,
//                              std::vector<uint64_t>>;
using Feature = std::variant<std::vector<std::vector<uint64_t>>,
                             std::vector<uint32_t>,
                             std::vector<uint64_t>>;

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
  const int sf_cnt_;
  const int sf_subf_;
};

class NTransformFeature : public FeatureCalculator {
public:
  NTransformFeature(const int sf_cnt = 3, const int sf_subf = 4)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  const int sf_cnt_;
  const int sf_subf_;
};

class OdessFeature : public FeatureCalculator {
public:
  OdessFeature(const int sf_cnt = default_odess_sf_cnt,
               const int sf_subf = default_odess_sf_subf,
               const int mask = default_odess_mask)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf), mask_(mask) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  const int sf_cnt_;
  const int sf_subf_;
  const int mask_;
};

class OdessSubfeatures : public FeatureCalculator {
public:
  Feature operator()(std::shared_ptr<Chunk> chunk) override;
};

class PalantirFeature : public FeatureCalculator {
public:
  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  OdessSubfeatures get_sub_features_;
};

// CDFE-Ordered-v1:
// 1) split a chunk into content-defined subblocks,
// 2) hash every subblock into a 64-bit fingerprint,
// 3) bucket subblocks by coarse byte offset,
// 4) keep one representative feature (minimum hash) per bucket.
class CDFEOrderedFeature : public FeatureCalculator {
public:
  CDFEOrderedFeature(
      int feature_count = default_cdfe_ordered_feature_count,
      int min_subblock_size = default_cdfe_ordered_min_subblock,
      int avg_subblock_size = default_cdfe_ordered_avg_subblock,
      int max_subblock_size = default_cdfe_ordered_max_subblock,
      uint64_t boundary_mask = default_cdfe_ordered_boundary_mask)
      : feature_count_(feature_count),
        min_subblock_size_(min_subblock_size),
        avg_subblock_size_(avg_subblock_size),
        max_subblock_size_(max_subblock_size),
        boundary_mask_(boundary_mask) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  const int feature_count_;
  const int min_subblock_size_;
  const int avg_subblock_size_;
  const int max_subblock_size_;
  const uint64_t boundary_mask_;
};
} // namespace Delta
