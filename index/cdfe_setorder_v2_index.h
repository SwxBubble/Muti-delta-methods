#pragma once

#include "index/index.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Delta {

struct CDFEPosting {
  chunk_id id;
  uint16_t subblock_rank;
  float norm_pos;
};

struct CandidateStat {
  uint16_t overlap_hits = 0;
  uint16_t aligned_hits = 0;
  std::vector<std::pair<uint16_t, uint16_t>> matched_ranks;
};

class CDFESetOrderV2Index : public Index {
public:
  CDFESetOrderV2Index(size_t topk_candidates = 4,
                      int min_overlap_hits = 3,
                      float pos_tolerance = 0.15f,
                      size_t hot_posting_limit = 256,
                      float order_lambda = 2.0f)
      : topk_candidates_(topk_candidates),
        min_overlap_hits_(min_overlap_hits),
        pos_tolerance_(pos_tolerance),
        hot_posting_limit_(hot_posting_limit),
        order_lambda_(order_lambda) {}

  std::optional<chunk_id> GetBaseChunkID(const Feature &feat) override;
  std::vector<chunk_id> GetBaseChunkCandidates(const Feature &feat,
                                               size_t topk) override;
  void AddFeature(const Feature &feat, chunk_id id) override;

  bool RecoverFromFile(const std::string &path) override { return true; }
  bool DumpToFile(const std::string &path) override { return true; }

private:
  std::unordered_map<uint64_t, std::vector<CDFEPosting>> inverted_;

  size_t topk_candidates_;
  int min_overlap_hits_;
  float pos_tolerance_;
  size_t hot_posting_limit_;
  float order_lambda_;

  float ComputeOrderConsistency(
      const std::vector<std::pair<uint16_t, uint16_t>> &matched_ranks) const;
};

} // namespace Delta