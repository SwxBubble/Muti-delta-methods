#pragma once

#include "index/index.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unordered_set>

namespace Delta {

struct CDFEPosting {
  chunk_id id;
  uint16_t subblock_rank;
  float norm_pos;
};

struct CandidateStat {
  // 命中了多少个不同的 query subblock
  std::unordered_set<uint16_t> matched_query_subblocks;

  // 命中了多少个位置也对得上的 query subblock
  std::unordered_set<uint16_t> aligned_query_subblocks;

  // 用于后续计算顺序一致性
  std::vector<std::pair<uint16_t, uint16_t>> matched_ranks;
};


struct CDFECandidateDebugInfo {
  chunk_id id;

  int query_subblocks = 0;
  int base_subblocks = 0;

  int matched_subblocks = 0;
  int aligned_subblocks = 0;

  float matched_ratio = 0.0f;
  float aligned_ratio = 0.0f;
  float order_consistency = 0.0f;
  float score = 0.0f;
};

class CDFESetOrderV2Index : public Index {
public:
CDFESetOrderV2Index(size_t topk_candidates = 4,
                    int min_matched_subblocks = 4,
                    int min_aligned_subblocks = 2,
                    float pos_tolerance = 0.15f,
                    size_t hot_posting_limit = 256,
                    float order_lambda = 2.0f)
    : topk_candidates_(topk_candidates),
      min_matched_subblocks_(min_matched_subblocks),
      min_aligned_subblocks_(min_aligned_subblocks),
      pos_tolerance_(pos_tolerance),
      hot_posting_limit_(hot_posting_limit),
      order_lambda_(order_lambda) {}

  std::optional<chunk_id> GetBaseChunkID(const Feature &feat) override;
  std::vector<chunk_id> GetBaseChunkCandidates(const Feature &feat,
                                               size_t topk) override;
  void AddFeature(const Feature &feat, chunk_id id) override;

  bool RecoverFromFile(const std::string &path) override { return true; }
  bool DumpToFile(const std::string &path) override { return true; }

  const std::vector<CDFECandidateDebugInfo> &GetLastTopKDebugInfos() const {
  return last_topk_debug_infos_;
  }

private:
  std::unordered_map<uint64_t, std::vector<CDFEPosting>> inverted_;

  size_t topk_candidates_;
  int min_matched_subblocks_;
  int min_aligned_subblocks_;
  float pos_tolerance_;
  size_t hot_posting_limit_;
  float order_lambda_;

  float ComputeOrderConsistency(
      const std::vector<std::pair<uint16_t, uint16_t>> &matched_ranks) const;


  std::unordered_map<chunk_id, int> chunk_subblock_count_;
  std::vector<CDFECandidateDebugInfo> last_topk_debug_infos_;

  static int CountSubblocks(const CDFESetOrderFeature &features);
};



} // namespace Delta