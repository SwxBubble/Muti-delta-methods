#pragma once
#include "feature/features.h"
#include "index/index.h"
#include <optional>
#include <unordered_map>
#include <vector>

namespace Delta {

class PalantirIndex3 : public Index {
public:
  PalantirIndex3() {
    // 3 层结构：分别对应 3/4/6 个超级特征位置。
    index_.push_back(
        std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>(3));
    index_.push_back(
        std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>(4));
    index_.push_back(
        std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>(6));
  }

  ~PalantirIndex3() = default;

  std::optional<chunk_id> GetBaseChunkID(const Feature &feat);
  std::vector<chunk_id> GetBaseChunkIDs(const Feature &feat, size_t top_k);
  void AddFeature(const Feature &feat, chunk_id id);
  bool RecoverFromFile(const std::string &path) { return true; }
  bool DumpToFile(const std::string &path) { return true; }

private:
  // index_[layer][feature_pos][feature_hash] -> posting list(chunk ids)
  std::vector<std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>>
      index_;

  // 每层最低票数阈值。
  const std::vector<uint32_t> thresholds_ = {1, 1, 2};

  // 关键改动 1：每个 posting list 仅保留最近 N 个 chunk_id，降低陈旧候选噪声。
  static constexpr size_t posting_list_cap_ = 16;
};

} // namespace Delta
