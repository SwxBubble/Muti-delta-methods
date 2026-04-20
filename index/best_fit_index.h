#pragma once
#include "feature/features.h"
#include "index/index.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Delta {

using chunk_id = uint32_t;
class Chunk;
class BestFitIndex : public Index {
public:
  BestFitIndex(const int feature_count = 12, const int match_threshold = 4)
      : feature_count_(feature_count), match_threshold_(match_threshold) {
    for (int i = 0; i < feature_count_; i++) {
      index_.push_back({});
    }
  }
  std::optional<chunk_id> GetBaseChunkID(const Feature &feat) override;
  void AddFeature(const Feature &feat, chunk_id id) override;
  bool RecoverFromFile(const std::string &path) override { return true; }
  bool DumpToFile(const std::string &path) override { return true; }

private:
  std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>> index_;
  const int feature_count_;
  const int match_threshold_;
};
} // namespace Delta
