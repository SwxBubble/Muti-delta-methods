#include "index/palantir_index_3.h"
#include <algorithm>
#include <unordered_set>

namespace Delta {

std::optional<chunk_id> PalantirIndex3::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> PalantirIndex3::GetBaseChunkIDs(const Feature &feat,size_t top_k) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  std::vector<chunk_id> result;
  if (top_k == 0) {
    return result;
  }

  std::unordered_set<chunk_id> seen;

  // 逐层瀑布检索。
  for (size_t layer = 0; layer < features.size(); ++layer) {
    std::unordered_map<chunk_id, uint32_t> match_count;
    const auto &layer_features = features[layer];

    // 1) Voting
    for (size_t i = 0; i < layer_features.size(); ++i) {
      const uint64_t feature_val = layer_features[i];
      if (!index_[layer][i].count(feature_val)) {
        continue;
      }
      for (const auto &id : index_[layer][i].at(feature_val)) {
        match_count[id]++;
      }
    }

    if (match_count.empty()) {
      continue;
    }

    // 2) Ranking
    std::vector<std::pair<chunk_id, uint32_t>> ranked(match_count.begin(),
                                                       match_count.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
      if (lhs.second != rhs.second) {
        return lhs.second > rhs.second;
      }
      // 关键改动 2：同票时优先较新的 chunk（chunk_id 更大）。
      return lhs.first > rhs.first;
    });

    // 3) Filtering
    const uint32_t threshold = thresholds_[layer];
    for (const auto &[id, count] : ranked) {
      if (count < threshold) {
        break;
      }
      if (seen.insert(id).second) {
        result.push_back(id);
        if (result.size() >= top_k) {
          return result;
        }
      }
    }
  }

  return result;
}

void PalantirIndex3::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  for (size_t layer = 0; layer < features.size(); ++layer) {
    for (size_t i = 0; i < features[layer].size(); ++i) {
      auto &posting = index_[layer][i][features[layer][i]];
      posting.push_back(id);
      if (posting.size() > posting_list_cap_) {
        posting.erase(posting.begin());
      }
    }
  }
}

} // namespace Delta
