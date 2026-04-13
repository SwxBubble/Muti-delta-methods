#include "index/palantir_index_5.h"
#include <algorithm>
#include <unordered_set>

namespace Delta {

std::optional<chunk_id> PalantirIndex5::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> PalantirIndex5::GetBaseChunkIDs(const Feature &feat, size_t top_k) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  std::vector<chunk_id> result;
  std::unordered_set<chunk_id> seen; // 全局防重复候选块

  for (int layer = 0; layer < features.size(); layer++) {
    std::unordered_map<chunk_id, uint32_t> match_count;
    
    // 【导师的防坑设计】：块内特征去重
    // 防止一个块包含全是0的重复数据，导致生成一样的超级特征，从而引起重复计票
    std::vector<uint64_t> unique_features = features[layer];
    std::sort(unique_features.begin(), unique_features.end());
    unique_features.erase(std::unique(unique_features.begin(), unique_features.end()), unique_features.end());

    // 1. 计票阶段：O(1) 极速跨界查找
    // 因为所有特征都在一个桶里，遍历自身的独特特征去大桶里查即可
    for (const uint64_t feature_val : unique_features) {
      if (!index_[layer].count(feature_val)) {
        continue;
      }
      for (const auto &id : index_[layer].at(feature_val)) {
        match_count[id]++;
      }
    }
    
    if (match_count.empty()) {
      continue; 
    }

    // 2. 排名阶段 (与之前完全一致)
    std::vector<std::pair<chunk_id, uint32_t>> ranked(match_count.begin(), match_count.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second != rhs.second) {
                  return lhs.second > rhs.second; 
                }
                return lhs.first < rhs.first; 
              });

    // 3. 淘汰与录取阶段 (与之前完全一致)
    uint32_t threshold = thresholds_[layer];
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

void PalantirIndex5::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  
  for (int layer = 0; layer < features.size(); layer++) {
    // 【导师的防坑设计】：入库前去重
    std::vector<uint64_t> unique_features = features[layer];
    std::sort(unique_features.begin(), unique_features.end());
    unique_features.erase(std::unique(unique_features.begin(), unique_features.end()), unique_features.end());

    // 数据入库：无视位置，全部扔进 layer 对应的大桶里
    for (const uint64_t feature_val : unique_features) {
      index_[layer][feature_val].push_back(id);
    }
  }
}

} // namespace Delta