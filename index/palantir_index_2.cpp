#include "index/palantir_index_2.h"
#include <algorithm>
#include <unordered_set>

namespace Delta {

std::optional<chunk_id> PalantirIndex2::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> PalantirIndex2::GetBaseChunkIDs(const Feature &feat, size_t top_k) {
  // 解析 Palantir 特征：包含 3 层的二维数组
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  std::vector<chunk_id> result;
  std::unordered_set<chunk_id> seen; // 全局去重，防止上一层选过的块在下一层重复出现

  // 逐层进行瀑布流式查询
  for (int layer = 0; layer < features.size(); layer++) {
    std::unordered_map<chunk_id, uint32_t> match_count;
    
    // 1. 计票阶段 (Voting)
    const auto& layer_features = features[layer];
    for (int i = 0; i < layer_features.size(); i++) {
      const uint64_t feature_val = layer_features[i];
      if (!index_[layer][i].count(feature_val)) {
        continue; // 这个位置的特征没有命中，跳过
      }
      for (const auto &id : index_[layer][i].at(feature_val)) {
        match_count[id]++;
      }
    }
    
    if (match_count.empty()) {
      continue; // 这一层全军覆没，直接去下一层找
    }

    // 2. 排名阶段 (Ranking)
    std::vector<std::pair<chunk_id, uint32_t>> ranked(match_count.begin(), match_count.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second != rhs.second) {
                  return lhs.second > rhs.second; // 得票高的优先
                }
                return lhs.first < rhs.first; // 票数相同，越老的数据块优先
                //return lhs.first < rhs.first; // 票数相同，越新的数据块优先
              });

    // 3. 淘汰与录取阶段 (Filtering)
    uint32_t threshold = thresholds_[layer];
    for (const auto &[id, count] : ranked) {
      if (count < threshold) {
        break; // 小优化：因为已经降序排好，一旦发现低于阈值的，后面的肯定也不及格，直接 break
      }
      if (seen.insert(id).second) {
        result.push_back(id);
        if (result.size() >= top_k) {
          return result; // 只要找够了 top_k，立刻提前返回，榨干每一滴性能
        }
      }
    }
  }
  
  return result;
}

void PalantirIndex2::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  // 数据入库：把当前块的 ID 登记到三层结构的对应票箱中
  for (int layer = 0; layer < features.size(); layer++) {
    for (int i = 0; i < features[layer].size(); i++) {
      index_[layer][i][features[layer][i]].push_back(id);
    }
  }
}

} // namespace Delta