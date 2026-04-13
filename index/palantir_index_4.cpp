#include "index/palantir_index_4.h"
#include <algorithm>
#include <unordered_set>

namespace Delta {

std::optional<chunk_id> PalantirIndex4::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> PalantirIndex4::GetBaseChunkIDs(const Feature &feat, size_t top_k) {
  // 解析 Palantir 特征：包含 3 层的二维数组
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  std::vector<chunk_id> result;
  std::unordered_set<chunk_id> seen; // 全局去重，防止上一层选过的块在下一层重复出现

  // 逐层进行瀑布流式查询
  for (int layer = 0; layer < features.size(); layer++) {
    std::unordered_map<chunk_id, uint32_t> match_count;
    
    // 1. 计票阶段 (Voting) - 引入抗偏移的跨界查找！
    const auto& layer_features = features[layer];
    for (int i = 0; i < layer_features.size(); i++) {
        const uint64_t feature_val = layer_features[i];
        bool found_in_place = false;

        // 步骤一：优先在“本职位置 (i)”的桶里找（精确对齐查找）
        if (index_[layer][i].count(feature_val)) {
            for (const auto &id : index_[layer][i].at(feature_val)) {
                match_count[id]++;
            }
            found_in_place = true; // 记录：在原本的位置找到了
        }

        // 步骤二（你的核心创新）：如果本职位置没找到，跨界去其他位置 (j) 的桶里“捡漏”！
        if (!found_in_place) {
            for (int j = 0; j < layer_features.size(); j++) {
                if (i == j) continue; // 自己的位置查过了，跳过

                if (index_[layer][j].count(feature_val)) {
                    for (const auto &id : index_[layer][j].at(feature_val)) {
                        match_count[id]++;
                    }
                    // 注：跨界找到后，是否要直接 break 退出 j 的循环，取决于你对“票数上限”的定义。
                    // 建议这里不写 break，让它把所有可能的相似块都投一票。
                }
            }
        }
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

void PalantirIndex4::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  // 数据入库：把当前块的 ID 登记到三层结构的对应票箱中
  for (int layer = 0; layer < features.size(); layer++) {
    for (int i = 0; i < features[layer].size(); i++) {
      index_[layer][i][features[layer][i]].push_back(id);
    }
  }
}

} // namespace Delta