#pragma once
#include "feature/features.h"
#include "index/index.h"
#include <unordered_map>
#include <vector>
#include <optional>

namespace Delta {

class PalantirIndex5 : public Index {
public:
  PalantirIndex5() {
    // 初始化 3 层索引结构，每层只建 1 个全局哈希表（大桶）
    index_.resize(3);
  }
  
  ~PalantirIndex5() = default;

  std::optional<chunk_id> GetBaseChunkID(const Feature &feat);
  std::vector<chunk_id> GetBaseChunkIDs(const Feature &feat, size_t top_k);
  void AddFeature(const Feature &feat, chunk_id id);
  bool RecoverFromFile(const std::string &path) { return true; }
  bool DumpToFile(const std::string &path) { return true; }

private:
  // 核心降维：index_[layer][feature_hash] -> vector<chunk_id>
  // 不再区分特征到底是第几个，全部扔进对应 layer 的哈希表中
  std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>> index_;
  
  // 阈值保持最佳实验配置
  const std::vector<uint32_t> thresholds_ = {1, 1, 2}; 
};

} // namespace Delta