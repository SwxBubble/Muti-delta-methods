#pragma once
#include "feature/features.h"
#include "index/index.h"
#include <unordered_map>
#include <vector>
#include <optional>

namespace Delta {

class PalantirIndex4 : public Index {
public:
  PalantirIndex4() {
    // 初始化 3 层索引结构，分别对应 3、4、6 个超级特征
    index_.push_back(std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>(3));
    index_.push_back(std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>(4));
    index_.push_back(std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>(6));
  }
  
  ~PalantirIndex4() = default;

  // 删除了所有的 override 关键字，保持与原版 PalantirIndex 相同的签名风格
  std::optional<chunk_id> GetBaseChunkID(const Feature &feat);
  std::vector<chunk_id> GetBaseChunkIDs(const Feature &feat, size_t top_k);
  void AddFeature(const Feature &feat, chunk_id id);
  bool RecoverFromFile(const std::string &path) { return true; }
  bool DumpToFile(const std::string &path) { return true; }

private:
  // 数据结构: index_[layer][feature_pos][feature_hash] -> vector<chunk_id>
  std::vector<std::vector<std::unordered_map<uint64_t, std::vector<chunk_id>>>> index_;
  
  // 核心创新：为每一层定制不同的及格线（阈值）。
  // 第 0 层（共 3 票）：得票 >= 2 才能成为候选
  // 第 1 层（共 4 票）：得票 >= 2 才能成为候选
  // 第 2 层（共 6 票）：得票 >= 3 才能成为候选
  const std::vector<uint32_t> thresholds_ = {1, 1, 2}; 
};

} // namespace Delta