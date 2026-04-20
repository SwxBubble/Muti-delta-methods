#pragma once

#include "feature/features.h"

#include <optional>
#include <string>
#include <vector>

namespace Delta {

using chunk_id = uint32_t;

class Index {
public:
  virtual ~Index() = default;

  virtual std::optional<chunk_id> GetBaseChunkID(const Feature &feat) = 0;

  // 新增：默认实现，兼容旧索引
  virtual std::vector<chunk_id> GetBaseChunkCandidates(const Feature &feat,
                                                       size_t topk) {
    auto id = GetBaseChunkID(feat);
    if (id.has_value()) {
      return {id.value()};
    }
    return {};
  }

  virtual void AddFeature(const Feature &feat, chunk_id id) = 0;

  virtual bool RecoverFromFile(const std::string &path) = 0;
  virtual bool DumpToFile(const std::string &path) = 0;
};

} // namespace Delta