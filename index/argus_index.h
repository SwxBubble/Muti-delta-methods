#pragma once

#include "feature/features.h"
#include "index/index.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Delta {

using chunk_id = uint32_t;

class ArgusIndex : public Index {
public:
    explicit ArgusIndex(const int feature_count = 3)
        : feature_count_(feature_count) {
        index_.resize(feature_count_);
    }

    std::optional<chunk_id> GetBaseChunkID(const Feature& feat) override;
    void AddFeature(const Feature& feat, chunk_id id) override;

    bool RecoverFromFile(const std::string& path) override { return true; }
    bool DumpToFile(const std::string& path) override { return true; }

private:
    // One plain-feature table per bin:
    // feature_value -> latest chunk_id
    std::vector<std::unordered_map<uint32_t, chunk_id>> index_;
    const int feature_count_;
};

} // namespace Delta