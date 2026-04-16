#include "index/argus_index.h"

#include <unordered_map>

namespace Delta {

std::optional<chunk_id> ArgusIndex::GetBaseChunkID(const Feature& feat) {
    const auto& features = std::get<std::vector<uint32_t>>(feat);

    std::unordered_map<chunk_id, uint32_t> match_count;

    for (int i = 0; i < feature_count_; i++) {
        const uint32_t feature = features[i];
        const auto& table = index_[i];

        auto it = table.find(feature);
        if (it == table.end()) {
            continue;
        }

        match_count[it->second]++;
    }

    if (match_count.empty()) {
        return std::nullopt;
    }

    chunk_id best_id = 0;
    uint32_t best_match = 0;

    for (const auto& [id, cnt] : match_count) {
        if (cnt > best_match || (cnt == best_match && id > best_id)) {
            best_match = cnt;
            best_id = id;
        }
    }

    return best_id;
}

void ArgusIndex::AddFeature(const Feature& feat, chunk_id id) {
    const auto& features = std::get<std::vector<uint32_t>>(feat);

    for (int i = 0; i < feature_count_; i++) {
        // Paper-style replacement:
        // newer chunk replaces older one on the same plain feature.
        index_[i][features[i]] = id;
    }
}

} // namespace Delta