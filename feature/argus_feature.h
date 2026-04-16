#pragma once

#include "feature/features.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace Delta {

constexpr int default_argus_bin_cnt = 3;
constexpr uint64_t default_argus_sample_mask = (1ULL << 7) - 1;  // 1/128

class Chunk;

class ArgusFeature : public FeatureCalculator {
public:
    ArgusFeature(const int bin_cnt = default_argus_bin_cnt,
                 const uint64_t sample_mask = default_argus_sample_mask)
        : bin_cnt_(bin_cnt), sample_mask_(sample_mask) {}

    Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
    const int bin_cnt_;
    const uint64_t sample_mask_;

    static constexpr uint8_t kSuffixMask = 0x0f;

    static constexpr std::array<uint8_t, 16> kBinMap = {
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2
    };
};

}  // namespace Delta