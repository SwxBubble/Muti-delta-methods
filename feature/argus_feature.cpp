#include "feature/argus_feature.h"
#include "chunk/chunk.h"
#include "utils/gear.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace Delta {

// Reuse the existing transform constants from features.cpp style.
// Since the paper does not publish exact mi/ai constants,
// using fixed constants here is the closest reproducible engineering choice.
static uint32_t ARGUS_M[] = {
    0x5b49898a, 0xe4f94e27, 0x95f658b2
};

static uint32_t ARGUS_A[] = {
    0x0ff4be8c, 0x6f485986, 0x012843ff
};

Feature ArgusFeature::operator()(std::shared_ptr<Chunk> chunk) {
    std::vector<uint32_t> features(bin_cnt_, 0);

    const int chunk_length = chunk->len();
    uint8_t* content = chunk->buf();

    uint64_t finger_print = 0;

    for (int i = 0; i < chunk_length; i++) {
        finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];

        // Content-defined sampling
        if ((finger_print & sample_mask_) != 0) {
            continue;
        }

        // Algorithm-1 style binning: use current byte low 4 bits
        const uint8_t suffix = content[i] & kSuffixMask;
        const int bin = kBinMap[suffix];

        // one transform per bin
        const uint32_t transform = ARGUS_M[bin] * static_cast<uint32_t>(finger_print) + ARGUS_A[bin];

        // keep min value in each bin
        if (features[bin] == 0 || transform < features[bin]) {
            features[bin] = transform;
        }
    }

    return features;
}

} // namespace Delta