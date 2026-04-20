#include "feature/features.h"

#include "chunk/chunk.h"
#include "utils/gear.h"
#include "utils/rabin.cpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

namespace Delta {
namespace {
uint64_t FNV1a64(const uint8_t *data, int len) {
  uint64_t hash = 1469598103934665603ULL;
  for (int i = 0; i < len; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}
} // namespace

Feature FinesseFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int sub_chunk_length = chunk->len() / (sf_subf_ * sf_cnt_);
  uint8_t *content = chunk->buf();
  std::vector<uint64_t> sub_features(sf_cnt_ * sf_subf_, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  for (int i = 0; i < static_cast<int>(sub_features.size()); i++) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);
    for (int j = 0; j < sub_chunk_length; j++) {
      rabin_append(&rabin_ctx, content[j]);
      sub_features[i] = std::max(rabin_ctx.digest, sub_features[i]);
    }
    content += sub_chunk_length;
  }

  for (int i = 0; i < static_cast<int>(sub_features.size()); i += sf_subf_) {
    std::sort(sub_features.begin() + i, sub_features.begin() + i + sf_subf_);
  }
  for (int i = 0; i < sf_cnt_; i++) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);
    for (int j = 0; j < sf_subf_; j++) {
      auto sub_feature = sub_features[sf_subf_ * i + j];
      auto data_ptr = reinterpret_cast<uint8_t *>(&sub_feature);
      for (int k = 0; k < 8; k++) {
        rabin_append(&rabin_ctx, data_ptr[k]);
      }
    }
    super_features[i] = rabin_ctx.digest;
  }
  return super_features;
}

static uint32_t M[] = {
    0x5b49898a, 0xe4f94e27, 0x95f658b2, 0x8f9c99fc, 0xeba8d4d8, 0xba2c8e92,
    0xa868aeb4, 0xd767df82, 0x843606a4, 0xc1e70129, 0x32d9d1b0, 0xeb91e53c,
};

static uint32_t A[] = {
    0x0ff4be8c, 0x6f485986, 0x012843ff, 0x5b47dc4d, 0x7faa9b8a, 0xd547b8ba,
    0xf9979921, 0x4f5400da, 0x725f79a9, 0x3c9321ac, 0x0032716d, 0x3f5adf5d,
};

Feature NTransformFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int features_num = sf_cnt_ * sf_subf_;
  std::vector<uint32_t> sub_features(features_num, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint64_t finger_print = 0;
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    for (int j = 0; j < features_num; j++) {
      const uint32_t transform = (M[j] * finger_print + A[j]);
      if (sub_features[j] >= transform || 0 == sub_features[j]) {
        sub_features[j] = transform;
      }
    }
  }

  auto hash_buf = reinterpret_cast<const uint8_t *>(sub_features.data());
  for (int i = 0; i < sf_cnt_; i++) {
    uint64_t hash_value = 0;
    auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);
    for (int j = 0; j < sf_subf_ * static_cast<int>(sizeof(uint32_t)); j++) {
      hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
    }
    super_features[i] = hash_value;
  }
  return super_features;
}

Feature OdessFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int features_num = sf_cnt_ * sf_subf_;
  std::vector<uint32_t> sub_features(features_num, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint64_t finger_print = 0;
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    if ((finger_print & mask_) == 0) {
      for (int j = 0; j < features_num; j++) {
        const uint32_t transform = (M[j] * finger_print + A[j]);
        if (sub_features[j] >= transform || 0 == sub_features[j]) {
          sub_features[j] = transform;
        }
      }
    }
  }

  auto hash_buf = reinterpret_cast<const uint8_t *>(sub_features.data());
  for (int i = 0; i < sf_cnt_; i++) {
    uint64_t hash_value = 0;
    auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);
    for (int j = 0; j < sf_subf_ * static_cast<int>(sizeof(uint32_t)); j++) {
      hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
    }
    super_features[i] = hash_value;
  }
  return super_features;
}

Feature OdessSubfeatures::operator()(std::shared_ptr<Chunk> chunk) {
  int mask_ = default_odess_mask;
  int features_num = 12;
  std::vector<uint64_t> sub_features(features_num, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint32_t finger_print = 0;
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    if ((finger_print & mask_) == 0) {
      for (int j = 0; j < features_num; j++) {
        const uint64_t transform = (M[j] * finger_print + A[j]);
        if (sub_features[j] >= transform || 0 == sub_features[j]) {
          sub_features[j] = transform;
        }
      }
    }
  }

  return sub_features;
}

Feature PalantirFeature::operator()(std::shared_ptr<Chunk> chunk) {
  auto sub_features = std::get<std::vector<uint64_t>>(get_sub_features_(chunk));
  std::vector<std::vector<uint64_t>> results;

  auto group = [&](int sf_cnt, int sf_subf) -> std::vector<uint64_t> {
    std::vector<uint64_t> super_features(sf_cnt, 0);
    auto hash_buf = reinterpret_cast<const uint8_t *>(sub_features.data());
    for (int i = 0; i < sf_cnt; i++) {
      uint64_t hash_value = 0;
      auto this_hash_buf = hash_buf + i * sf_subf * sizeof(uint64_t);
      for (int j = 4; j < sf_subf * static_cast<int>(sizeof(uint64_t)); j++) {
        hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
      }
      super_features[i] = hash_value;
    }
    return super_features;
  };

  results.push_back(group(3, 4));
  results.push_back(group(4, 3));
  results.push_back(group(6, 2));
  return results;
}

Feature CDFEOrderedFeature::operator()(std::shared_ptr<Chunk> chunk) {
  const int chunk_length = chunk->len();
  const uint8_t *content = chunk->buf();
  std::vector<uint64_t> ordered_features(feature_count_, std::numeric_limits<uint64_t>::max());
  std::vector<bool> bucket_has_value(feature_count_, false);

  if (chunk_length <= 0) {
    return std::vector<uint64_t>(feature_count_, 0);
  }

  int start = 0;
  while (start < chunk_length) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);

    int pos = start;
    int best_cut = std::min(start + max_subblock_size_, chunk_length);
    while (pos < chunk_length && (pos - start) < max_subblock_size_) {
      rabin_append(&rabin_ctx, content[pos]);
      const int current_len = pos - start + 1;
      const bool reached_end = (pos + 1 == chunk_length);
      const bool reached_avg = current_len >= avg_subblock_size_;
      const bool reached_max = current_len >= max_subblock_size_;
      const bool boundary_hit = reached_avg && ((rabin_ctx.digest & boundary_mask_) == 0);
      if ((current_len >= min_subblock_size_ && boundary_hit) || reached_max || reached_end) {
        best_cut = pos + 1;
        break;
      }
      ++pos;
    }

    const int subblock_len = best_cut - start;
    const uint64_t subblock_hash = FNV1a64(content + start, subblock_len);
    int bucket = static_cast<int>((static_cast<uint64_t>(start) * feature_count_) /
                                  static_cast<uint64_t>(chunk_length));
    if (bucket >= feature_count_) {
      bucket = feature_count_ - 1;
    }
    ordered_features[bucket] = std::min(ordered_features[bucket], subblock_hash);
    bucket_has_value[bucket] = true;
    start = best_cut;
  }

  int first_non_empty = -1;
  for (int i = 0; i < feature_count_; ++i) {
    if (bucket_has_value[i]) {
      first_non_empty = i;
      break;
    }
  }
  if (first_non_empty == -1) {
    return std::vector<uint64_t>(feature_count_, 0);
  }

  for (int i = first_non_empty - 1; i >= 0; --i) {
    ordered_features[i] = ordered_features[i + 1];
    bucket_has_value[i] = true;
  }
  for (int i = first_non_empty + 1; i < feature_count_; ++i) {
    if (!bucket_has_value[i]) {
      ordered_features[i] = ordered_features[i - 1];
      bucket_has_value[i] = true;
    }
  }
  return ordered_features;
}
} // namespace Delta
