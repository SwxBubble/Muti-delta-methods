#include "feature/features.h"
#include "chunk/chunk.h"
#include "utils/gear.h"
#include "utils/rabin.cpp"
#include <algorithm>
#include <cstdint>
#include <queue>

#include <iomanip>
#include <iostream>
#include <unordered_set>

namespace Delta {
Feature FinesseFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int sub_chunk_length = chunk->len() / (sf_subf_ * sf_cnt_);
  uint8_t *content = chunk->buf();
  std::vector<uint64_t> sub_features(sf_cnt_ * sf_subf_, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  // calculate sub features.
  for (int i = 0; i < sub_features.size(); i++) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);
    for (int j = 0; j < sub_chunk_length; j++) {
      rabin_append(&rabin_ctx, content[j]);
      sub_features[i] = std::max(rabin_ctx.digest, sub_features[i]);
    }
    content += sub_chunk_length;
  }

  // group the sub features into super features.
  for (int i = 0; i < sub_features.size(); i += sf_subf_) {
    std::sort(sub_features.begin() + i, sub_features.begin() + i + sf_subf_);
  }
  for (int i = 0; i < sf_cnt_; i++) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);
    for (int j = 0; j < sf_subf_; j++) {
      auto sub_feature = sub_features[sf_subf_ * i + j];
      auto data_ptr = (uint8_t *)&sub_feature;
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
    0xff4be8c,  0x6f485986, 0x12843ff,  0x5b47dc4d, 0x7faa9b8a, 0xd547b8ba,
    0xf9979921, 0x4f5400da, 0x725f79a9, 0x3c9321ac, 0x32716d,   0x3f5adf5d,
};

Feature NTransformFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int features_num = sf_cnt_ * sf_subf_;
  std::vector<uint32_t> sub_features(features_num, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint64_t finger_print = 0;
  // calculate sub features.
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    for (int j = 0; j < features_num; j++) {
      const uint32_t transform = (M[j] * finger_print + A[j]);
      // we need to guarantee that when sub_features[i] is not inited,
      // always set its value
      if (sub_features[j] >= transform || 0 == sub_features[j])
        sub_features[j] = transform;
    }
  }

  // group sub features into super features.
  auto hash_buf = (const uint8_t *const)(sub_features.data());
  for (int i = 0; i < sf_cnt_; i++) {
    uint64_t hash_value = 0;
    auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);
    for (int j = 0; j < sf_subf_ * sizeof(uint32_t); j++) {
      hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
    }
    super_features[i] = hash_value;
  }
  return super_features;
}


/*
64bit 的指纹还是截断成了32bit的transform特征，主要是为了适配odess subfeature的特征类型变化，改为vector<uint32_t>，如果不改的话，odess subfeature的特征类型是vector<uint64_t>，就无法直接使用了。
*/
// Feature OdessFeature::operator()(std::shared_ptr<Chunk> chunk) {
//   int features_num = sf_cnt_ * sf_subf_;
//   std::vector<uint32_t> sub_features(features_num, 0);
//   std::vector<uint64_t> super_features(sf_cnt_, 0);

//   int chunk_length = chunk->len();
//   uint8_t *content = chunk->buf();
//   uint64_t finger_print = 0;
//   // calculate sub features.
//   for (int i = 0; i < chunk_length; i++) {
//     finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
//     if ((finger_print & mask_) == 0) {
//       for (int j = 0; j < features_num; j++) {
//         const uint32_t transform = (M[j] * finger_print + A[j]);
//         // we need to guarantee that when sub_features[i] is not inited,
//         // always set its value
//         if (sub_features[j] >= transform || 0 == sub_features[j])
//           sub_features[j] = transform;
//       }
//     }
//   }

//   // group sub features into super features.
//   auto hash_buf = (const uint8_t *const)(sub_features.data());
//   for (int i = 0; i < sf_cnt_; i++) {
//     uint64_t hash_value = 0;
//     auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);
//     for (int j = 0; j < sf_subf_ * sizeof(uint32_t); j++) {
//       hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
//     }
//     super_features[i] = hash_value;
//   }
//   return super_features;
// }

Feature OdessFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int features_num = sf_cnt_ * sf_subf_;

  std::vector<uint32_t> sub_features(features_num, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  // 记录每个 sub-feature 最终由哪个采样点产生
  // -1 表示没有被任何 sampled point 更新
  std::vector<int> source_pos(features_num, -1);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint64_t finger_print = 0;

  uint64_t sampled_points = 0;

  // calculate sub features
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];

    if ((finger_print & mask_) == 0) {
      sampled_points++;

      for (int j = 0; j < features_num; j++) {
        const uint32_t transform =
            static_cast<uint32_t>(M[j] * finger_print + A[j]);

        // we need to guarantee that when sub_features[j] is not inited,
        // always set its value
        if (sub_features[j] >= transform || 0 == sub_features[j]) {
          sub_features[j] = transform;
          source_pos[j] = i;
        }
      }
    }
  }

  // ===== update Odess statistics =====
  total_chunks_++;
  total_chunk_bytes_ += static_cast<uint64_t>(chunk_length);
  total_sampled_points_ += sampled_points;
  total_generated_subfeatures_ += static_cast<uint64_t>(features_num);

  if (sampled_points == 0) {
    zero_sample_chunks_++;
  }

  std::unordered_set<int> unique_sources;
  int valid_source_count = 0;

  for (int pos : source_pos) {
    if (pos >= 0) {
      unique_sources.insert(pos);
      valid_source_count++;
    }
  }

  const uint64_t unique_cnt =
      static_cast<uint64_t>(unique_sources.size());

  total_unique_source_points_ += unique_cnt;

  if (valid_source_count > static_cast<int>(unique_cnt)) {
    total_duplicate_source_features_ +=
        static_cast<uint64_t>(valid_source_count - unique_cnt);
  }

  // group sub features into super features
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

OdessFeature::~OdessFeature() {
  std::cout << "\n[Odess Feature Stats]" << std::endl;

  std::cout << "Total chunks processed: "
            << total_chunks_ << std::endl;

  std::cout << "Total chunk bytes: "
            << total_chunk_bytes_ << std::endl;

  std::cout << "Total sampled points: "
            << total_sampled_points_ << std::endl;

  if (total_chunks_ > 0) {
    std::cout << "Average sampled points per chunk: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_sampled_points_) /
                     static_cast<double>(total_chunks_)
              << std::defaultfloat << std::endl;

    std::cout << "Average chunk size: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_chunk_bytes_) /
                     static_cast<double>(total_chunks_)
              << " bytes"
              << std::defaultfloat << std::endl;

    std::cout << "Zero-sample chunks: "
              << zero_sample_chunks_ << " ("
              << std::fixed << std::setprecision(2)
              << static_cast<double>(zero_sample_chunks_) * 100.0 /
                     static_cast<double>(total_chunks_)
              << "%)"
              << std::defaultfloat << std::endl;

    std::cout << "Average unique source points per chunk: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_unique_source_points_) /
                     static_cast<double>(total_chunks_)
              << std::defaultfloat << std::endl;
  }

  std::cout << "Total generated subfeatures: "
            << total_generated_subfeatures_ << std::endl;

  std::cout << "Duplicate-source subfeatures: "
            << total_duplicate_source_features_;

  if (total_generated_subfeatures_ > 0) {
    std::cout << " ("
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_duplicate_source_features_) * 100.0 /
                     static_cast<double>(total_generated_subfeatures_)
              << "%)"
              << std::defaultfloat;
  }

  std::cout << std::endl << std::endl;
}


// 1  64 bit---32bit 修改点：适配odess subfeature的特征类型变化，改为vector<uint32_t>
// Feature OdessSubfeatures::operator()(std::shared_ptr<Chunk> chunk) {
//   int mask_ = default_odess_mask;
//   int features_num = 12;
//   //std::vector<uint64_t> sub_features(features_num, 0);
//   std::vector<uint32_t> sub_features(features_num, 0);

//   int chunk_length = chunk->len();
//   uint8_t *content = chunk->buf();
//   uint32_t finger_print = 0;
//   // calculate sub features.
//   for (int i = 0; i < chunk_length; i++) {
//     finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
//     if ((finger_print & mask_) == 0) {
//       for (int j = 0; j < features_num; j++) {
//         //const uint64_t transform = (M[j] * finger_print + A[j]);
//         const uint32_t transform = (M[j] * finger_print + A[j]);
//         // we need to guarantee that when sub_features[i] is not inited,
//         // always set its value
//         if (sub_features[j] >= transform || 0 == sub_features[j])
//           sub_features[j] = transform;
//       }
//     }
//   }

//   return sub_features;
// }


// 1 /*Palantir提取分层特征*/
// Feature PalantirFeature::operator()(std::shared_ptr<Chunk> chunk) {
//   // 修改点 1：接收的类型必须改为 uint32_t
//   auto sub_features = std::get<std::vector<uint32_t>>(get_sub_features_(chunk));
//   std::vector<std::vector<uint64_t>> results; // 超级特征仍然保持 uint64_t

//   auto group = [&](int sf_cnt, int sf_subf) -> std::vector<uint64_t> {
//     std::vector<uint64_t> super_features(sf_cnt, 0);
//     auto hash_buf = (const uint8_t *const)(sub_features.data());
    
//     for (int i = 0; i < sf_cnt; i++) {
//       uint64_t hash_value = 0;
//       // 修改点 2：步长偏移量改为 sizeof(uint32_t)
//       auto this_hash_buf = hash_buf + i * sf_subf * sizeof(uint32_t);
      
//       // 修改点 3（修复 Bug）：从 j = 0 开始，边界改为 sizeof(uint32_t)
//       for (int j = 0; j < sf_subf * sizeof(uint32_t); j++) {
//         hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
//       }
//       super_features[i] = hash_value;
//     }
//     return super_features;
//   };

//   results.push_back(group(3, 4));
//   results.push_back(group(4, 3));
//   results.push_back(group(6, 2));
//   return results;
// }


// 原版（已注释掉）
Feature OdessSubfeatures::operator()(std::shared_ptr<Chunk> chunk) {
  int mask_ = default_odess_mask;
  int features_num = 12;
  std::vector<uint64_t> sub_features(features_num, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint32_t finger_print = 0;
  // calculate sub features.
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    if ((finger_print & mask_) == 0) {
      for (int j = 0; j < features_num; j++) {
        const uint64_t transform = (M[j] * finger_print + A[j]);     
        // we need to guarantee that when sub_features[i] is not inited,
        // always set its value
        if (sub_features[j] >= transform || 0 == sub_features[j])
          sub_features[j] = transform;
      }
    }
  }

  return sub_features;
}


//原版
Feature PalantirFeature::operator()(std::shared_ptr<Chunk> chunk) {
  auto sub_features = std::get<std::vector<uint64_t>>(get_sub_features_(chunk));
  std::vector<std::vector<uint64_t>> results;

  auto group = [&](int sf_cnt, int sf_subf) -> std::vector<uint64_t> {
    std::vector<uint64_t> super_features(sf_cnt, 0);
    auto hash_buf = (const uint8_t *const)(sub_features.data());
    for (int i = 0; i < sf_cnt; i++) {
      uint64_t hash_value = 0;
      auto this_hash_buf = hash_buf + i * sf_subf * sizeof(uint64_t);
      for (int j = 4; j < sf_subf * sizeof(uint64_t); j++) {
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
} // namespace Delta