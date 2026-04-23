#include "delta_compression.h"
#include "chunk/chunk.h"
#include "chunk/fast_cdc.h"
#include "chunk/rabin_cdc.h"
#include "config.h"
#include "encoder/xdelta.h"
#include "index/best_fit_index.h"
#include "index/palantir_index.h"
#include "index/palantir_index_2.h"  // <--- 添加这一行 palantir_index_2 每层选择匹配特征最多的块，且每层的及格线（阈值）不同，形成瀑布流式的多层过滤机制
#include "index/palantir_index_3.h"
#include "index/palantir_index_4.h"
#include "index/palantir_index_5.h"

#include "feature/argus_feature.h"
#include "index/argus_index.h"

#include "index/super_feature_index.h"
#include "storage/storage.h"
#include <glog/logging.h>
#include <iomanip>
#include <iostream>

#include <limits>

#include <string>
#include <vector>
//cdfe-v2
#include "feature/cdfe.h"
#include "index/cdfe_setorder_v2_index.h"
namespace Delta {
void DeltaCompression::AddFile(const std::string &file_name) {
  FileMeta file_meta;
  file_meta.file_name = file_name;
  file_meta.start_chunk_id = -1;
  this->chunker_->ReinitWithFile(file_name);
  while (true) {
    auto chunk = chunker_->GetNextChunk();
    if (nullptr == chunk)
      break;
    if (-1 == file_meta.start_chunk_id)
      file_meta.start_chunk_id = chunk->id();
    uint32_t dedup_base_id = dedup_->ProcessChunk(chunk);
    total_size_origin_ += chunk->len();
    // duplicate chunk
    if (dedup_base_id != chunk->id()) {
      storage_->WriteDuplicateChunk(chunk, dedup_base_id);
      duplicate_chunk_count_++;
      continue;
    }

    auto write_base_chunk = [this](const std::shared_ptr<Chunk> &chunk) {
      storage_->WriteBaseChunk(chunk);
      base_chunk_count_++;
      total_size_compressed_ += chunk->len();
    };

    auto write_delta_chunk = [this](const std::shared_ptr<Chunk> &chunk,
                                    const std::shared_ptr<Chunk> &delta_chunk,
                                    const uint32_t base_chunk_id) {
      chunk_size_before_delta_ += chunk->len();
      storage_->WriteDeltaChunk(delta_chunk, base_chunk_id);
      delta_chunk_count_++;
      chunk_size_after_delta_ += delta_chunk->len();
      total_size_compressed_ += delta_chunk->len();
    };

    // auto feature = (*feature_)(chunk);
    // auto base_chunk_id = index_->GetBaseChunkID(feature);
    // if (!base_chunk_id.has_value()) {
    //   index_->AddFeature(feature, chunk->id());
    //   write_base_chunk(chunk);
    //   continue;
    // }

    // auto delta_chunk =
    //     storage_->GetDeltaEncodedChunk(chunk, base_chunk_id.value());
    // write_delta_chunk(chunk, delta_chunk, base_chunk_id.value());

    //cdfe-v2
    auto feature = (*feature_)(chunk);

    // 默认走旧索引时，topk_candidates=1；
    // 只有 cdfe-setorder-v2 才会真正返回多个候选
    auto cfg = Config::Instance().get();
    auto feature_cfg = cfg->get_table("feature");
    size_t topk_candidates =
        static_cast<size_t>(
            feature_cfg->get_as<int64_t>("topk_candidates").value_or(1));

    auto candidate_ids = index_->GetBaseChunkCandidates(feature, topk_candidates);

    if (candidate_ids.empty()) {
      index_->AddFeature(feature, chunk->id());
      write_base_chunk(chunk);
      continue;
    }

    // 对 top-k 候选逐个做真实 delta 编码，选最优 base
    std::shared_ptr<Chunk> best_delta_chunk = nullptr;
    chunk_id best_base_id = 0;
    size_t best_delta_size = std::numeric_limits<size_t>::max();

    for (auto cid : candidate_ids) {
      auto delta_chunk = storage_->GetDeltaEncodedChunk(chunk, cid);
      if (delta_chunk &&
          static_cast<size_t>(delta_chunk->len()) < best_delta_size) {
        best_delta_size = static_cast<size_t>(delta_chunk->len());
        best_delta_chunk = delta_chunk;
        best_base_id = cid;
      }
    }

    // 如果最优 delta 仍然不划算，则退回写 base chunk
    if (!best_delta_chunk ||
        best_delta_size >= static_cast<size_t>(chunk->len())) {
      index_->AddFeature(feature, chunk->id());
      write_base_chunk(chunk);
      continue;
    }

    write_delta_chunk(chunk, best_delta_chunk, best_base_id);

    //**************** */
    file_meta.end_chunk_id = chunk->id();
  }
  file_meta_writer_.Write(file_meta);
}

DeltaCompression::~DeltaCompression() {
  auto print_ratio = [](size_t a, size_t b) {
    double ratio = (double)a / (double)b;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "(" << ratio * 100 << "%)" << std::endl;
    std::cout << std::defaultfloat;
  };
  uint32_t chunk_count =
      base_chunk_count_ + delta_chunk_count_ + duplicate_chunk_count_;
  std::cout << "Total chunk count: " << chunk_count << std::endl;
  std::cout << "Base chunk count: " << base_chunk_count_;
  print_ratio(base_chunk_count_, chunk_count);
  std::cout << "Delta chunk count: " << delta_chunk_count_;
  print_ratio(delta_chunk_count_, chunk_count);
  std::cout << "Duplicate chunk count: " << duplicate_chunk_count_;
  print_ratio(duplicate_chunk_count_, chunk_count);
  std::cout << "DCR (Delta Compression Ratio): ";
  print_ratio(total_size_origin_, total_size_compressed_);
  std::cout << "before " << total_size_origin_
            << " after: " << total_size_compressed_ << std::endl;
  std::cout << "DCE (Delta Compression Efficiency): ";
  print_ratio(chunk_size_after_delta_, chunk_size_before_delta_);
}

#define declare_feature_type(NAME, FEATURE, INDEX)                             \
  {                                                                            \
#NAME, \
[]() -> FeatureIndex { \
  return {std::make_unique<FEATURE>(), \
          std::make_unique<INDEX>()}; \
}                                                                       \
  }

DeltaCompression::DeltaCompression() {
  auto config = Config::Instance().get();
  auto index_path = *config->get_as<std::string>("index_path");
  auto chunk_data_path = *config->get_as<std::string>("chunk_data_path");
  auto chunk_meta_path = *config->get_as<std::string>("chunk_meta_path");
  auto file_meta_path = *config->get_as<std::string>("file_meta_path");
  auto dedup_index_path = *config->get_as<std::string>("dedup_index_path");

  auto chunker = config->get_table("chunker");
  auto chunker_type = *chunker->get_as<std::string>("type");
  if (chunker_type == "rabin-cdc" || chunker_type == "fast-cdc") {
    auto min_chunk_size = *chunker->get_as<int64_t>("min_chunk_size");
    auto max_chunk_size = *chunker->get_as<int64_t>("max_chunk_size");
    auto stop_mask = *chunker->get_as<int64_t>("stop_mask");
    if (chunker_type == "rabin-cdc") {
      this->chunker_ =
          std::make_unique<RabinCDC>(min_chunk_size, max_chunk_size, stop_mask);
      LOG(INFO) << "Add RabinCDC chunker, min_chunk_size=" << min_chunk_size
                << " max_chunk_size=" << max_chunk_size
                << " stop_mask=" << stop_mask;
    } else if (chunker_type == "fast-cdc") {
      this->chunker_ =
          std::make_unique<FastCDC>(min_chunk_size, max_chunk_size, stop_mask);
      LOG(INFO) << "Add FastCDC chunker, min_chunk_size=" << min_chunk_size
                << " max_chunk_size=" << max_chunk_size
                << " stop_mask=" << stop_mask;
    }
  } else {
    LOG(FATAL) << "Unknown chunker type " << chunker_type;
  }

  auto feature = config->get_table("feature");
  auto feature_type = *feature->get_as<std::string>("type");

  
  using FeatureIndex =
      std::pair<std::unique_ptr<FeatureCalculator>, std::unique_ptr<Index>>;
  std::unordered_map<std::string, std::function<FeatureIndex()>>
      feature_index_map = {
          declare_feature_type(finesse, FinesseFeature, SuperFeatureIndex),
          declare_feature_type(odess, OdessFeature, SuperFeatureIndex),
          declare_feature_type(n-transform, NTransformFeature,SuperFeatureIndex),
          //declare_feature_type(palantir, PalantirFeature, PalantirIndex), //原版
          //declare_feature_type(palantir2, PalantirFeature, PalantirIndex2), // <---  PalantirIndex2 
          //declare_feature_type(palantir3, PalantirFeature, PalantirIndex3), // <---  PalantirIndex3 在2的基础上增加了每个 posting list 的容量限制，降低陈旧候选的噪声
          //declare_feature_type(palantir4, PalantirFeature, PalantirIndex4), // <---  PalantirIndex4 在2的基础上修改了跨界查找特征
          declare_feature_type(palantir5, PalantirFeature, PalantirIndex5), // <---  PalantirIndex5 在4的基础上进一步优化了性能
          declare_feature_type(argus, ArgusFeature, ArgusIndex), // <---  ArgusFeature + ArgusIndex 实现了论文中基于 min-hash 的 Argus 方法，作为一个完全不同设计思路的对照组加入实验
          declare_feature_type(bestfit, OdessSubfeatures, BestFitIndex)};
  //cdfe-v2      
  if (feature_type == "cdfe-setorder-v2") {
  CDFEParams params;
  params.min_subblock_size =
      feature->get_as<int64_t>("min_subblock_size").value_or(64);
  params.max_subblock_size =
      feature->get_as<int64_t>("max_subblock_size").value_or(512);
  params.boundary_divisor =
      static_cast<uint64_t>(
          feature->get_as<int64_t>("boundary_divisor").value_or(16));

  params.split_window_size =
      feature->get_as<int64_t>("split_window_size").value_or(16);
  params.feature_window_size =
      feature->get_as<int64_t>("feature_window_size").value_or(16);
  params.local_features_per_subblock =
      feature->get_as<int64_t>("local_features_per_subblock").value_or(2);

  size_t topk_candidates =
      static_cast<size_t>(
          feature->get_as<int64_t>("topk_candidates").value_or(4));
  int min_matched_subblocks =
      feature->get_as<int64_t>("min_matched_subblocks").value_or(4);
  int min_aligned_subblocks =
      feature->get_as<int64_t>("min_aligned_subblocks").value_or(2);
  float pos_tolerance =
      static_cast<float>(
          feature->get_as<double>("pos_tolerance").value_or(0.15));
  size_t hot_posting_limit =
      static_cast<size_t>(
          feature->get_as<int64_t>("hot_posting_limit").value_or(256));
  float order_lambda =
      static_cast<float>(
          feature->get_as<double>("order_lambda").value_or(2.0));

  this->feature_ = std::make_unique<CDFESetOrderV2Feature>(params);
  this->index_ = std::make_unique<CDFESetOrderV2Index>(
      topk_candidates,
      min_matched_subblocks,
      min_aligned_subblocks,
      pos_tolerance,
      hot_posting_limit,
      order_lambda);

  LOG(INFO) << "Add CDFE-SetOrder-v2 feature extractor"
            << " min_subblock_size=" << params.min_subblock_size
            << " max_subblock_size=" << params.max_subblock_size
            << " boundary_divisor=" << params.boundary_divisor
            << " split_window_size=" << params.split_window_size
            << " feature_window_size=" << params.feature_window_size
            << " topk_candidates=" << topk_candidates;
} else {
  if (!feature_index_map.count(feature_type))
    LOG(FATAL) << "Unknown feature type " << feature_type;

  auto [feature_ptr, index_ptr] = feature_index_map[feature_type]();
  this->feature_ = std::move(feature_ptr);
  this->index_ = std::move(index_ptr);
}        
  // if (!feature_index_map.count(feature_type))
  //   LOG(FATAL) << "Unknown feature type " << feature_type;
  // auto [feature_ptr, index_ptr] = feature_index_map[feature_type]();
  // this->feature_ = std::move(feature_ptr);
  // this->index_ = std::move(index_ptr);

  this->dedup_ = std::make_unique<Dedup>(dedup_index_path);

  auto storage = config->get_table("storage");
  auto encoder_name = *storage->get_as<std::string>("encoder");
  auto cache_size = *storage->get_as<int64_t>("cache_size");
  std::unique_ptr<Encoder> encoder;
  if (encoder_name == "xdelta") {
    encoder = std::make_unique<XDelta>();
  } else {
    LOG(FATAL) << "Unknown encoder type " << encoder_name;
  }
  this->storage_ = std::make_unique<Storage>(
      chunk_data_path, chunk_meta_path, std::move(encoder), true, cache_size);
  this->file_meta_writer_.Init(file_meta_path);
}
} // namespace Delta