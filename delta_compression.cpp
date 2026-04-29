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

//
#include <fstream>

//
#include <chrono>

namespace Delta {
void DeltaCompression::AddFile(const std::string &file_name) {
  FileMeta file_meta;
  file_meta.file_name = file_name;
  file_meta.start_chunk_id = -1;
  //this->chunker_->ReinitWithFile(file_name);

  {
    auto t = Clock::now();
    this->chunker_->ReinitWithFile(file_name);
    file_reinit_time_ns_ += ElapsedNs(t);
  }

  while (true) {
    // auto chunk = chunker_->GetNextChunk();
    // if (nullptr == chunk)
    std::shared_ptr<Chunk> chunk;
    {
        auto t = Clock::now();
        chunk = chunker_->GetNextChunk();
        chunking_time_ns_ += ElapsedNs(t);
    }
    if (nullptr == chunk) break;
      break;
    if (-1 == file_meta.start_chunk_id)
      file_meta.start_chunk_id = chunk->id();
    //uint32_t dedup_base_id = dedup_->ProcessChunk(chunk);
    uint32_t dedup_base_id;
    {
        auto t = Clock::now();
        dedup_base_id = dedup_->ProcessChunk(chunk);
        dedup_time_ns_ += ElapsedNs(t);
    }

    total_size_origin_ += chunk->len();
    // duplicate chunk
    if (dedup_base_id != chunk->id()) {
      // storage_->WriteDuplicateChunk(chunk, dedup_base_id);
      // duplicate_chunk_count_++;
      // continue;
      {
          auto t = Clock::now();
          storage_->WriteDuplicateChunk(chunk, dedup_base_id);
          duplicate_write_time_ns_ += ElapsedNs(t);
      }
      duplicate_chunk_count_++;
      continue;
    }

    // auto write_base_chunk = [this](const std::shared_ptr<Chunk> &chunk) {
    //   storage_->WriteBaseChunk(chunk);
    //   base_chunk_count_++;
    //   total_size_compressed_ += chunk->len();
    // };

    // auto write_delta_chunk = [this](const std::shared_ptr<Chunk> &chunk,
    //                                 const std::shared_ptr<Chunk> &delta_chunk,
    //                                 const uint32_t base_chunk_id) {
    //   chunk_size_before_delta_ += chunk->len();
    //   storage_->WriteDeltaChunk(delta_chunk, base_chunk_id);
    //   delta_chunk_count_++;
    //   chunk_size_after_delta_ += delta_chunk->len();
    //   total_size_compressed_ += delta_chunk->len();
    // };

    auto write_base_chunk = [this](const std::shared_ptr<Chunk> &chunk) {
    auto t = Clock::now();
    storage_->WriteBaseChunk(chunk);
    base_write_time_ns_ += ElapsedNs(t);

    base_chunk_count_++;
    total_size_compressed_ += chunk->len();
    };

    auto write_delta_chunk = [this](const std::shared_ptr<Chunk> &chunk,
                                    const std::shared_ptr<Chunk> &delta_chunk,
                                    const uint32_t base_chunk_id) {
        chunk_size_before_delta_ += chunk->len();

        auto t = Clock::now();
        storage_->WriteDeltaChunk(delta_chunk, base_chunk_id);
        delta_write_time_ns_ += ElapsedNs(t);

        delta_chunk_count_++;
        chunk_size_after_delta_ += delta_chunk->len();
        total_size_compressed_ += delta_chunk->len();
    };

    //cdfe-v2
    //auto feature = (*feature_)(chunk);
    Feature feature;
    {
        auto t = Clock::now();
        feature = (*feature_)(chunk);
        feature_time_ns_ += ElapsedNs(t);
    }

    // 默认走旧索引时，topk_candidates=1；
    // 只有 cdfe-setorder-v2 才会真正返回多个候选
    auto cfg = Config::Instance().get();
    auto feature_cfg = cfg->get_table("feature");
    size_t topk_candidates =
        static_cast<size_t>(
            feature_cfg->get_as<int64_t>("topk_candidates").value_or(1));

// 为了和 cdfe_topk_debug.log 对齐，
// 这里每调用一次 GetBaseChunkCandidates，就递增一次 query id。
static size_t cdfe_debug_query_count = 0;
const size_t cdfe_debug_query_id = cdfe_debug_query_count++;

//auto candidate_ids = index_->GetBaseChunkCandidates(feature, topk_candidates);

std::vector<chunk_id> candidate_ids;
{
    auto t = Clock::now();
    candidate_ids = index_->GetBaseChunkCandidates(feature, topk_candidates);
    index_lookup_time_ns_ += ElapsedNs(t);
}

candidate_query_count_++;
candidate_total_count_ += candidate_ids.size();

// 从 CDFESetOrderV2Index 中取出本次 top-k 的子块诊断信息。
// 这些信息由 index/cdfe_setorder_v2_index.cpp 中的
// last_topk_debug_infos_ 保存。
// std::vector<CDFECandidateDebugInfo> topk_debug_infos;
// if (auto *cdfe_index =
//         dynamic_cast<CDFESetOrderV2Index *>(index_.get())) {
//   topk_debug_infos = cdfe_index->GetLastTopKDebugInfos();
// }

std::vector<CDFECandidateDebugInfo> topk_debug_infos;

if (enable_cdfe_debug_log_) {
    if (auto *cdfe_index =
            dynamic_cast<CDFESetOrderV2Index *>(index_.get())) {
        topk_debug_infos = cdfe_index->GetLastTopKDebugInfos();
    }
}


// if (candidate_ids.empty()) {
//   index_->AddFeature(feature, chunk->id());
//   write_base_chunk(chunk);
//   continue;
// }

if (candidate_ids.empty()) {
    {
        auto t = Clock::now();
        index_->AddFeature(feature, chunk->id());
        index_insert_time_ns_ += ElapsedNs(t);
    }
    write_base_chunk(chunk);
    continue;
}


// 对 top-k 候选逐个做真实 delta 编码，选最优 base
std::shared_ptr<Chunk> best_delta_chunk = nullptr;
chunk_id best_base_id = 0;
size_t best_delta_size = std::numeric_limits<size_t>::max();
int best_rank = -1;

struct CandidateDeltaDebug {
  size_t rank;
  chunk_id id;
  size_t delta_size;
  bool valid;

  int query_subblocks = 0;
  int base_subblocks = 0;
  int matched_query_subblocks = 0;
  int matched_base_subblocks = 0;
  int aligned_subblocks = 0;

  float query_matched_ratio = 0.0f;
  float base_matched_ratio = 0.0f;
  float aligned_ratio = 0.0f;
  float jaccard_proxy = 0.0f;
  float order_consistency = 0.0f;
  float score = 0.0f;
};

// std::vector<CandidateDeltaDebug> delta_debug_records;
// delta_debug_records.reserve(candidate_ids.size());

std::vector<CandidateDeltaDebug> delta_debug_records;

if (enable_cdfe_debug_log_) {
    delta_debug_records.reserve(candidate_ids.size());
}

// for (size_t rank = 0; rank < candidate_ids.size(); ++rank) {
//   auto cid = candidate_ids[rank];
//   //auto delta_chunk = storage_->GetDeltaEncodedChunk(chunk, cid);
//   std::shared_ptr<Chunk> delta_chunk;
//   {
//       auto t = Clock::now();
//       delta_chunk = storage_->GetDeltaEncodedChunk(chunk, cid);
//       delta_encode_time_ns_ += ElapsedNs(t);
//   }
//   delta_attempt_count_++;

//   if (delta_chunk) {
//       valid_delta_attempt_count_++;
//   }

//   CandidateDeltaDebug rec;
//   rec.rank = rank;
//   rec.id = cid;
//   rec.delta_size = 0;
//   rec.valid = false;

//   // 将 index 层保存的子块诊断信息合并进来。
//   // 正常情况下 topk_debug_infos[rank].id 应该和 candidate_ids[rank] 一致。
//   if (rank < topk_debug_infos.size() &&
//       topk_debug_infos[rank].id == cid) {
//     const auto &info = topk_debug_infos[rank];

//     rec.query_subblocks = info.query_subblocks;
//     rec.base_subblocks = info.base_subblocks;
//     rec.matched_query_subblocks = info.matched_query_subblocks;
//     rec.matched_base_subblocks = info.matched_base_subblocks;
//     rec.aligned_subblocks = info.aligned_subblocks;

//     rec.query_matched_ratio = info.query_matched_ratio;
//     rec.base_matched_ratio = info.base_matched_ratio;
//     rec.aligned_ratio = info.aligned_ratio;
//     rec.jaccard_proxy = info.jaccard_proxy;
//     rec.order_consistency = info.order_consistency;
//     rec.score = info.score;
//   }

//   if (delta_chunk) {
//     const size_t cur_delta_size =
//         static_cast<size_t>(delta_chunk->len());

//     rec.delta_size = cur_delta_size;
//     rec.valid = true;

//     if (cur_delta_size < best_delta_size) {
//       best_delta_size = cur_delta_size;
//       best_delta_chunk = delta_chunk;
//       best_base_id = cid;
//       best_rank = static_cast<int>(rank);
//     }
//   }

//   delta_debug_records.push_back(rec);
// }


for (size_t rank = 0; rank < candidate_ids.size(); ++rank) {
    auto cid = candidate_ids[rank];

    auto delta_chunk = storage_->GetDeltaEncodedChunk(chunk, cid);

    if (delta_chunk) {
        const size_t cur_delta_size =
            static_cast<size_t>(delta_chunk->len());

        if (cur_delta_size < best_delta_size) {
            best_delta_size = cur_delta_size;
            best_delta_chunk = delta_chunk;
            best_base_id = cid;
            best_rank = static_cast<int>(rank);
        }
    }

    if (enable_cdfe_debug_log_) {
        CandidateDeltaDebug rec;
        rec.rank = rank;
        rec.id = cid;
        rec.delta_size = 0;
        rec.valid = false;

        if (rank < topk_debug_infos.size() &&
            topk_debug_infos[rank].id == cid) {
            const auto &info = topk_debug_infos[rank];

            rec.query_subblocks = info.query_subblocks;
            rec.base_subblocks = info.base_subblocks;
            rec.matched_query_subblocks = info.matched_query_subblocks;
            rec.matched_base_subblocks = info.matched_base_subblocks;
            rec.aligned_subblocks = info.aligned_subblocks;
            rec.query_matched_ratio = info.query_matched_ratio;
            rec.base_matched_ratio = info.base_matched_ratio;
            rec.aligned_ratio = info.aligned_ratio;
            rec.jaccard_proxy = info.jaccard_proxy;
            rec.order_consistency = info.order_consistency;
            rec.score = info.score;
        }

        if (delta_chunk) {
            rec.delta_size = static_cast<size_t>(delta_chunk->len());
            rec.valid = true;
        }

        delta_debug_records.push_back(rec);
    }
}




if (enable_cdfe_debug_log_){
// 旧日志：只记录 top-k 的真实 delta size。
// 这个日志可以继续保留，用于和之前结果对比。
  static std::ofstream cdfe_delta_debug_log(
      "cdfe_topk_delta_debug.log", std::ios::out);

  static bool cdfe_delta_debug_notice = false;
  if (!cdfe_delta_debug_notice) {
    std::cerr << "[CDFE debug] writing topk delta debug log to "
              << "cdfe_topk_delta_debug.log" << std::endl;
    cdfe_delta_debug_notice = true;
  }

  if (cdfe_delta_debug_log.is_open()) {
    cdfe_delta_debug_log
        << "[CDFE delta debug]"
        << " query=" << cdfe_debug_query_id
        << " chunk_id=" << chunk->id()
        << " chunk_len=" << chunk->len()
        << " candidate_count=" << candidate_ids.size()
        << " best_rank=" << best_rank
        << " best_base=" << best_base_id
        << " best_delta_size=" << best_delta_size
        << " fallback_to_base="
        << ((!best_delta_chunk ||
            best_delta_size >= static_cast<size_t>(chunk->len()))
                ? 1
                : 0)
        << " | ";

    for (const auto &rec : delta_debug_records) {
      cdfe_delta_debug_log
          << "rank" << rec.rank
          << "(id=" << rec.id;

      if (rec.valid) {
        cdfe_delta_debug_log
            << ", delta_size=" << rec.delta_size;
      } else {
        cdfe_delta_debug_log
            << ", delta_size=INVALID";
      }

      cdfe_delta_debug_log << ") ";
    }

    cdfe_delta_debug_log << std::endl;
    cdfe_delta_debug_log.flush();
  }

  // 新日志：pair-level 子块划分诊断。
  // 用这个日志判断子块划分是否稳定。
  static std::ofstream cdfe_pair_debug_log(
      "cdfe_pair_partition_debug.log", std::ios::out);

  static bool cdfe_pair_debug_notice = false;
  if (!cdfe_pair_debug_notice) {
    std::cerr << "[CDFE debug] writing pair partition debug log to "
              << "cdfe_pair_partition_debug.log" << std::endl;
    cdfe_pair_debug_notice = true;
  }

  if (cdfe_pair_debug_log.is_open()) {
    cdfe_pair_debug_log
        << "[CDFE pair partition]"
        << " query=" << cdfe_debug_query_id
        << " chunk_id=" << chunk->id()
        << " chunk_len=" << chunk->len()
        << " candidate_count=" << candidate_ids.size()
        << " best_rank=" << best_rank
        << " best_base=" << best_base_id
        << " best_delta_size=" << best_delta_size
        << " best_delta_ratio=";

    if (chunk->len() > 0 &&
        best_delta_size != std::numeric_limits<size_t>::max()) {
      cdfe_pair_debug_log
          << static_cast<double>(best_delta_size) /
                static_cast<double>(chunk->len());
    } else {
      cdfe_pair_debug_log << "NA";
    }

    cdfe_pair_debug_log
        << " fallback_to_base="
        << ((!best_delta_chunk ||
            best_delta_size >= static_cast<size_t>(chunk->len()))
                ? 1
                : 0)
        << " | ";

    for (const auto &rec : delta_debug_records) {
      cdfe_pair_debug_log
          << "rank" << rec.rank
          << "(id=" << rec.id
          << ", q_subblocks=" << rec.query_subblocks
          << ", b_subblocks=" << rec.base_subblocks
          << ", matched_q=" << rec.matched_query_subblocks
          << ", matched_b=" << rec.matched_base_subblocks
          << ", aligned=" << rec.aligned_subblocks
          << ", q_matched_ratio=" << rec.query_matched_ratio
          << ", b_matched_ratio=" << rec.base_matched_ratio
          << ", aligned_ratio=" << rec.aligned_ratio
          << ", jaccard=" << rec.jaccard_proxy
          << ", order=" << rec.order_consistency
          << ", score=" << rec.score;

      if (rec.valid) {
        cdfe_pair_debug_log
            << ", delta_size=" << rec.delta_size;

        if (chunk->len() > 0) {
          cdfe_pair_debug_log
              << ", delta_ratio="
              << static_cast<double>(rec.delta_size) /
                    static_cast<double>(chunk->len());
        } else {
          cdfe_pair_debug_log << ", delta_ratio=NA";
        }
      } else {
        cdfe_pair_debug_log
            << ", delta_size=INVALID"
            << ", delta_ratio=NA";
      }

      cdfe_pair_debug_log << ") ";
    }

    cdfe_pair_debug_log << std::endl;
    cdfe_pair_debug_log.flush();
  }
}
// 如果最优 delta 仍然不划算，则退回写 base chunk
// if (!best_delta_chunk ||
//     best_delta_size >= static_cast<size_t>(chunk->len())) {
//   index_->AddFeature(feature, chunk->id());
//   write_base_chunk(chunk);
//   continue;
// }

if (!best_delta_chunk || best_delta_size >= static_cast<size_t>(chunk->len())) {
    {
        auto t = Clock::now();
        index_->AddFeature(feature, chunk->id());
        index_insert_time_ns_ += ElapsedNs(t);
    }
    write_base_chunk(chunk);
    continue;
}

write_delta_chunk(chunk, best_delta_chunk, best_base_id);

    //**************** */
    file_meta.end_chunk_id = chunk->id();
  }
  //file_meta_writer_.Write(file_meta);
  {
    auto t = Clock::now();
    file_meta_writer_.Write(file_meta);
    file_meta_write_time_ns_ += ElapsedNs(t);
  }
}

// DeltaCompression::~DeltaCompression() {
//   auto print_ratio = [](size_t a, size_t b) {
//     double ratio = (double)a / (double)b;
//     std::cout << std::fixed << std::setprecision(1);
//     std::cout << "(" << ratio * 100 << "%)" << std::endl;
//     std::cout << std::defaultfloat;
//   };
//   uint32_t chunk_count =
//       base_chunk_count_ + delta_chunk_count_ + duplicate_chunk_count_;
//   std::cout << "Total chunk count: " << chunk_count << std::endl;
//   std::cout << "Base chunk count: " << base_chunk_count_;
//   print_ratio(base_chunk_count_, chunk_count);
//   std::cout << "Delta chunk count: " << delta_chunk_count_;
//   print_ratio(delta_chunk_count_, chunk_count);
//   std::cout << "Duplicate chunk count: " << duplicate_chunk_count_;
//   print_ratio(duplicate_chunk_count_, chunk_count);
//   std::cout << "DCR (Delta Compression Ratio): ";
//   print_ratio(total_size_origin_, total_size_compressed_);
//   std::cout << "before " << total_size_origin_
//             << " after: " << total_size_compressed_ << std::endl;
//   std::cout << "DCE (Delta Compression Efficiency): ";
//   print_ratio(chunk_size_after_delta_, chunk_size_before_delta_);
// }

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

    // 手动释放，让 CDFE 子块统计、Storage cache 统计先打印出来。
    // 这样最后打印 Time Stats，终端末尾更清楚。
    feature_.reset();
    index_.reset();
    storage_.reset();
    dedup_.reset();

    const uint64_t total_ns = ElapsedNs(total_time_start_);

    const uint64_t accounted_ns =
        file_reinit_time_ns_ +
        chunking_time_ns_ +
        dedup_time_ns_ +
        duplicate_write_time_ns_ +
        feature_time_ns_ +
        index_lookup_time_ns_ +
        index_insert_time_ns_ +
        delta_encode_time_ns_ +
        base_write_time_ns_ +
        delta_write_time_ns_ +
        file_meta_write_time_ns_ +
        debug_log_time_ns_;

    const uint64_t other_ns =
        total_ns > accounted_ns ? total_ns - accounted_ns : 0;

    std::cout << "\n[Time Stats]" << std::endl;
    PrintTimeLine("Total runtime", total_ns, total_ns);
    PrintTimeLine("File reinit/map", file_reinit_time_ns_, total_ns);
    PrintTimeLine("Chunking", chunking_time_ns_, total_ns);
    PrintTimeLine("Dedup", dedup_time_ns_, total_ns);
    PrintTimeLine("Duplicate write", duplicate_write_time_ns_, total_ns);
    PrintTimeLine("Feature extraction", feature_time_ns_, total_ns);
    PrintTimeLine("Index lookup", index_lookup_time_ns_, total_ns);
    PrintTimeLine("Index insert", index_insert_time_ns_, total_ns);
    PrintTimeLine("Delta encode + base fetch", delta_encode_time_ns_, total_ns);
    PrintTimeLine("Base write", base_write_time_ns_, total_ns);
    PrintTimeLine("Delta write", delta_write_time_ns_, total_ns);
    PrintTimeLine("File meta write", file_meta_write_time_ns_, total_ns);
    PrintTimeLine("Debug log write", debug_log_time_ns_, total_ns);
    PrintTimeLine("Other / loop overhead", other_ns, total_ns);

    std::cout << "\n[Candidate Stats]" << std::endl;
    std::cout << "Candidate query count: " << candidate_query_count_ << std::endl;
    std::cout << "Total returned candidates: " << candidate_total_count_ << std::endl;

    if (candidate_query_count_ > 0) {
        std::cout << "Average candidates per query: "
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(candidate_total_count_) /
                     static_cast<double>(candidate_query_count_)
                  << std::defaultfloat << std::endl;
    }

    std::cout << "Delta encode attempts: " << delta_attempt_count_ << std::endl;
    std::cout << "Valid delta encode attempts: "
              << valid_delta_attempt_count_ << std::endl;

    if (delta_chunk_count_ > 0) {
        std::cout << "Delta attempts per stored delta chunk: "
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(delta_attempt_count_) /
                     static_cast<double>(delta_chunk_count_)
                  << std::defaultfloat << std::endl;
    }

    std::cout << std::endl;
}



//统计时间
uint64_t DeltaCompression::ElapsedNs(TimePoint start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start).count());
}

void DeltaCompression::PrintTimeLine(const std::string &name,
                                     uint64_t ns,
                                     uint64_t total_ns) {
    const double sec = static_cast<double>(ns) / 1e9;
    const double pct = total_ns > 0
        ? static_cast<double>(ns) * 100.0 / static_cast<double>(total_ns)
        : 0.0;

    std::cout << std::left << std::setw(32) << name
              << std::right << std::fixed << std::setprecision(3)
              << sec << " s"
              << "  (" << std::setprecision(1) << pct << "%)"
              << std::defaultfloat << std::endl;
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
      static_cast<int>(feature->get_as<int64_t>("min_subblock_size").value_or(256));

    params.avg_subblock_size =
        static_cast<int>(feature->get_as<int64_t>("avg_subblock_size").value_or(512));

    params.max_subblock_size =
        static_cast<int>(feature->get_as<int64_t>("max_subblock_size").value_or(1024));

    params.boundary_mask =
        static_cast<uint64_t>(feature->get_as<int64_t>("boundary_mask").value_or(511));

    params.split_window_size =
        static_cast<int>(feature->get_as<int64_t>("split_window_size").value_or(32));

    params.feature_window_size =
        static_cast<int>(feature->get_as<int64_t>("feature_window_size").value_or(16));

    // 固定为 1；保留读取也可以，但不再用于循环生成多个 feature
    params.local_features_per_subblock = 1;

  size_t topk_candidates =
      static_cast<size_t>(
          feature->get_as<int64_t>("topk_candidates").value_or(4));
      int min_matched_subblocks =
          feature->get_as<int64_t>("min_matched_subblocks").value_or(4);
      int min_aligned_subblocks =
          feature->get_as<int64_t>("min_aligned_subblocks").value_or(2);
      float min_jaccard_proxy =
          static_cast<float>(
              feature->get_as<double>("min_jaccard_proxy").value_or(0.0));
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
      min_jaccard_proxy,
      pos_tolerance,
      hot_posting_limit,
      order_lambda);

  LOG(INFO) << "Add CDFE-SetOrder-v2 feature extractor"
            << " min_subblock_size=" << params.min_subblock_size
            << " avg_subblock_size=" << params.avg_subblock_size
            << " max_subblock_size=" << params.max_subblock_size
            << " boundary_mask=" << params.boundary_mask
            << " split_window_size=" << params.split_window_size
            << " feature_window_size=" << params.feature_window_size
            << " local_features_per_subblock=1"
            << " topk_candidates=" << topk_candidates
            << " min_jaccard_proxy=" << min_jaccard_proxy;
} else {
  if (!feature_index_map.count(feature_type))
    LOG(FATAL) << "Unknown feature type " << feature_type;

  auto [feature_ptr, index_ptr] = feature_index_map[feature_type]();
  this->feature_ = std::move(feature_ptr);
  this->index_ = std::move(index_ptr);
}        


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