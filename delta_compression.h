#pragma once
#include "chunk/chunker.h"
#include "encoder/encoder.h"
#include "index/index.h"
#include "storage/file_meta.h"
#include "storage/storage.h"
#include "dedup/dedup.h"
#include <memory>
#include <string>

#include <chrono>
#include <cstdint>

namespace Delta {
class DeltaCompression {
public:
  DeltaCompression();
  ~DeltaCompression();
  virtual void AddFile(const std::string &file_name);

protected:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  static uint64_t ElapsedNs(TimePoint start);
  static void PrintTimeLine(const std::string &name,
                            uint64_t ns,
                            uint64_t total_ns);

  TimePoint total_time_start_ = Clock::now();

  uint64_t file_reinit_time_ns_ = 0;
  uint64_t chunking_time_ns_ = 0;
  uint64_t dedup_time_ns_ = 0;
  uint64_t duplicate_write_time_ns_ = 0;
  uint64_t feature_time_ns_ = 0;
  uint64_t index_lookup_time_ns_ = 0;
  uint64_t index_insert_time_ns_ = 0;
  uint64_t delta_encode_time_ns_ = 0;
  uint64_t base_write_time_ns_ = 0;
  uint64_t delta_write_time_ns_ = 0;
  uint64_t file_meta_write_time_ns_ = 0;
  uint64_t debug_log_time_ns_ = 0;

  size_t delta_attempt_count_ = 0;
  size_t valid_delta_attempt_count_ = 0;
  size_t candidate_query_count_ = 0;
  size_t candidate_total_count_ = 0;


  std::string out_data_path_;
  std::string out_meta_path_;
  std::string index_path_;

  bool enable_cdfe_debug_log_ = false;

  std::unique_ptr<Chunker> chunker_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<Dedup> dedup_;
  std::unique_ptr<Storage> storage_;
  std::unique_ptr<FeatureCalculator> feature_;

  FileMetaWriter file_meta_writer_;

  uint32_t base_chunk_count_ = 0;
  uint32_t delta_chunk_count_ = 0;
  uint32_t duplicate_chunk_count_ = 0;

  size_t total_size_origin_ = 0;
  size_t total_size_compressed_ = 0;
  size_t chunk_size_before_delta_ = 0;
  size_t chunk_size_after_delta_ = 0;
};
} // namespace Delta