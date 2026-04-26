#include "index/cdfe_setorder_v2_index.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

//增加了 子块诊断信息的保存能力
namespace Delta {

int CDFESetOrderV2Index::CountSubblocks(
    const CDFESetOrderFeature &features) {
  if (features.empty()) {
    return 0;
  }

  uint16_t max_rank = 0;
  for (const auto &f : features) {
    if (f.subblock_rank > max_rank) {
      max_rank = f.subblock_rank;
    }
  }

  return static_cast<int>(max_rank) + 1;
}

void CDFESetOrderV2Index::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<CDFESetOrderFeature>(feat);

  // 记录每个进入索引的 base chunk 的子块数量。
  // 后续 pair-level 诊断时，需要知道 candidate/base 被切成了多少子块。
  chunk_subblock_count_[id] = CountSubblocks(features);

  for (const auto &f : features) {
    auto &plist = inverted_[f.value];
    plist.push_back(CDFEPosting{id, f.subblock_rank, f.norm_pos});

    // 当前策略：posting list 只保留最近 hot_posting_limit_ 条。
    // 注意：由于这里已经截断，查询阶段 plist.size() > hot_posting_limit_
    // 通常不会触发。也就是说当前更像是 posting cap，而不是真正 stop-feature。
    if (plist.size() > hot_posting_limit_) {
      plist.erase(plist.begin(),
                  plist.begin() + (plist.size() - hot_posting_limit_));
    }
  }
}

float CDFESetOrderV2Index::ComputeOrderConsistency(
    const std::vector<std::pair<uint16_t, uint16_t>> &matched_ranks) const {
  if (matched_ranks.empty()) {
    return 0.0f;
  }

  auto pairs = matched_ranks;
  std::sort(pairs.begin(), pairs.end(),
            [](const auto &a, const auto &b) {
              if (a.first != b.first) return a.first < b.first;
              return a.second < b.second;
            });

  std::vector<uint16_t> seq;
  seq.reserve(pairs.size());
  for (const auto &p : pairs) {
    seq.push_back(p.second);
  }

  // LIS on candidate ranks
  std::vector<uint16_t> tails;
  for (auto x : seq) {
    auto it = std::lower_bound(tails.begin(), tails.end(), x);
    if (it == tails.end()) {
      tails.push_back(x);
    } else {
      *it = x;
    }
  }

  return static_cast<float>(tails.size()) /
         static_cast<float>(seq.size());
}

std::vector<chunk_id>
CDFESetOrderV2Index::GetBaseChunkCandidates(const Feature &feat,
                                            size_t topk) {
  last_topk_debug_infos_.clear();

  const auto &features = std::get<CDFESetOrderFeature>(feat);
  const int query_subblock_count = CountSubblocks(features);

  // 调试统计
  const size_t query_feature_count = features.size();
  size_t feature_not_found_count = 0;
  size_t hot_feature_skipped_count = 0;
  size_t posting_scanned_count = 0;

  std::unordered_map<chunk_id, CandidateStat> stats;

  for (const auto &qf : features) {
    auto it = inverted_.find(qf.value);
    if (it == inverted_.end()) {
      feature_not_found_count++;
      continue;
    }

    const auto &plist = it->second;

    // 当前由于 AddFeature 中已经做了 posting cap，
    // 这里一般不会触发，但先保留统计，方便后面改成真正 hot-feature skip。
    if (plist.size() > hot_posting_limit_) {
      hot_feature_skipped_count++;
      continue;
    }

    posting_scanned_count += plist.size();

    for (const auto &posting : plist) {
      auto &stat = stats[posting.id];

      // 按 query subblock 去重计票
      stat.matched_query_subblocks.insert(qf.subblock_rank);

      // base/candidate 侧命中
      stat.matched_base_subblocks.insert(posting.subblock_rank);

      // 位置接近时，记录 aligned query subblock
      if (std::fabs(qf.norm_pos - posting.norm_pos) <= pos_tolerance_) {
        stat.aligned_query_subblocks.insert(qf.subblock_rank);
      }

      // 顺序一致性计算需要 rank 对
      if (stat.matched_ranks.size() < 64) {
        stat.matched_ranks.emplace_back(qf.subblock_rank,
                                        posting.subblock_rank);
      }
    }
  }

  // struct ScoredCandidate {
  //   chunk_id id;
  //   int matched_subblocks;
  //   int aligned_subblocks;
  //   float order_consistency;
  //   float score;
  // };

  struct ScoredCandidate {
  chunk_id id;

  int matched_query_subblocks;
  int matched_base_subblocks;
  int aligned_subblocks;

  int query_subblock_count;
  int base_subblock_count;

  float query_matched_ratio;
  float base_matched_ratio;
  float aligned_ratio;
  float jaccard_proxy;

  float order_consistency;
  float score;
};

  std::vector<ScoredCandidate> scored;
  scored.reserve(stats.size());

  // for (auto &[cid, stat] : stats) {
  //   const int matched_subblocks =
  //       static_cast<int>(stat.matched_query_subblocks.size());
  //   const int aligned_subblocks =
  //       static_cast<int>(stat.aligned_query_subblocks.size());

  //   // 硬过滤：至少覆盖多少个 query subblock
  //   if (matched_subblocks < min_matched_subblocks_) {
  //     continue;
  //   }

  //   // 如果 min_aligned_subblocks_ = 0，则这一项相当于关闭。
  //   if (aligned_subblocks < min_aligned_subblocks_) {
  //     continue;
  //   }

  //   const float order_consistency =
  //       ComputeOrderConsistency(stat.matched_ranks);

  //   // 当前默认保留你上传文件中的 matched-only 评分。
  //   // 如果要使用当前实验最好的组合评分，把下面这行替换为注释中的组合 score。
  //   const float score = static_cast<float>(matched_subblocks);

  //   // 组合 score 版本：
  //   // const float score =
  //   //     6.0f * static_cast<float>(matched_subblocks) +
  //   //     3.0f * static_cast<float>(aligned_subblocks) +
  //   //     4.0f * order_consistency;

  //   scored.push_back({
  //       cid,
  //       matched_subblocks,
  //       aligned_subblocks,
  //       order_consistency,
  //       score
  //   });
  // }

  for (auto &[cid, stat] : stats) {
  const int matched_query_subblocks =
      static_cast<int>(stat.matched_query_subblocks.size());
  const int matched_base_subblocks =
      static_cast<int>(stat.matched_base_subblocks.size());
  const int aligned_subblocks =
      static_cast<int>(stat.aligned_query_subblocks.size());

  int base_subblock_count = 0;
  auto cnt_it = chunk_subblock_count_.find(cid);
  if (cnt_it != chunk_subblock_count_.end()) {
    base_subblock_count = cnt_it->second;
  }

  const float query_matched_ratio =
      query_subblock_count > 0
          ? static_cast<float>(matched_query_subblocks) /
                static_cast<float>(query_subblock_count)
          : 0.0f;

  const float base_matched_ratio =
      base_subblock_count > 0
          ? static_cast<float>(matched_base_subblocks) /
                static_cast<float>(base_subblock_count)
          : 0.0f;

  const float aligned_ratio =
      matched_query_subblocks > 0
          ? static_cast<float>(aligned_subblocks) /
                static_cast<float>(matched_query_subblocks)
          : 0.0f;

  // Jaccard proxy：
  // 由于命中是多对多关系，这里使用 min(query覆盖数, base覆盖数)
  // 作为 intersection 的保守近似。
  const int intersection_proxy =
      std::min(matched_query_subblocks, matched_base_subblocks);

  const int union_proxy =
      query_subblock_count + base_subblock_count - intersection_proxy;

  const float jaccard_proxy =
      union_proxy > 0
          ? static_cast<float>(intersection_proxy) /
                static_cast<float>(union_proxy)
          : 0.0f;

  // 绝对覆盖阈值
  if (matched_query_subblocks < min_matched_subblocks_) {
    continue;
  }

  // aligned 硬阈值，默认 0 等于关闭
  if (aligned_subblocks < min_aligned_subblocks_) {
    continue;
  }

  // Jaccard proxy 阈值，默认 0 等于关闭
  if (jaccard_proxy < min_jaccard_proxy_) {
    continue;
  }

  const float order_consistency =
      ComputeOrderConsistency(stat.matched_ranks);

  // 当前建议保留你实验较好的组合 score
  const float score = jaccard_proxy;
  // const float score = 
  //     6.0f * static_cast<float>(matched_query_subblocks) +
  //     3.0f * static_cast<float>(aligned_subblocks) +
  //     4.0f * order_consistency;

  scored.push_back({
      cid,
      matched_query_subblocks,
      matched_base_subblocks,
      aligned_subblocks,
      query_subblock_count,
      base_subblock_count,
      query_matched_ratio,
      base_matched_ratio,
      aligned_ratio,
      jaccard_proxy,
      order_consistency,
      score
  });
}

  std::sort(scored.begin(), scored.end(),
            [](const auto &a, const auto &b) {
              if (a.score != b.score) return a.score > b.score;
              return a.id < b.id;
            });

  std::vector<chunk_id> result;
  const size_t limit = std::min(topk, scored.size());
  result.reserve(limit);

  // 保存本次 top-k 的子块诊断信息，供 delta_compression.cpp 读取。
  last_topk_debug_infos_.clear();
  last_topk_debug_infos_.reserve(limit);

  for (size_t i = 0; i < limit; ++i) {
    const auto &cand = scored[i];

    int base_subblock_count = 0;
    auto cnt_it = chunk_subblock_count_.find(cand.id);
    if (cnt_it != chunk_subblock_count_.end()) {
      base_subblock_count = cnt_it->second;
    }

    CDFECandidateDebugInfo info;

    // info.id = cand.id;
    // info.query_subblocks = query_subblock_count;
    // info.base_subblocks = base_subblock_count;
    // info.matched_subblocks = cand.matched_subblocks;
    // info.aligned_subblocks = cand.aligned_subblocks;
    // info.order_consistency = cand.order_consistency;
    // info.score = cand.score;

    // if (query_subblock_count > 0) {
    //   info.matched_ratio =
    //       static_cast<float>(cand.matched_subblocks) /
    //       static_cast<float>(query_subblock_count);
    // }

    // if (cand.matched_subblocks > 0) {
    //   info.aligned_ratio =
    //       static_cast<float>(cand.aligned_subblocks) /
    //       static_cast<float>(cand.matched_subblocks);
    // }

    info.id = cand.id;
    info.query_subblocks = cand.query_subblock_count;
    info.base_subblocks = cand.base_subblock_count;

    info.matched_query_subblocks = cand.matched_query_subblocks;
    info.matched_base_subblocks = cand.matched_base_subblocks;
    info.aligned_subblocks = cand.aligned_subblocks;

    info.query_matched_ratio = cand.query_matched_ratio;
    info.base_matched_ratio = cand.base_matched_ratio;
    info.aligned_ratio = cand.aligned_ratio;
    info.jaccard_proxy = cand.jaccard_proxy;

    info.order_consistency = cand.order_consistency;
    info.score = cand.score;

    last_topk_debug_infos_.push_back(info);
    result.push_back(cand.id);
  }

  // 继续保留原来的 top-k 调试日志。
  // 这个日志只输出 topk > 0 的 query。
  static size_t debug_query_count = 0;
  static std::ofstream debug_log("cdfe_topk_debug.log", std::ios::out);

  const size_t debug_query_id = debug_query_count++;

  static bool debug_log_notice = false;
  if (!debug_log_notice) {
    std::cerr << "[CDFE debug] writing topk debug log to "
              << "cdfe_topk_debug.log" << std::endl;
    debug_log_notice = true;
  }

  if (limit > 0 && debug_log.is_open()) {
    debug_log << "[CDFE topk debug]"
              << " query=" << debug_query_id
              << " query_features=" << query_feature_count
              << " query_subblocks=" << query_subblock_count
              << " feature_not_found=" << feature_not_found_count
              << " hot_feature_skipped=" << hot_feature_skipped_count
              << " posting_scanned=" << posting_scanned_count
              << " raw_candidates=" << stats.size()
              << " filtered_candidates=" << scored.size()
              << " topk=" << limit
              << " | ";

    for (size_t i = 0; i < limit; ++i) {
      const auto &info = last_topk_debug_infos_[i];

      debug_log << "rank" << i
                << "(id=" << info.id
                << ", q_subblocks=" << info.query_subblocks
                << ", b_subblocks=" << info.base_subblocks
                << ", matched_q=" << info.matched_query_subblocks
                << ", matched_b=" << info.matched_base_subblocks
                << ", aligned=" << info.aligned_subblocks
                << ", q_matched_ratio=" << info.query_matched_ratio
                << ", b_matched_ratio=" << info.base_matched_ratio
                << ", aligned_ratio=" << info.aligned_ratio
                << ", jaccard=" << info.jaccard_proxy
                << ", order=" << info.order_consistency
                << ", score=" << info.score
                << ") ";
    }

    debug_log << std::endl;
    debug_log.flush();
  }

  return result;
}

std::optional<chunk_id>
CDFESetOrderV2Index::GetBaseChunkID(const Feature &feat) {
  auto cands = GetBaseChunkCandidates(feat, 1);
  if (cands.empty()) {
    return std::nullopt;
  }
  return cands.front();
}

} // namespace Delta


