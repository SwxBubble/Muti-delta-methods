#include "index/cdfe_setorder_v2_index.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <iostream>

namespace Delta {

void CDFESetOrderV2Index::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<CDFESetOrderFeature>(feat);

  for (const auto &f : features) {
    auto &plist = inverted_[f.value];
    plist.push_back(CDFEPosting{id, f.subblock_rank, f.norm_pos});

    // 简单 posting cap：先保留最新的 hot_posting_limit_ 条
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
  static size_t debug_enter_count = 0;
  if (debug_enter_count < 20) {
    std::cerr << "[CDFE debug enter] GetBaseChunkCandidates called, count="
              << debug_enter_count << ", topk=" << topk << std::endl;
  }
  debug_enter_count++;                                            

  const auto &features = std::get<CDFESetOrderFeature>(feat);

  std::unordered_map<chunk_id, CandidateStat> stats;

  for (const auto &qf : features) {
    auto it = inverted_.find(qf.value);
    if (it == inverted_.end()) {
      continue;
    }

    const auto &plist = it->second;

    // 关键修改 1：查询时直接跳过热特征
    if (plist.size() > hot_posting_limit_) {
      continue;
    }

    for (const auto &posting : plist) {
      auto &stat = stats[posting.id];

      // 关键修改 2：按 query subblock 去重计票
      stat.matched_query_subblocks.insert(qf.subblock_rank);

      // 关键修改 3：位置接近时，再记 aligned 的 query subblock
      if (std::fabs(qf.norm_pos - posting.norm_pos) <= pos_tolerance_) {
        stat.aligned_query_subblocks.insert(qf.subblock_rank);
      }

      // 顺序一致性仍然需要记录 rank 对
      if (stat.matched_ranks.size() < 64) {
        stat.matched_ranks.emplace_back(qf.subblock_rank,
                                        posting.subblock_rank);
      }
    }
  }

  // struct ScoredCandidate {
  //   chunk_id id;
  //   float score;
  // };

  struct ScoredCandidate {
  chunk_id id;
  int matched_subblocks;
  int aligned_subblocks;
  float order_consistency;
  float score;
  };

  std::vector<ScoredCandidate> scored;
  scored.reserve(stats.size());

  for (auto &[cid, stat] : stats) {
    const int matched_subblocks =
        static_cast<int>(stat.matched_query_subblocks.size());
    const int aligned_subblocks =
        static_cast<int>(stat.aligned_query_subblocks.size());

    // 关键修改 4：硬过滤，不再只看 raw overlap
    if (matched_subblocks < min_matched_subblocks_) {
      continue;
    }
    if (aligned_subblocks < min_aligned_subblocks_) {
      continue;
    }

    const float order_consistency =
        ComputeOrderConsistency(stat.matched_ranks);

    // 关键修改 5：score 改成按 query subblock 计票
    
        // 5.0f * static_cast<float>(matched_subblocks) +
        // 3.0f * static_cast<float>(aligned_subblocks) +
        // order_lambda_ * order_consistency;

    // scored.push_back({cid, score});

    const float score = static_cast<float>(matched_subblocks);

      scored.push_back({
          cid,
          matched_subblocks,
          aligned_subblocks,
          order_consistency,
          score
      });

  }

  std::sort(scored.begin(), scored.end(),
            [](const auto &a, const auto &b) {
              if (a.score != b.score) return a.score > b.score;
              return a.id < b.id;
            });

  // std::vector<chunk_id> result;
  // const size_t limit = std::min(topk, scored.size());
  // result.reserve(limit);

  // for (size_t i = 0; i < limit; ++i) {
  //   result.push_back(scored[i].id);
  // }

  // return result;

  std::vector<chunk_id> result;
  const size_t limit = std::min(topk, scored.size());
  result.reserve(limit);

  // 只打印前 100 个 query，避免日志爆炸
  static size_t debug_query_count = 0;
  if (debug_query_count < 100) {
    std::cerr << "[CDFE topk debug] query=" << debug_query_count
              << " filtered_candidates=" << scored.size()
              << " topk=" << limit
              << " | ";

    for (size_t i = 0; i < limit; ++i) {
      std::cerr << "rank" << i
                << "(id=" << scored[i].id
                << ", matched=" << scored[i].matched_subblocks
                << ", aligned=" << scored[i].aligned_subblocks
                << ", order=" << scored[i].order_consistency
                << ", score=" << scored[i].score
                << ") ";
    }

    std::cerr << std::endl;
  }
  debug_query_count++;

  for (size_t i = 0; i < limit; ++i) {
    result.push_back(scored[i].id);
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