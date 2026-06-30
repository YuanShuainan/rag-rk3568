#pragma once

#include "types.h"
#include "bm25_index.h"
#include "tokenizer.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace rag {

class Embedding;  // forward decl avoids leaking ONNX headers here

// Two-stage retrieval pipeline:
//   Stage 1: BM25 coarse retrieval (top-K candidates)
//   Stage 2: Embedding dense re-ranking (top-N final)
// Merged score = 0.4*BM25 + 0.6*cosine + section_bonus + density_bonus
class HybridRetriever {
public:
    struct Config {
        int bm25_top_k   = 20;   // coarse candidates
        int final_top_n  = 3;    // final chunks to feed LLM
    };

    HybridRetriever() = default;

    void build_index(const std::vector<Chunk>& chunks,
                     const std::vector<std::vector<std::string>>& tokenized_chunks,
                     Embedding* embedding);

    // Retrieve top-N chunks for a query.
    std::vector<RetrievedChunk> retrieve(const std::string& query);

private:
    struct ScoredChunk {
        int   chunk_id;
        float bm25_score;
        float dense_score;
        float final_score;
    };

    float density_bonus(const Chunk& chunk) const;
    void  compute_section_hits(const std::vector<std::string>& query_terms);
    float rerank_score(int chunk_id, const std::vector<float>& query_vec);

    Config config_;
    std::vector<Chunk>                chunks_;
    std::vector<std::vector<float>>   chunk_embeddings_;  // L2-normalized
    std::unordered_map<int, bool>     section_hit_;
    std::unordered_map<int, float>    bm25_scores_;       // per-chunk BM25 scores
    std::unique_ptr<BM25Index>        bm25_;
    Embedding*                        embedding_ = nullptr;
    Tokenizer                         tokenizer_;
};

} // namespace rag
