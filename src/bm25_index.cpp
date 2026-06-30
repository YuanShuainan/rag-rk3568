#include "rag/bm25_index.h"

#include <cmath>
#include <algorithm>

namespace rag {

void BM25Index::build(const std::vector<Chunk>& chunks,
                       const std::vector<std::vector<std::string>>& tokenized_chunks) {
    N_ = (int)chunks.size();
    if (N_ == 0) return;

    inverted_.clear();
    doc_len_norm_.resize(N_);

    float total_len = 0.0f;

    for (int i = 0; i < N_; ++i) {
        const auto& terms = tokenized_chunks[i];
        doc_len_norm_[i] = (float)terms.size();
        total_len += doc_len_norm_[i];

        // Count term frequencies in this chunk
        std::unordered_map<std::string, float> tf_map;
        for (const auto& t : terms) {
            tf_map[t] += 1.0f;
        }

        for (const auto& [term, tf] : tf_map) {
            inverted_[term].push_back({i, tf});
        }
    }

    avg_len_ = (total_len > 0.0f) ? total_len / N_ : 1.0f;

    // Normalize length: doc_len_norm_[i] / avg_len_
    for (int i = 0; i < N_; ++i) {
        doc_len_norm_[i] /= avg_len_;
    }
}

float BM25Index::score(const std::vector<TermMatch>& matches, int chunk_id) const {
    float s = 0.0f;
    for (const auto& m : matches) {
        // IDF
        float df = (float)m.df;
        float idf = std::log((N_ - df + 0.5f) / (df + 0.5f) + 1.0f);

        // TF (saturated)
        float tf_raw = (chunk_id < (int)m.tf.size()) ? m.tf[chunk_id] : 0.0f;
        float tf = (tf_raw * (k1_ + 1.0f)) /
                   (tf_raw + k1_ * (1.0f - b_ + b_ * doc_len_norm_[chunk_id]));

        s += idf * tf;
    }
    return s;
}

std::vector<std::pair<int, float>> BM25Index::search(
    const std::vector<std::string>& query_terms, int top_k) const {

    // Build TermMatch list from query terms
    std::vector<TermMatch> matches;
    matches.reserve(query_terms.size());

    for (const auto& term : query_terms) {
        auto it = inverted_.find(term);
        if (it == inverted_.end()) continue;

        TermMatch m;
        m.text = term;
        m.df   = (int)it->second.size();
        m.tf.resize(N_, 0.0f);
        for (const auto& [chunk_id, tf] : it->second) {
            m.tf[chunk_id] = tf;
        }
        matches.push_back(std::move(m));
    }

    if (matches.empty()) return {};

    // Score all chunks
    std::vector<std::pair<int, float>> scored;
    scored.reserve(N_);

    for (int i = 0; i < N_; ++i) {
        float s = score(matches, i);
        if (s > 0.0f) {
            scored.push_back({i, s});
        }
    }

    // Sort descending by score, keep top_k
    std::partial_sort(scored.begin(),
                       scored.begin() + std::min(top_k, (int)scored.size()),
                       scored.end(),
                       [](const auto& a, const auto& b) { return a.second > b.second; });

    if ((int)scored.size() > top_k) {
        scored.resize(top_k);
    }

    return scored;
}

} // namespace rag
