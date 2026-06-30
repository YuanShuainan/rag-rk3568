#pragma once

#include "types.h"
#include "tokenizer.h"
#include "doc_loader.h"
#include "bm25_index.h"
#include "embedding.h"
#include "hybrid_retriever.h"
#include "context_builder.h"
#include "kv_cache.h"
#include "llm_inference.h"

#include <string>
#include <vector>
#include <memory>

namespace rag {

// Top-level RAG pipeline orchestrator.
// Owns all subsystem instances; exposes simple query/followup API.
class RAGPipeline {
public:
    RAGPipeline() = default;

    // Initialize all subsystems. Must be called once before query().
    bool initialize(const RAGConfig& cfg);

    // First query in a conversation or standalone.
    std::string query(const std::string& question);

    // Follow-up query. Preserves chat history, reuses KV cache pages.
    std::string followup(const std::string& question);

    // Access to subsystems (for testing).
    HybridRetriever* retriever() { return &hybrid_retriever_; }
    PageManager*     page_mgr()  { return &page_mgr_; }
    const std::vector<Chunk>& chunks() const { return chunks_; }
    std::size_t num_chunks() const { return chunks_.size(); }

    bool is_ready() const { return initialized_; }

private:
    std::string run_inference(const std::string& prompt,
                              std::vector<int>* out_page_ids = nullptr);

    RAGConfig        config_;
    bool             initialized_ = false;

    DocLoader        doc_loader_;
    Tokenizer        tokenizer_;
    Embedding        embedding_;
    BM25Index        bm25_;
    HybridRetriever  hybrid_retriever_;
    ContextBuilder   context_builder_;
    PageManager      page_mgr_;
    LLMInference     llm_;

    std::vector<ChatTurn>          history_;
    std::vector<Chunk>             chunks_;
    std::vector<std::vector<std::string>> tokenized_chunks_;
    std::vector<int>               sys_block_ids_;  // pinned system-prompt blocks
};

} // namespace rag
