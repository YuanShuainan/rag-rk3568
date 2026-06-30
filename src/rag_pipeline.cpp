#include "rag/rag_pipeline.h"

#include "rag/logger.h"

#include <algorithm>

namespace rag {

bool RAGPipeline::initialize(const RAGConfig& cfg) {
    config_ = cfg;

    // 1. Load document and chunk
    SPDLOG_info("Loading document: {}", cfg.doc_path);
    chunks_ = doc_loader_.load(cfg.doc_path);
    if (chunks_.empty()) {
        SPDLOG_error("No chunks loaded from document");
        return false;
    }
    SPDLOG_info("Loaded {} chunks from document", chunks_.size());

    // 2. Load tokenizer vocab
    if (!cfg.vocab_path.empty()) {
        SPDLOG_info("Loading vocabulary: {}", cfg.vocab_path);
        if (!tokenizer_.load_vocab(cfg.vocab_path)) {
            SPDLOG_warn("Failed to load vocab, using character-level fallback");
        }
    }

    // 3. Tokenize all chunks for BM25
    SPDLOG_info("Tokenizing chunks...");
    tokenized_chunks_.clear();
    tokenized_chunks_.reserve(chunks_.size());
    for (const auto& c : chunks_) {
        auto tokens = tokenizer_.tokenize(c.text);
        std::vector<std::string> term_strings;
        term_strings.reserve(tokens.size());
        for (const auto& t : tokens) {
            term_strings.push_back(t.text);
        }
        tokenized_chunks_.push_back(std::move(term_strings));
    }

    // 4. Initialize embedding model
    SPDLOG_info("Initializing embedding model...");
    Embedding::Config emb_cfg;
    emb_cfg.model_path = cfg.onnx_model_path;
    if (!embedding_.init(emb_cfg)) {
        SPDLOG_warn("ONNX embedding model not available, using BM25-only retrieval");
    }

    // 5. Build hybrid retriever index
    SPDLOG_info("Building retrieval index...");
    hybrid_retriever_.build_index(chunks_, tokenized_chunks_,
                                   embedding_.is_ready() ? &embedding_ : nullptr);

    // 6. Initialize LLM
    SPDLOG_info("Initializing rknn-llm...");
    LLMInference::Config llm_cfg;
    llm_cfg.model_path     = cfg.rknn_model_path;
    llm_cfg.max_context    = cfg.max_context;
    llm_cfg.max_new_tokens = cfg.max_new_tokens;
    if (!llm_.init(llm_cfg, &tokenizer_)) {
        SPDLOG_error("Failed to initialize rknn-llm");
        return false;
    }

    // 7. Pre-allocate system prompt blocks and pin them so they are
    //    preserved across turns (KV cache prefix reuse).
    {
        std::string sys_prompt = ContextBuilder::system_prompt();
        auto sys_tokens = tokenizer_.encode(sys_prompt);
        if (!sys_tokens.empty()) {
            auto sys_blocks = page_mgr_.map_tokens_to_blocks(sys_tokens);
            for (int bid : sys_blocks) {
                if (bid >= 0) {
                    page_mgr_.pin_block(bid);
                    // The system prompt is part of the persistent prefix,
                    // so hold a reference on behalf of the pipeline.
                    sys_block_ids_.push_back(bid);
                }
            }
            // The prefill for the system prompt happens during the first
            // query; for now we just reserve the blocks.
        }
    }

    initialized_ = true;
    SPDLOG_info("RAG pipeline initialized successfully");
    return true;
}

std::string RAGPipeline::query(const std::string& question) {
    if (!initialized_) return "[ERROR] Pipeline not initialized";

    // Step 1: Retrieve context
    auto top_chunks = hybrid_retriever_.retrieve(question);

    // Step 2: Build prompt
    std::string prompt = context_builder_.build(top_chunks, question);

    // Step 3: Encode + run inference (record which page blocks the prompt
    //         consumed so a followup can release them safely).
    std::vector<int> prompt_page_ids;
    std::string answer = run_inference(prompt, &prompt_page_ids);

    // Step 4: Record in history
    ChatTurn turn;
    turn.query             = question;
    turn.response          = answer;
    turn.context_page_ids  = std::move(prompt_page_ids);
    history_.push_back(std::move(turn));

    return answer;
}

std::string RAGPipeline::followup(const std::string& question) {
    if (!initialized_) return "[ERROR] Pipeline not initialized";

    // Step 1: Retrieve new context for the follow-up question
    auto top_chunks = hybrid_retriever_.retrieve(question);

    // Step 2: Build prompt with chat history
    std::string prompt = context_builder_.build_followup(top_chunks, question, history_);

    // Step 3: Release non-system context pages held by the previous turn.
    //         Pinned system-prompt blocks stay alive automatically.
    for (auto& turn : history_) {
        for (int pid : turn.context_page_ids) {
            page_mgr_.release_block(pid);
        }
        turn.context_page_ids.clear();
    }

    // Step 4: Run inference
    std::vector<int> prompt_page_ids;
    std::string answer = run_inference(prompt, &prompt_page_ids);

    // Step 5: Record
    ChatTurn turn;
    turn.query             = question;
    turn.response          = answer;
    turn.context_page_ids  = std::move(prompt_page_ids);
    history_.push_back(std::move(turn));

    return answer;
}

std::string RAGPipeline::run_inference(const std::string& prompt,
                                         std::vector<int>* out_page_ids) {
    auto tokens = tokenizer_.encode(prompt);
    if (tokens.empty()) {
        return "[ERROR] Failed to tokenize prompt";
    }

    // Truncate to max_context if needed
    if ((int)tokens.size() > config_.max_context) {
        tokens.resize(config_.max_context);
    }

    std::string text = llm_.generate(tokens, page_mgr_, out_page_ids);
    return text;
}

} // namespace rag
