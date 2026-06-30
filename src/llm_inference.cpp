#include "rag/llm_inference.h"
#include "rag/logger.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>

// ---- rknn-llm SDK type stubs (for compilation without the SDK) ----

#ifndef RAG_HAS_RKNNLLM

// These stubs allow compilation on x86 for development/testing.
// On LubanCat 2 with rknn-llm installed, the real SDK headers are used.

struct rknn_llm_kv_cache_info {
    const int* page_ids;
    uint32_t   num_pages;
    const int* prefill_pages;
    uint32_t   num_prefill;
};

struct rknn_llm_context {
    void* model_mem;
    int   max_context;
    int   max_new_tokens;
    int   decode_call_count;  // used by stub for repeatable behavior
};

extern "C" {

static rknn_llm_context* rknn_llm_create_context(const char* model_path,
                                                   int max_context,
                                                   int max_new_tokens) {
    (void)model_path;
    auto* ctx = new rknn_llm_context();
    ctx->max_context        = max_context;
    ctx->max_new_tokens     = max_new_tokens;
    ctx->model_mem          = nullptr;
    ctx->decode_call_count  = 0;
    return ctx;
}

// Stub reset hook: real SDK resets per-generate counters internally.
static void rknn_llm_reset(rknn_llm_context* ctx) {
    if (ctx) ctx->decode_call_count = 0;
}

static void rknn_llm_destroy_context(rknn_llm_context* ctx) {
    delete ctx;
}

// Batched prefill: encode all new tokens in a single forward pass.
// Returns 0 on success, negative on error.
static int rknn_llm_forward(rknn_llm_context* ctx,
                             const int* input_ids,
                             uint32_t   num_inputs,
                             rknn_llm_kv_cache_info* kv_info,
                             int is_prefill) {
    (void)ctx; (void)input_ids; (void)num_inputs; (void)kv_info; (void)is_prefill;
    return 0;
}

// Single-token decode step.
// Returns the next token ID, or -1 for EOS/error.
static int rknn_llm_decode(rknn_llm_context* ctx,
                            int input_id,
                            rknn_llm_kv_cache_info* kv_info) {
    (void)ctx; (void)kv_info;
    // Stub: emit a deterministic, short sequence (24 tokens) then EOS,
    // so the test path can run end-to-end without an actual model.
    if (ctx) ctx->decode_call_count++;
    int n = ctx ? ctx->decode_call_count : 1;
    constexpr int kMaxStubTokens = 24;
    if (n > kMaxStubTokens) return 151643;  // <|endoftext|>
    return input_id + 1;  // dummy next token
}

} // extern "C"

#endif // RAG_HAS_RKNNLLM

namespace rag {

LLMInference::~LLMInference() {
    if (ctx_) {
        rknn_llm_destroy_context(ctx_);
        ctx_ = nullptr;
    }
}

bool LLMInference::init(const Config& cfg, Tokenizer* tokenizer) {
    config_    = cfg;
    tokenizer_ = tokenizer;

    ctx_ = rknn_llm_create_context(
        cfg.model_path.c_str(),
        cfg.max_context,
        cfg.max_new_tokens);

    if (!ctx_) {
        SPDLOG_error("rknn_llm_create_context returned null");
        return false;
    }
    return true;
}

void LLMInference::batched_prefill(const std::vector<int>& tokens,
                                    const std::vector<int>& block_ids,
                                    PageManager& page_mgr) {
    if (tokens.empty() || block_ids.empty()) return;

    // The real NPU driver handles the mixed (clean + dirty) case via
    // prefill_pages. In the stub we simply mark dirty blocks as clean.
    rknn_llm_kv_cache_info kv_info;
    kv_info.page_ids      = block_ids.data();
    kv_info.num_pages     = (uint32_t)block_ids.size();
    kv_info.prefill_pages = block_ids.data();
    kv_info.num_prefill   = (uint32_t)block_ids.size();

    rknn_llm_forward(ctx_,
                     tokens.data(),
                     (uint32_t)tokens.size(),
                     &kv_info,
                     /*is_prefill=*/1);

    for (int bid : block_ids) {
        if (bid >= 0) page_mgr.mark_clean(bid);
    }
}

std::string LLMInference::generate(const std::vector<int>& tokens,
                                    PageManager& page_mgr,
                                    std::vector<int>* out_page_ids) {
    if (!ctx_ || tokens.empty()) return "";

    // Stub-only: reset per-generate counter so repeated calls don't
    // immediately return EOS. In the real SDK this is implicit.
    rknn_llm_reset(ctx_);

    // ---- Phase 1: Block mapping ----
    auto block_ids = page_mgr.map_tokens_to_blocks(tokens);

    // Filter out failed allocations; keep the rest for prefill/decode.
    std::vector<int> live_block_ids;
    live_block_ids.reserve(block_ids.size());
    for (int bid : block_ids) {
        if (bid >= 0) live_block_ids.push_back(bid);
    }

    // ---- Phase 2: Batched Prefill (single IOCTL) ----
    if (!live_block_ids.empty()) {
        batched_prefill(tokens, live_block_ids, page_mgr);
    }

    // ---- Phase 3: Decode loop ----
    std::string result;
    int last_token = tokens.back();

    rknn_llm_kv_cache_info kv_info;
    kv_info.page_ids      = live_block_ids.empty() ? nullptr : live_block_ids.data();
    kv_info.num_pages     = (uint32_t)live_block_ids.size();
    kv_info.prefill_pages = nullptr;
    kv_info.num_prefill   = 0;

    for (int i = 0; i < config_.max_new_tokens; ++i) {
        int output = rknn_llm_decode(ctx_, last_token, &kv_info);

        if (output < 0) break;
        if (output == config_.eos_token_id) break;
        if (tokenizer_ && tokenizer_->eos_token_id() >= 0 &&
            output == tokenizer_->eos_token_id()) break;

        std::string text = tokenizer_ ? tokenizer_->decode(output) : std::string{};
        if (text.empty()) break;

        result += text;
        last_token = output;

        // Degeneration guard: stop on three identical trailing chars.
        if (i >= 5) {
            size_t rlen = result.size();
            if (rlen >= 3 && result[rlen-1] == result[rlen-2] &&
                result[rlen-2] == result[rlen-3]) {
                break;
            }
        }
    }

    if (out_page_ids) *out_page_ids = block_ids;
    return result;
}

int LLMInference::decode_step(int input_id, const std::vector<int>& block_ids) {
    if (!ctx_) return -1;

    rknn_llm_kv_cache_info kv_info;
    kv_info.page_ids      = block_ids.empty() ? nullptr : block_ids.data();
    kv_info.num_pages     = (uint32_t)block_ids.size();
    kv_info.prefill_pages = nullptr;
    kv_info.num_prefill   = 0;

    return rknn_llm_decode(ctx_, input_id, &kv_info);
}

} // namespace rag
