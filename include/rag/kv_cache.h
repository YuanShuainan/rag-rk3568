#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace rag {

// 64-bit hash for page content addressing.
// Based on MurmurHash3 finalizer — compact, no external dependency.
inline uint64_t page_hash64(const int* data, size_t count) {
    uint64_t h = 0x9ae16a3b2f90405fULL;
    for (size_t i = 0; i < count; ++i) {
        uint64_t k = static_cast<uint64_t>(data[i]);
        k *= 0x87c37b91114253d5ULL;
        k  = (k << 31) | (k >> 33);
        k *= 0x4cf5ad432745937fULL;
        h ^= k;
        h  = (h << 27) | (h >> 37);
        h  = h * 5 + 0x52dce729;
    }
    return h;
}

// Page constants
static constexpr int KV_PAGE_SIZE   = 16;   // tokens per page
static constexpr int KV_MAX_PAGES   = 32;   // 512 tokens / 16
static constexpr int KV_TOTAL_SLOTS = KV_PAGE_SIZE * KV_MAX_PAGES;

struct KVCacheTag {
    int              block_id;
    std::vector<int> token_snapshot;  // exact token IDs for collision defense

    bool matches(const std::vector<int>& tokens) const {
        return token_snapshot.size() == tokens.size() &&
               (tokens.empty() ||
                std::memcmp(token_snapshot.data(), tokens.data(),
                            tokens.size() * sizeof(int)) == 0);
    }
};

struct KVBlock {
    int         block_id     = 0;
    uint64_t    content_hash = 0;
    bool        pinned       = false;  // system prompt blocks: never evict
    int         ref_count    = 0;
    bool        dirty        = false;  // needs prefill on next use
    void*       kv_dma_buf   = nullptr;  // ION/DMA-BUF physical memory pointer

    // Logical-to-physical token range this block covers in the current sequence
    int         seq_start    = -1;
    int         seq_end      = -1;
};

class PageManager {
public:
    PageManager();

    // Map a token sequence to block IDs.
    // Returns list of block_ids covering the sequence.
    // New/unmatched blocks are allocated and marked dirty.
    std::vector<int> map_tokens_to_blocks(const std::vector<int>& tokens);

    // Collect tokens from blocks that need prefill.
    std::vector<int> collect_prefill_tokens(const std::vector<int>& block_ids,
                                             const std::vector<int>& all_tokens) const;

    // Pin a block (system prompt).
    void pin_block(int block_id);

    // Release a block reference. Unpinned blocks with ref_count=0 are evictable.
    void release_block(int block_id);

    // Mark a block as clean (its KV cache is now valid in NPU memory).
    void mark_clean(int block_id);

    // Evict context blocks to make room. Returns number of blocks evicted.
    int  evict_context_blocks(int count);

    int  num_blocks() const { return (int)blocks_.size(); }
    int  num_free_slots() const;
    bool is_block_pinned(int block_id) const;
    int  find_block_by_hash(const std::vector<int>& tokens) const;

private:
    // Reuse an existing slot: clear its state and return its block_id.
    // Returns -1 if no slot is available.
    int  reuse_slot(const std::vector<int>& tokens);
    int  alloc_new_block(const std::vector<int>& tokens);
    void remove_hash_for_block(int block_id);

    std::vector<KVBlock>                          blocks_;
    std::unordered_map<uint64_t, KVCacheTag>      hash_to_block_;
};

} // namespace rag
