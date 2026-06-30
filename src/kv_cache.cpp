#include "rag/kv_cache.h"

#include <algorithm>
#include <stdexcept>

namespace rag {

PageManager::PageManager() {
    blocks_.reserve(KV_MAX_PAGES);
}

void PageManager::remove_hash_for_block(int block_id) {
    for (auto it = hash_to_block_.begin(); it != hash_to_block_.end(); ++it) {
        if (it->second.block_id == block_id) {
            hash_to_block_.erase(it);
            return;
        }
    }
}

int PageManager::find_block_by_hash(const std::vector<int>& tokens) const {
    if (tokens.empty()) return -1;
    uint64_t hash = page_hash64(tokens.data(), tokens.size());
    auto it = hash_to_block_.find(hash);
    if (it == hash_to_block_.end()) return -1;
    return it->second.matches(tokens) ? it->second.block_id : -1;
}

int PageManager::alloc_new_block(const std::vector<int>& tokens) {
    int id = (int)blocks_.size();
    KVBlock b;
    b.block_id     = id;
    b.content_hash = page_hash64(tokens.data(), tokens.size());
    b.dirty        = true;
    b.ref_count    = 1;
    blocks_.push_back(b);
    return id;
}

int PageManager::reuse_slot(const std::vector<int>& tokens) {
    // Look for an existing slot we can overwrite: unpinned, ref_count==0.
    for (auto& b : blocks_) {
        if (!b.pinned && b.ref_count == 0) {
            remove_hash_for_block(b.block_id);
            b.content_hash = page_hash64(tokens.data(), tokens.size());
            b.dirty        = true;
            b.ref_count    = 1;
            b.kv_dma_buf   = nullptr;
            b.seq_start    = -1;
            b.seq_end      = -1;
            return b.block_id;
        }
    }
    return -1;
}

std::vector<int> PageManager::map_tokens_to_blocks(const std::vector<int>& tokens) {
    std::vector<int> result;

    int num_pages = ((int)tokens.size() + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
    if (num_pages == 0) return result;

    result.reserve(num_pages);

    for (int p = 0; p < num_pages; ++p) {
        int start = p * KV_PAGE_SIZE;
        int end   = std::min(start + KV_PAGE_SIZE, (int)tokens.size());
        std::vector<int> slice(tokens.begin() + start, tokens.begin() + end);

        int existing = find_block_by_hash(slice);
        if (existing >= 0) {
            // Hash hit + exact match → reuse
            blocks_[existing].ref_count++;
            blocks_[existing].dirty = false;
            result.push_back(existing);
        } else {
            // Miss or collision → allocate (new slot or reuse a freed one)
            int bid = -1;
            if ((int)blocks_.size() < KV_MAX_PAGES) {
                bid = alloc_new_block(slice);
            } else {
                bid = reuse_slot(slice);
            }
            if (bid < 0) {
                // Out of memory: still emit a placeholder so callers can
                // detect the failure via the negative id, but track it
                // as needing prefill.
                result.push_back(-1);
                continue;
            }
            uint64_t hash = page_hash64(slice.data(), slice.size());
            hash_to_block_[hash] = KVCacheTag{bid, slice};
            result.push_back(bid);
        }
    }

    return result;
}

std::vector<int> PageManager::collect_prefill_tokens(
    const std::vector<int>& block_ids,
    const std::vector<int>& all_tokens) const {

    std::vector<int> prefill_tokens;

    for (int i = 0; i < (int)block_ids.size(); ++i) {
        int bid = block_ids[i];
        if (bid < 0 || bid >= (int)blocks_.size()) continue;
        if (!blocks_[bid].dirty) continue;

        int start = i * KV_PAGE_SIZE;
        int end   = std::min(start + KV_PAGE_SIZE, (int)all_tokens.size());
        prefill_tokens.insert(prefill_tokens.end(),
                              all_tokens.begin() + start,
                              all_tokens.begin() + end);
    }

    return prefill_tokens;
}

void PageManager::pin_block(int block_id) {
    if (block_id >= 0 && block_id < (int)blocks_.size()) {
        blocks_[block_id].pinned = true;
    }
}

void PageManager::release_block(int block_id) {
    if (block_id < 0 || block_id >= (int)blocks_.size()) return;
    auto& b = blocks_[block_id];
    if (b.ref_count > 0) b.ref_count--;
}

void PageManager::mark_clean(int block_id) {
    if (block_id < 0 || block_id >= (int)blocks_.size()) return;
    blocks_[block_id].dirty = false;
}

int PageManager::evict_context_blocks(int count) {
    int evicted = 0;
    for (auto& b : blocks_) {
        if (evicted >= count) break;
        if (!b.pinned && b.ref_count == 0) {
            remove_hash_for_block(b.block_id);
            b.kv_dma_buf = nullptr;
            b.dirty      = true;
            // Keep the slot in `blocks_` (its id is stable); reuse_slot
            // will reset its content_hash when it gets reclaimed.
            b.content_hash = 0;
            evicted++;
        }
    }
    return evicted;
}

int PageManager::num_free_slots() const {
    int used = 0;
    for (const auto& b : blocks_) {
        if (b.ref_count > 0 || b.pinned) used++;
    }
    return KV_MAX_PAGES - used;
}

bool PageManager::is_block_pinned(int block_id) const {
    if (block_id < 0 || block_id >= (int)blocks_.size()) return false;
    return blocks_[block_id].pinned;
}

} // namespace rag
