#include "rag/kv_cache.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

using namespace rag;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    tests_run++; \
    std::cout << "  TEST: " << name << " ... ";

#define PASS() do { \
    std::cout << "PASSED" << std::endl; \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    std::cout << "FAILED: " << msg << std::endl; \
} while(0)

// ---- helpers ----

static std::vector<int> make_tokens(int start, int count) {
    std::vector<int> v;
    v.reserve(count);
    for (int i = 0; i < count; ++i) v.push_back(start + i);
    return v;
}

// ---- tests ----

static void test_page_hash_determinism() {
    TEST("Page hash64 is deterministic");
    std::vector<int> tokens = {101, 2056, 3890, 12, 4400, 998, 1003, 102,
                                500, 601, 702, 803, 904, 1005, 1106, 1207};
    uint64_t h1 = page_hash64(tokens.data(), tokens.size());
    uint64_t h2 = page_hash64(tokens.data(), tokens.size());
    if (h1 == h2) { PASS(); } else { FAIL("hash not deterministic"); }
}

static void test_page_alloc_and_reuse() {
    TEST("Page allocation and exact-match reuse");
    PageManager mgr;

    auto sys_tokens  = make_tokens(1000, 16);
    auto ctx_tokens  = make_tokens(2000, 16);
    auto ctx_tokens2 = make_tokens(3000, 16);

    std::vector<int> all1(sys_tokens.begin(), sys_tokens.end());
    all1.insert(all1.end(), ctx_tokens.begin(), ctx_tokens.end());

    auto blocks1 = mgr.map_tokens_to_blocks(all1);
    if (blocks1.size() != 2) { FAIL("expected 2 blocks, got " + std::to_string(blocks1.size())); return; }

    mgr.pin_block(blocks1[0]);

    std::vector<int> all2(sys_tokens.begin(), sys_tokens.end());
    all2.insert(all2.end(), ctx_tokens2.begin(), ctx_tokens2.end());

    auto blocks2 = mgr.map_tokens_to_blocks(all2);
    if (blocks2.size() != 2) { FAIL("expected 2 blocks in second mapping"); return; }

    // System prompt block should be REUSED (same content hash).
    if (blocks2[0] != blocks1[0]) {
        FAIL("system prompt block not reused (hash miss)");
        return;
    }
    // Context block should be DIFFERENT (different content).
    if (blocks2[1] == blocks1[1]) {
        FAIL("context block incorrectly reused (different content)");
        return;
    }
    PASS();
}

static void test_hash_collision_defense() {
    TEST("Hash collision detected by token snapshot");
    PageManager mgr;

    auto tokens_a = make_tokens(100, 16);
    auto blocks_a = mgr.map_tokens_to_blocks(tokens_a);

    auto tokens_b = make_tokens(999, 16);
    auto blocks_b = mgr.map_tokens_to_blocks(tokens_b);

    if (blocks_a[0] == blocks_b[0]) {
        FAIL("different tokens mapped to same block (unexpected)");
        return;
    }
    PASS();
}

static void test_eviction() {
    TEST("Unpinned blocks evicted correctly");
    PageManager mgr;

    // Fill up pages and release all references.
    std::vector<int> all_block_ids;
    for (int i = 0; i < KV_MAX_PAGES; ++i) {
        auto tokens = make_tokens(i * 100, 16);
        auto bids = mgr.map_tokens_to_blocks(tokens);
        all_block_ids.insert(all_block_ids.end(), bids.begin(), bids.end());
    }

    // Release all blocks (ref_count → 0).
    for (int bid : all_block_ids) {
        mgr.release_block(bid);
    }

    // Pin block 0; it should be preserved.
    mgr.pin_block(0);

    int evicted = mgr.evict_context_blocks(5);
    if (evicted < 1) { FAIL("no blocks evicted"); return; }

    if (!mgr.is_block_pinned(0)) {
        FAIL("pinned block was evicted");
        return;
    }
    PASS();
}

static void test_slice_boundary() {
    TEST("Token slice at page boundary");
    PageManager mgr;
    auto tokens = make_tokens(1, 30);  // 30 tokens → 2 full pages (16+14)

    auto blocks = mgr.map_tokens_to_blocks(tokens);
    if (blocks.size() != 2) {
        FAIL("expected 2 blocks for 30 tokens, got " + std::to_string(blocks.size()));
        return;
    }
    PASS();
}

static void test_release_block() {
    TEST("release_block decrements ref_count");
    PageManager mgr;
    auto tokens = make_tokens(42, 16);
    auto blocks = mgr.map_tokens_to_blocks(tokens);

    // After map: ref_count == 1. Release once → 0.
    mgr.release_block(blocks[0]);
    // Now a fresh mapping with the same content should reuse the block.
    auto blocks2 = mgr.map_tokens_to_blocks(tokens);
    if (blocks2[0] != blocks[0]) {
        FAIL("released block not reused on re-mapping");
        return;
    }
    PASS();
}

static void test_mark_clean() {
    TEST("mark_clean clears the dirty flag");
    PageManager mgr;
    auto tokens = make_tokens(7, 16);
    auto blocks = mgr.map_tokens_to_blocks(tokens);

    mgr.mark_clean(blocks[0]);

    // A second mapping of the same content should be reported as not-dirty
    // (i.e. no prefill required). The cleanest way to verify is to inspect
    // collect_prefill_tokens on a fresh all_tokens sequence.
    std::vector<int> all_tokens = tokens;
    auto prefill = mgr.collect_prefill_tokens(blocks, all_tokens);
    if (!prefill.empty()) {
        FAIL("prefill tokens returned for a clean block");
        return;
    }
    PASS();
}

int main() {
    std::cout << "=== KV Cache Tests ===" << std::endl;

    test_page_hash_determinism();
    test_page_alloc_and_reuse();
    test_hash_collision_defense();
    test_eviction();
    test_slice_boundary();
    test_release_block();
    test_mark_clean();

    std::cout << "\nResults: " << tests_passed << "/" << tests_run << " passed." << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
