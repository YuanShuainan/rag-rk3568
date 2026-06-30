#include "rag/hybrid_retriever.h"
#include "rag/doc_loader.h"
#include "rag/tokenizer.h"
#include "rag/embedding.h"

#include <iostream>
#include <cassert>
#include <cmath>

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

static std::vector<std::vector<std::string>> tokenize_chunks(
    Tokenizer& tok, const std::vector<Chunk>& chunks) {
    std::vector<std::vector<std::string>> result;
    for (auto& c : chunks) {
        auto tokens = tok.tokenize(c.text);
        std::vector<std::string> terms;
        for (auto& t : tokens) terms.push_back(t.text);
        result.push_back(terms);
    }
    return result;
}

static void test_density_bonus() {
    TEST("Density bonus for keyword-rich chunks");
    DocLoader loader;
    auto chunks = loader.load("data/manual.txt");
    if (chunks.empty()) { FAIL("no chunks loaded"); return; }

    HybridRetriever retriever;
    Tokenizer tok;
    auto tok_chunks = tokenize_chunks(tok, chunks);

    // Build with a non-ready embedding — retriever should fall back to BM25.
    Embedding emb;  // intentionally not initialized
    retriever.build_index(chunks, tok_chunks, &emb);

    auto results = retriever.retrieve("仪表盘有个小扳手亮了怎么消掉");
    if (results.empty()) { FAIL("no results"); return; }

    bool found = false;
    for (auto& r : results) {
        if (r.section_title.find("二") != std::string::npos ||
            r.section_title.find("三") != std::string::npos ||
            r.text.find("小扳手") != std::string::npos ||
            r.text.find("保养") != std::string::npos ||
            r.text.find("复位") != std::string::npos) {
            found = true;
            break;
        }
    }
    if (found) { PASS(); } else { FAIL("no result about maintenance light/reset"); }
}

static void test_section_detection() {
    TEST("Section header detection in doc loader");
    DocLoader loader;
    auto chunks = loader.load("data/manual.txt");
    if (chunks.empty()) { FAIL("no chunks loaded"); return; }

    bool has_section_2 = false;
    bool has_section_3 = false;
    for (auto& c : chunks) {
        if (c.section_title.find("二") != std::string::npos) has_section_2 = true;
        if (c.section_title.find("三") != std::string::npos) has_section_3 = true;
    }
    if (!has_section_2) { FAIL("section 二 not detected"); return; }
    if (!has_section_3) { FAIL("section 三 not detected"); return; }
    PASS();
}

static void test_retrieval_without_onnx() {
    TEST("Retrieval works without ONNX (BM25-only fallback)");
    DocLoader loader;
    auto chunks = loader.load("data/manual.txt");
    if (chunks.empty()) { FAIL("no chunks loaded"); return; }

    HybridRetriever retriever;
    Tokenizer tok;
    auto tok_chunks = tokenize_chunks(tok, chunks);

    Embedding emb;  // not initialized → BM25-only
    retriever.build_index(chunks, tok_chunks, &emb);

    auto results = retriever.retrieve("胎压监测");
    if (results.empty()) { FAIL("no results for 胎压监测"); return; }

    bool found_tire = false;
    for (auto& r : results) {
        if (r.text.find("胎压") != std::string::npos) found_tire = true;
    }
    if (found_tire) { PASS(); } else { FAIL("no tire pressure in results"); }
}

static void test_retrieval_null_embedding() {
    TEST("Retrieval works with null embedding pointer");
    DocLoader loader;
    auto chunks = loader.load("data/manual.txt");
    if (chunks.empty()) { FAIL("no chunks loaded"); return; }

    HybridRetriever retriever;
    Tokenizer tok;
    auto tok_chunks = tokenize_chunks(tok, chunks);

    retriever.build_index(chunks, tok_chunks, /*embedding=*/nullptr);

    auto results = retriever.retrieve("保养");
    if (results.empty()) { FAIL("no results with null embedding"); return; }
    PASS();
}

int main() {
    std::cout << "=== Retriever Tests ===" << std::endl;

    test_section_detection();
    test_density_bonus();
    test_retrieval_without_onnx();
    test_retrieval_null_embedding();

    std::cout << "\nResults: " << tests_passed << "/" << tests_run << " passed." << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
