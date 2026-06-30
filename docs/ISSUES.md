# Issue 草稿

> 本文件汇总 2026-06-30 代码审查发现的问题，方便直接复制到 GitHub Issues。
> 所有问题已在本机修复（见 commit history），此处仅作为缺陷记录。

---

## Issue #1 — Bug 报告：2026-06-30 代码审查

**标签**：`bug` `code-review` `correctness`

**严重度**：高（多处会在无 ONNX / 无 vocab / 中文标题等场景下崩溃或产出错误结果）

---

### 摘要

通读 `src/` 与 `include/rag/` 下的所有源文件后，发现以下若干影响功能正确性或运行时稳定性的问题。

按模块列出。每条都给出文件:行号（基于审查时的 HEAD）+ 问题描述 + 复现条件 + 建议修复方向（已附本仓库采用的修复方案）。

---

### 1. KV Cache：`evict_context_blocks` 与 `alloc_block` 状态不一致（严重）

**文件**：`src/kv_cache.cpp`、`include/rag/kv_cache.h`

**问题**：
- `evict_context_blocks(count)` 只从 `hash_to_block_` 中移除条目、置 `b.dirty=true`、`b.kv_dma_buf=nullptr`，**但不动 `blocks_` 数组**，因此 `b.block_id` 仍是 stale 的。
- `alloc_block` 满了之后调用 `evict_context_blocks(1)`，紧接着 fallback 循环里用 `b.block_id` 做 key，但 `b.kv_dma_buf` 已被清零，下一次 `map_tokens_to_blocks` 通过 hash 命中这条 stale 记录会用到无效指针。

**复现**：
- 调用 32 次 `map_tokens_to_blocks` 填满所有 slot
- 再次 `map_tokens_to_blocks` 触发 `alloc_block` 的 fallback 分支
- 观察 `hash_to_block_` 中残留指向已 evict 槽位的 `KVCacheTag`，但 `blocks_` 实际未腾出

**建议**：
- 引入 `reuse_slot`：真正复用 slot 时清空其 `content_hash`/`ref_count`/`kv_dma_buf` 并把 `hash_to_block_` 中对应条目删除，返回该 slot 的稳定 `block_id`。
- 不要从 `blocks_` 中 erase 元素（保持 id 稳定），slot 仅作为“空闲 / 已用”状态机使用。

---

### 2. ONNX 资源管理与类型不匹配（严重）

**文件**：`include/rag/embedding.h`、`src/embedding.cpp`

**问题**：
- `embedding.h` 用 `struct OrtEnv;`/`struct OrtSession;` 等 C 风格前向声明，但 `onnxruntime_cxx_api.h` 提供的是 `Ort::Env` 等 C++ 类型，二者**不兼容**。
- 析构函数里 `delete static_cast<Ort::Env*>(env_)`：使用 `new Ort::Env` 后再 `delete` 通过 C++ 包装类，行为未定义（`Ort::Env` 析构不可见）。
- `init()` 在 `try/catch` 失败时仅记录日志但**不重置**裸指针，下一次 `~Embedding` 仍会重复 `delete`。

**复现**：在任何 ONNX Runtime 可用的平台编译运行，`Embedding::init` 失败后进程退出时可能崩溃。

**建议**：
- 用 `namespace Ort { class Env; class Session; class MemoryInfo; }` 做前向声明。
- 三个资源改为 `std::unique_ptr<Ort::Env>` / `<Session>` / `<MemoryInfo>`，析构由 unique_ptr 自动处理。
- `init` 失败时直接 reset 三个智能指针，析构函数 `= default`。

---

### 3. `HybridRetriever` 在 embedding 未就绪时崩溃

**文件**：`src/hybrid_retriever.cpp`

**问题**：
- `build_index` 立即调用 `embedding_->embedding_dim()`：当 `embedding_` 未 `init`（`is_ready()==false`）时，`embedding_dim` 返回 0。
- 随后 `emb.resize(dim, 0.0f)` 得到空 vector，循环把所有 chunk 的嵌入 push 进去变成 0 向量，余弦相似度恒为 0，BM25 仍能工作但 dense 重排无意义。
- `retrieve` 中无 `is_ready()` 守卫，`embedding_->encode_single(query)` 直接段错误当 `embedding_` 为 `nullptr`（如单元测试传入 `&uninit_emb`）。

**复现**：
```cpp
Embedding emb;  // 未 init
HybridRetriever r;
r.build_index(chunks, toks, &emb);
r.retrieve("问题");   // crash
```

**建议**：在 `build_index` 与 `retrieve` 入口都检查 `embedding_ && embedding_->is_ready() && embedding_->embedding_dim() > 0`，不可用则降级为 BM25-only。

---

### 4. `ContextBuilder::merge_context` 章节聚合失效

**文件**：`src/context_builder.cpp`

**问题**：
- 原代码用 `(c >= 0x30 && c <= 0x39)` 判断 ASCII 数字字符作为 section_id，但中文标题“一、仪表盘概述”不含 ASCII 数字，导致所有 chunk 都被错误归到 `by_section[0]`。
- 紧接的 `if (by_section.empty() || by_section.find(0) == by_section.end())` 永远不进入，形成死分支。
- 最终输出的 prompt 里章节标题全部变成第一项的 title，section_bonus 在 retriever 端也基本失效。

**复现**：用 `data/manual.txt` 加载文档后调用 `context_builder.build(...)`，观察 `ctx` 串中是否正确包含 `[二、仪表盘警告灯与指示灯]`、`[三、警告灯复位操作]` 等多个标题。

**建议**：
- 实现 `section_id_from_title(title)`，用 UTF-8 字节级前缀匹配 `一、..十九、`，返回数字 id。
- 删除冗余的 `by_section.empty()` 分支。

---

### 5. `DocLoader` 章节切分逻辑错误

**文件**：`src/doc_loader.cpp`

**问题**：
- `detect_section` 中 `十` 在两个 for 循环里都被检查（一次作为 `SECTION_NUMBERS[9]`，一次作为 `十一..十九` 的前缀），逻辑重复。
- `load` 的 `if (sec != current_section_id || current_section_id == -1)` 在第一行非章节内容时把 `current_section_id` 设为该值（实际 sec=-1），且**直接丢弃第一行**。
- `chunk_section` 里 `section_start` 使用 `stream.tellg()`，但 `tellg` 在文本流（尤其 Windows CRLF + UTF-8）下行为不可靠，会导致 `start_pos` 失真。

**建议**：
- 单次匹配 `一、..十九、`，统一返回 0-based id，未识别返回 -1。
- 用 `have_section` 状态标志显式区分 “已识别章节标题” 与 “preamble”，preamble 跳过。
- `start_pos` 改用按行累计的字符 offset，避免依赖 `tellg`。

---

### 6. `Tokenizer::encode` fallback 路径永远走不到

**文件**：`src/tokenizer.cpp`

**问题**：
- `if (text.empty() || vocab_.empty()) return ids;` 在没有 vocab 时直接返回空 vector，**永远走不到下面** `if (token_to_id_.empty())` 的 UTF-8 字符级 fallback。
- 结果：未提供 `qwen_vocab.json` 时 `encode("什么是RK3568？")` 返回空 → `LLMInference::generate` 因 `tokens.empty()` 直接 return 空串。

**复现**：不传 `vocab_path` 启动 `rag_cli`，任何问题都得到空答案。

**建议**：把判定改为 `if (text.empty()) return ids;` 然后判断 `token_to_id_.empty() && vocab_.empty()` 进入字符级 fallback。

---

### 7. `RAGPipeline` 没有真正实现 KV Cache 跨轮复用

**文件**：`src/rag_pipeline.cpp`、`include/rag/rag_pipeline.h`

**问题**：
- `query()` 不把本次 prompt 占用的 page id 写入 `ChatTurn`，所以 `followup()` 中的 `release_block(turn.context_page_ids)` 操作的是空 vector。
- `initialize` 中 `pin_block(bid)` 没有引用计数保护：sys prompt 的 page 在 `followup` 中既不会被 evict（因为 pinned），也不会被复用（ref_count 维持 1），但每次新 prompt 都会重新 map → 重新 alloc → 重新 prefill，**等价于完全没有 KV 复用**，与 README 描述不符。

**复现**：
- 启动 `rag_cli` 问一个问题
- 追问，系统提示已经“缓存”
- 实测 `PageManager::num_blocks()` 与 `max_new_tokens` 之后会一直增长，直到触发 evict

**建议**：
- `LLMInference::generate(..., out_page_ids*)` 把 block id 列表返回
- `query`/`followup` 把列表存入 `turn.context_page_ids`
- `followup` 在生成新 prompt 之前 release 上一轮非 system block
- `sys_block_ids_` 单独保存初始 pin 的 block，析构时再 release 一次

---

### 8. `LLMInference` stub 状态跨调用泄漏

**文件**：`src/llm_inference.cpp`

**问题**：
- `static int call_count` 是文件级 `static`，跨多次 `generate` 调用累加。
- 第一次 `generate` 跑到 `call_count > 30` 返回 EOS 后，**第二次**调用一开始就 `call_count > 30`，立即返回 EOS。
- stub 函数无 `extern "C"`，C++ 名称改编可能与真实 SDK 不一致。

**建议**：
- 把计数搬到 `rknn_llm_context::decode_call_count` 成员
- 增加 `rknn_llm_reset(ctx)`，每次 `generate` 入口重置（模拟真实 SDK 的 per-call 行为）
- stub 全部用 `extern "C"` 包裹

---

### 9. `CMakeLists.txt` 缺少 `BUILD_TESTS` 选项

**文件**：`CMakeLists.txt`、`README.md`

**问题**：
- README 写 `cmake .. -DBUILD_TESTS=ON`，但 CMakeLists 没有 `option(BUILD_TESTS ...)`，该变量被忽略。
- 实际 `enable_testing()` + `add_subdirectory(tests)` 无条件执行。
- 找不到 spdlog 时不会定义 `RAG_HAS_SPDLOG`，logger 永远走 fallback。

**建议**：
```cmake
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SPDLOG_INCLUDE_DIR)
    target_compile_definitions(rag PUBLIC RAG_HAS_SPDLOG=1)
endif()
```

---

### 10. `tests/test_retriever.cpp` 传未初始化 `Embedding` 触发崩溃

**文件**：`tests/test_retriever.cpp`

**问题**：与 Issue #3 同一根因，测试侧也需要更新以反映新行为。

**建议**：
- `test_density_bonus` / `test_retrieval_without_onnx` 显式说明在 BM25-only 路径下也能工作
- 新增 `test_retrieval_null_embedding` 覆盖 `embedding=nullptr` 入参

---

### 11. `tests/test_kv_cache.cpp` 用例依赖旧 alloc 行为

**文件**：`tests/test_kv_cache.cpp`

**问题**：
- `test_eviction` 在 32 个 slot 全满、ref_count=2 的旧实现下恰好能 evict；新实现下 ref_count 在 alloc 时为 1，map 后 +1 变成 2，release 一次到 1，**永远 evict 不到 0**。
- 用例会因此失败。

**建议**：调整 `test_eviction` 中 release 次数与新实现的引用计数一致；新增 `test_release_block` / `test_mark_clean` 覆盖新 API。

---

### 12. 杂项 / 风格

- `include/rag/logger.h` 的 fallback `_fmt` 用 `find` 推进位置，遇到 `{}` 个数少于 args 时直接退出并丢失尾部 → 改为循环填充
- `src/bm25_index.cpp` 引入未使用的 `<set>` 头
- `include/rag/hybrid_retriever.h` 直接 include `embedding.h`，把 ONNX 头拖入所有使用 hybrid retriever 的 TU → 改为前向声明
- `src/main.cpp` `/doc` 是占位打印，未调用 `pipeline.chunks()` 暴露统计；输入行未 trim，命令大小写敏感
- `src/context_builder.cpp::trim_context` 可能切到 UTF-8 多字节中间 → 增加 `trim_to_utf8_boundary`

---

### 验收标准

- [ ] 所有崩溃类问题（#1 #2 #3 #6 #7 #8）已修复
- [ ] 章节聚合（#4 #5）能正确按 `一、..十九、` 区分 section
- [ ] `BUILD_TESTS=OFF` 时不构建 tests
- [ ] 单轮 / 追问两种 query 路径下 `PageManager::num_blocks()` 增长可控
- [ ] 没有 ONNX / 没有 vocab 时仍能跑通 BM25-only fallback（输出无意义但路径完整）

---

### 复现脚本（可贴在评论里）

```bash
# 1. KV Cache 死循环 / 悬空指针
./test_kv_cache   # 应看到 test_eviction FAILED

# 2. ONNX 资源 / 类型不匹配
grep -n "static_cast<Ort::" src/embedding.cpp

# 3. HybridRetriever nullptr 崩溃
./test_retriever  # 应在 test_density_bonus / test_retrieval_without_onnx 段错误
```

---

**Assignee**: 待分配
**Milestone**: v1.0.1
