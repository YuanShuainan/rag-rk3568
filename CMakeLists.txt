cmake_minimum_required(VERSION 3.16)
project(rag-rk3568 VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_TESTS "Build unit tests" ON)

# ---- Cross-compilation (uncomment for LubanCat 2) ----
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_PROCESSOR aarch64)
# set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# ---- Dependencies ----

# ONNX Runtime
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib
    HINTS ${ONNXRUNTIME_DIR}/lib)

# rknn-llm
find_library(RKNNLLM_LIB rknnllm
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib
    HINTS ${RKNNLLM_DIR}/lib)

# spdlog (header-only)
find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h
    PATHS /usr/include /usr/local/include)

# ---- rag static library ----

set(RAG_HEADERS
    include/rag/types.h
    include/rag/tokenizer.h
    include/rag/doc_loader.h
    include/rag/bm25_index.h
    include/rag/embedding.h
    include/rag/hybrid_retriever.h
    include/rag/context_builder.h
    include/rag/kv_cache.h
    include/rag/llm_inference.h
    include/rag/rag_pipeline.h
    include/rag/logger.h)

set(RAG_SOURCES
    src/tokenizer.cpp
    src/doc_loader.cpp
    src/bm25_index.cpp
    src/embedding.cpp
    src/hybrid_retriever.cpp
    src/context_builder.cpp
    src/kv_cache.cpp
    src/llm_inference.cpp
    src/rag_pipeline.cpp)

add_library(rag STATIC ${RAG_SOURCES} ${RAG_HEADERS})
target_include_directories(rag PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${SPDLOG_INCLUDE_DIR})
target_compile_options(rag PRIVATE -Wall -Wextra -O2 -march=armv8-a)

if(SPDLOG_INCLUDE_DIR)
    target_compile_definitions(rag PUBLIC RAG_HAS_SPDLOG=1)
endif()

if(ONNXRUNTIME_LIB)
    target_link_libraries(rag PUBLIC ${ONNXRUNTIME_LIB})
    target_compile_definitions(rag PUBLIC RAG_HAS_ONNX=1)
endif()

if(RKNNLLM_LIB)
    target_link_libraries(rag PUBLIC ${RKNNLLM_LIB})
    target_compile_definitions(rag PUBLIC RAG_HAS_RKNNLLM=1)
endif()

# ---- CLI executable ----

add_executable(rag_cli src/main.cpp)
target_link_libraries(rag_cli rag)
target_compile_options(rag_cli PRIVATE -Wall -Wextra -O2)

# ---- Tests ----

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Install ----

install(TARGETS rag rag_cli
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib)
install(FILES ${RAG_HEADERS} DESTINATION include/rag)
