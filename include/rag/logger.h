#pragma once

// Minimal logging wrapper. Uses spdlog if available, otherwise prints to stderr.

// Define RAG_HAS_SPDLOG to use the real spdlog library; otherwise the
// lightweight fallback below is used. CMakeLists.txt should set this macro
// (via target_compile_definitions) when the spdlog headers are found.
#ifdef RAG_HAS_SPDLOG

#include <spdlog/spdlog.h>

#define SPDLOG_info(...)  spdlog::info(__VA_ARGS__)
#define SPDLOG_warn(...)  spdlog::warn(__VA_ARGS__)
#define SPDLOG_error(...) spdlog::error(__VA_ARGS__)

#else

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace rag {
namespace log {

enum level { info, warn, error };

inline std::string _fmt(const std::string& tmpl,
                         const std::vector<std::string>& args) {
    std::string result;
    size_t pos = 0;
    size_t arg_idx = 0;
    while (pos < tmpl.size()) {
        size_t p = tmpl.find("{}", pos);
        if (p == std::string::npos) {
            result.append(tmpl, pos, std::string::npos);
            break;
        }
        result.append(tmpl, pos, p - pos);
        if (arg_idx < args.size()) {
            result += args[arg_idx++];
        } else {
            result += "{}";
        }
        pos = p + 2;
    }
    return result;
}

template<typename... Args>
inline void _log(level lv, const std::string& fmt, Args&&... args) {
    const char* tag = "";
    switch (lv) {
        case info:  tag = "[INFO] ";  break;
        case warn:  tag = "[WARN] ";  break;
        case error: tag = "[ERROR] "; break;
    }

    std::vector<std::string> arg_strs;
    (arg_strs.push_back((std::ostringstream{} << args).str()), ...);

    std::cerr << tag << _fmt(fmt, arg_strs) << std::endl;
}

} // namespace log
} // namespace rag

#define SPDLOG_info(...)  ::rag::log::_log(::rag::log::info, __VA_ARGS__)
#define SPDLOG_warn(...)  ::rag::log::_log(::rag::log::warn, __VA_ARGS__)
#define SPDLOG_error(...) ::rag::log::_log(::rag::log::error, __VA_ARGS__)

#endif // RAG_HAS_SPDLOG
