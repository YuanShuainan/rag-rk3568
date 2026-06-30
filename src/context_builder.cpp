#include "rag/context_builder.h"

#include <algorithm>
#include <cstdint>
#include <map>

namespace rag {

const char* ContextBuilder::system_prompt() {
    return "<|im_start|>system\n"
           "你是汽车仪表盘维修助手。根据以下用户手册内容回答问题。\n"
           "规则:\n"
           "1. 只基于提供的文档内容作答，不要编造信息\n"
           "2. 如果文档不足以回答，直接说\"手册中未找到相关信息\"\n"
           "3. 回答分步骤，每步不超过一句话\n"
           "<|im_end|>\n";
}

const char* ContextBuilder::system_prompt_end() {
    return "<|im_end|>\n";
}

namespace {

// Convert the Chinese-numeric section prefix (一/二/.../十/十一/...) found
// at the start of a title to a stable numeric id. Returns -1 if no
// recognized prefix is found.
int section_id_from_title(const std::string& title) {
    static const char* NUMS[] = {
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"
    };
    constexpr int N = sizeof(NUMS) / sizeof(NUMS[0]);

    if (title.empty()) return -1;

    // First try two-char combinations (十一..十九)
    if (title.size() >= 6) {
        // UTF-8 CJK is 3 bytes; "十" + "一".."九" = 6 bytes total
        for (int i = 1; i <= 9; ++i) {
            std::string prefix = std::string("十") + NUMS[i - 1];
            // Compare as UTF-8 byte sequences; Chinese numerals are 3 bytes each.
            if (title.compare(0, prefix.size(), prefix) == 0) {
                return 10 + i - 1;  // 11..19 → 10..18
            }
        }
    }
    // Single-char prefix
    for (int i = 0; i < N; ++i) {
        if (title.compare(0, std::strlen(NUMS[i]), NUMS[i]) == 0) {
            return i;
        }
    }
    return -1;
}

// Trim `s` so that it does not end in the middle of a UTF-8 code point.
void trim_to_utf8_boundary(std::string& s) {
    while (!s.empty()) {
        unsigned char c = (unsigned char)s.back();
        // ASCII or leading byte of a multi-byte sequence: stop.
        if ((c & 0x80) == 0 || (c & 0xC0) == 0xC0) return;
        s.pop_back();
    }
}

} // namespace

std::string ContextBuilder::merge_context(const std::vector<RetrievedChunk>& retrieved) {
    // Group by section id. We use the numeric id derived from the section
    // title so that the same section from different retrievals collapses
    // together, even if the title was truncated.
    std::map<int, std::vector<const RetrievedChunk*>> by_section;

    for (auto& rc : retrieved) {
        int sid = section_id_from_title(rc.section_title);
        if (sid < 0) sid = 0;  // bucket unknown sections together
        by_section[sid].push_back(&rc);
    }

    std::string ctx;
    for (auto& [sid, rcs] : by_section) {
        // Use the first non-empty section title as the header.
        std::string header;
        for (auto* rc : rcs) {
            if (!rc->section_title.empty()) { header = rc->section_title; break; }
        }
        if (!header.empty()) {
            ctx += "[" + header + "]\n";
        }
        for (auto* rc : rcs) {
            ctx += rc->text + "\n";
        }
    }

    return trim_context(ctx);
}

std::string ContextBuilder::trim_context(const std::string& ctx) {
    if ((int)ctx.size() <= config_.max_context_chars) return ctx;

    std::string trimmed = ctx.substr(0, config_.max_context_chars);
    auto last_nl = trimmed.rfind('\n');
    if (last_nl != std::string::npos && last_nl > (size_t)(config_.max_context_chars / 2)) {
        trimmed.resize(last_nl);
    }
    // Make sure we did not chop a multi-byte CJK character in half.
    trim_to_utf8_boundary(trimmed);
    return trimmed;
}

std::string ContextBuilder::build(const std::vector<RetrievedChunk>& retrieved,
                                   const std::string& user_query) {
    std::string prompt;
    prompt += system_prompt();  // includes <|im_end|>
    prompt += "<|im_start|>user\n";
    prompt += "参考文档:\n";
    prompt += merge_context(retrieved);
    prompt += "\n问题: " + user_query + "\n";
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

std::string ContextBuilder::build_followup(const std::vector<RetrievedChunk>& retrieved,
                                            const std::string& user_query,
                                            const std::vector<ChatTurn>& history) {
    std::string prompt;

    // System prompt
    prompt += system_prompt();

    // Chat history
    for (auto& turn : history) {
        prompt += "<|im_start|>user\n";
        prompt += turn.query;
        prompt += "\n<|im_end|>\n";
        prompt += "<|im_start|>assistant\n";
        prompt += turn.response;
        prompt += "\n<|im_end|>\n";
    }

    // Current query with new context
    prompt += "<|im_start|>user\n";
    prompt += "参考文档:\n";
    prompt += merge_context(retrieved);
    prompt += "\n问题: " + user_query + "\n";
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";

    return prompt;
}

} // namespace rag
