#include "rag/doc_loader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace rag {

std::string DocLoader::read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

int DocLoader::detect_section(const std::string& line, int /*current_id*/) {
    // Match Chinese-numbered section headers like:
    //   "一、xxx", "二、xxx", ..., "十、xxx",
    //   "十一、xxx", "十二、xxx", ..., "十九、xxx"
    // Returns a 0-based section id, or -1 if not a header.
    static const char* SINGLE_NUMS[] = {
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"
    };
    constexpr int SINGLE_COUNT = sizeof(SINGLE_NUMS) / sizeof(SINGLE_NUMS[0]);

    if (line.empty()) return -1;

    // Two-character forms: 十一..十九 (each CJK char is 3 bytes in UTF-8)
    for (int i = 1; i <= 9; ++i) {
        std::string prefix = std::string("十") + SINGLE_NUMS[i - 1] + "、";
        if (line.compare(0, prefix.size(), prefix) == 0) {
            return 10 + i - 1;  // 11..19 → 10..18
        }
    }

    // Single-character forms
    for (int i = 0; i < SINGLE_COUNT; ++i) {
        std::string prefix = std::string(SINGLE_NUMS[i]) + "、";
        if (line.compare(0, prefix.size(), prefix) == 0) {
            return i;
        }
    }

    return -1;
}

void DocLoader::chunk_section(const std::string& text, int section_id,
                               const std::string& title, int offset,
                               std::vector<Chunk>& out) {
    if (text.empty()) return;

    // Trim leading/trailing whitespace
    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\n' || text[start] == '\r' || text[start] == '\t'))
        start++;
    size_t end = text.size();
    while (end > start && (text[end-1] == ' ' || text[end-1] == '\n' || text[end-1] == '\r' || text[end-1] == '\t'))
        end--;

    if (start >= end) return;
    std::string clean = text.substr(start, end - start);
    if ((int)clean.size() < config_.min_chunk) return;

    int pos = 0;
    int base = offset + (int)start;
    int step = std::max(1, config_.chunk_size - config_.overlap);

    while (pos < (int)clean.size()) {
        int remain = (int)clean.size() - pos;
        if (remain < config_.min_chunk) break;

        int chunk_len = std::min(config_.chunk_size, remain);

        // Don't cut in the middle of a UTF-8 character (only when not at end)
        if (pos + chunk_len < (int)clean.size()) {
            while (chunk_len > 0) {
                unsigned char c = (unsigned char)clean[pos + chunk_len - 1];
                if ((c & 0x80) == 0 || (c & 0xC0) == 0xC0) break;  // ASCII or leading byte
                chunk_len--;
            }
        }

        if (chunk_len <= 0) break;

        Chunk c;
        c.id            = (int)out.size();
        c.text          = clean.substr(pos, chunk_len);
        c.section_id    = section_id;
        c.section_title = title;
        c.start_pos     = base + pos;
        out.push_back(c);

        pos += step;

        // Re-align to next UTF-8 leading byte after the step
        while (pos > 0 && pos < (int)clean.size()) {
            unsigned char cb = (unsigned char)clean[pos];
            if ((cb & 0x80) == 0 || (cb & 0xC0) == 0xC0) break;
            pos++;
        }
    }
}

std::vector<Chunk> DocLoader::load(const std::string& file_path) {
    std::string raw = read_file(file_path);
    if (raw.empty()) return {};

    std::vector<Chunk> chunks;
    chunks.reserve(raw.size() / (config_.chunk_size - config_.overlap) + 4);

    std::istringstream stream(raw);
    std::string line;

    int         current_section_id    = -1;
    std::string current_section_title;
    std::string current_section_text;
    int         section_start         = 0;
    bool        have_section          = false;

    while (std::getline(stream, line)) {
        // Remove trailing \r (CRLF)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        int sec = detect_section(line, current_section_id);

        if (sec >= 0) {
            // New section header — flush previous section's text first.
            if (have_section && !current_section_text.empty()) {
                chunk_section(current_section_text, current_section_id,
                              current_section_title, section_start, chunks);
            }
            current_section_id    = sec;
            current_section_title = line;
            current_section_text.clear();
            section_start         = (int)stream.tellg();
            have_section          = true;
        } else {
            // Body line — accumulate into the current section.
            if (!have_section) {
                // Pre-section preamble (e.g. blank lines or a leading
                // title that doesn't match a numbered header). Skip it
                // rather than attributing it to a phantom section.
                continue;
            }
            if (!current_section_text.empty()) current_section_text += "\n";
            current_section_text += line;
        }
    }

    // Flush last section
    if (have_section && !current_section_text.empty()) {
        chunk_section(current_section_text, current_section_id,
                      current_section_title, section_start, chunks);
    }

    return chunks;
}

} // namespace rag
