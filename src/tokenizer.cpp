#include "rag/tokenizer.h"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace rag {

// ---- stop-word table ----
// 30+ hardcoded characters: particles, punctuation, common fillers
// that should NOT participate in bigram formation.
static const char* STOP_CHARS =
    "的了是在和就也都要会能可这那有不个人一"
    "吗呢吧啊哈哦嗯么呀啦"
    "，。！？、：；""''（）…—\n\r\t ";

Tokenizer::Tokenizer() {
    for (const char* p = STOP_CHARS; *p; ++p) {
        stop_table_[(unsigned char)*p] = true;
    }
}

bool Tokenizer::is_stop_char(char c) const {
    return stop_table_[(unsigned char)c];
}

std::vector<Tokenizer::Token> Tokenizer::tokenize(const std::string& text) const {
    std::vector<Token> result;
    if (text.empty()) return result;

    // Phase 1: collect all non-space unigrams
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        // Skip pure whitespace/punctuation for unigram collection,
        // but keep them for bigram boundary detection.
        if (!is_stop_char(c) || c == ' ') {
            chars.push_back(std::string(1, c));
        }
    }

    // Unigrams: all collected chars
    for (auto& ch : chars) {
        result.push_back({ch, false});
    }

    // Phase 2: bigrams — only where BOTH chars are non-stop
    // Iterate through raw characters, filtering stop-word boundaries
    std::vector<std::pair<char, int>> clean;  // (char, original_index)
    for (size_t i = 0; i < text.size(); ++i) {
        if (!is_stop_char(text[i])) {
            clean.push_back({text[i], (int)i});
        }
    }

    for (size_t i = 0; i + 1 < clean.size(); ++i) {
        // Ensure consecutive characters in original text (no skipped stop-words between)
        if (clean[i + 1].second == clean[i].second + 1) {
            std::string bigram;
            bigram += clean[i].first;
            bigram += clean[i + 1].first;
            result.push_back({bigram, true});
        }
    }

    return result;
}

// ---- Qwen BPE tokenizer ----

bool Tokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    vocab_.clear();
    token_to_id_.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Format: "token_text token_id" or JSON-style
        // Try JSON format first: "token_text": id
        size_t colon = line.rfind(':');
        std::string token;
        int id = -1;

        if (colon != std::string::npos) {
            // JSON-like: extract quoted key and numeric value
            size_t q1 = line.find('"');
            size_t q2 = line.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos) {
                token = line.substr(q1 + 1, q2 - q1 - 1);
            }
            std::string val = line.substr(colon + 1);
            // strip non-digit
            val.erase(std::remove_if(val.begin(), val.end(),
                     [](char c) { return !std::isdigit(c) && c != '-'; }), val.end());
            if (!val.empty()) id = std::stoi(val);
        } else {
            // Plain format: "token id"
            std::istringstream iss(line);
            iss >> token >> id;
        }

        if (id >= 0 && !token.empty()) {
            if ((int)vocab_.size() <= id) vocab_.resize(id + 1);
            vocab_[id] = token;
            token_to_id_[token] = id;
        }
    }

    // Find EOS token
    auto it = token_to_id_.find("<|endoftext|>");
    if (it != token_to_id_.end()) eos_id_ = it->second;

    return !vocab_.empty();
}

int Tokenizer::find_token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : -1;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (text.empty()) return ids;

    // Character-level fallback when no BPE vocab was loaded:
    // emit one token per UTF-8 character (using its first byte value).
    // This is the path used during development / testing without a
    // real Qwen vocab file. The model output will be nonsensical, but
    // the pipeline still runs end-to-end.
    if (token_to_id_.empty() && vocab_.empty()) {
        for (size_t i = 0; i < text.size(); ) {
            unsigned char c = (unsigned char)text[i];
            int len = 1;
            if      ((c & 0x80) == 0)    len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            else { i++; continue; }

            if (i + len > text.size()) break;

            ids.push_back((int)(c));
            i += len;
        }
        return ids;
    }

    if (token_to_id_.empty()) {
        // Vocab exists but the lookup table is empty (corrupt vocab) — bail.
        return ids;
    }

    // BPE encoding (greedy longest-match)
    size_t pos = 0;
    while (pos < text.size()) {
        int best_len = 0;
        int best_id  = -1;

        // Try progressively shorter substrings
        for (size_t len = std::min(size_t(16), text.size() - pos); len > 0; --len) {
            std::string sub = text.substr(pos, len);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                best_len = (int)len;
                best_id  = it->second;
                break;
            }
        }

        if (best_len > 0) {
            ids.push_back(best_id);
            pos += best_len;
        } else {
            // Fallback: emit byte value
            ids.push_back((int)((unsigned char)text[pos]));
            pos++;
        }
    }

    return ids;
}

std::string Tokenizer::decode(int token_id) const {
    if (!vocab_.empty() && token_id >= 0 && token_id < (int)vocab_.size()) {
        return vocab_[token_id];
    }
    // Fallback
    std::string s;
    s += (char)(token_id & 0xFF);
    return s;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        result += decode(id);
    }
    return result;
}

int Tokenizer::vocab_size() const { return (int)vocab_.size(); }
int Tokenizer::eos_token_id() const { return eos_id_; }

} // namespace rag
