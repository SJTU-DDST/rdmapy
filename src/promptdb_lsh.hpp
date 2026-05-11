#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace promptdb_rdma {

constexpr std::uint16_t kProtocolVersion = 3;
constexpr std::uint32_t kClientHelloMagic = 0x48444250u; // PDBH
constexpr std::uint32_t kServerHelloMagic = 0x68534450u; // PDSh
constexpr std::uint32_t kRequestMagic = 0x51424450u;     // PDBQ
constexpr std::uint32_t kAckMagic = 0x41424450u;         // PDBA
constexpr std::uint32_t kResultMagic = 0x52424450u;      // PDBR

enum class RequestOp : std::uint16_t {
    query = 1,
    shutdown = 2,
};

enum class Status : std::uint16_t {
    ok = 0,
    not_found = 1,
    bad_request = 2,
    server_error = 3,
};

#pragma pack(push, 1)
struct ClientHelloWire {
    std::uint32_t magic = kClientHelloMagic;
    std::uint16_t version = kProtocolVersion;
    std::uint16_t reserved = 0;
    std::uint32_t max_tokens = 0;
    std::uint32_t result_bytes = 0;
};

struct ServerHelloWire {
    std::uint32_t magic = kServerHelloMagic;
    std::uint16_t version = kProtocolVersion;
    std::uint16_t status = static_cast<std::uint16_t>(Status::ok);
    std::uint32_t max_tokens = 0;
    std::uint32_t result_header_bytes = 0;
};

struct RequestWire {
    std::uint32_t magic = kRequestMagic;
    std::uint16_t version = kProtocolVersion;
    std::uint16_t op = static_cast<std::uint16_t>(RequestOp::query);
    std::uint64_t request_id = 0;
    std::uint32_t token_count = 0;
    std::uint32_t reserved = 0;
};

struct AckWire {
    std::uint32_t magic = kAckMagic;
    std::uint16_t version = kProtocolVersion;
    std::uint16_t status = static_cast<std::uint16_t>(Status::ok);
    std::uint64_t request_id = 0;
    std::uint32_t result_bytes = 0;
    std::int32_t best_idx = -1;
    double prompt_transfer_us = 0.0;
    double server_match_us = 0.0;
    double result_transfer_us = 0.0;
};

struct ResultHeaderWire {
    std::uint32_t magic = kResultMagic;
    std::uint16_t version = kProtocolVersion;
    std::uint16_t status = static_cast<std::uint16_t>(Status::ok);
    std::uint64_t request_id = 0;
    std::int32_t best_idx = -1;
    double best_score = -1.0;
    std::uint32_t token_count = 0;
    std::uint32_t result_bytes = 0;
};
#pragma pack(pop)

inline std::uint32_t checked_u32_size(std::uint64_t value,
                                      const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t prompt_bytes_for_tokens(std::uint32_t token_count) {
    return checked_u32_size(
        static_cast<std::uint64_t>(token_count) * sizeof(std::uint32_t),
        "prompt byte count");
}

inline std::uint32_t result_bytes_for_tokens(std::uint32_t token_count) {
    return checked_u32_size(
        static_cast<std::uint64_t>(sizeof(ResultHeaderWire)) +
            static_cast<std::uint64_t>(token_count) * sizeof(std::uint32_t),
        "result byte count");
}

class NumpyMt19937 {
public:
    explicit NumpyMt19937(std::uint32_t seed = 5489u) {
        this->seed(seed);
    }

    void seed(std::uint32_t seed_value) {
        state_[0] = seed_value;
        for (std::size_t i = 1; i < state_.size(); ++i) {
            state_[i] = static_cast<std::uint32_t>(
                1812433253u * (state_[i - 1] ^ (state_[i - 1] >> 30)) +
                static_cast<std::uint32_t>(i));
        }
        index_ = state_.size();
    }

    std::uint32_t next_uint32() {
        if (index_ >= state_.size()) {
            twist();
        }
        std::uint32_t y = state_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    double random_sample() {
        const std::uint32_t a = next_uint32() >> 5;
        const std::uint32_t b = next_uint32() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) *
               (1.0 / 9007199254740992.0);
    }

    std::uint32_t uniform_below(std::uint32_t bound) {
        if (bound == 0) {
            throw std::runtime_error("uniform_below bound must be positive");
        }
        const std::uint64_t span = 1ull << 32;
        const std::uint64_t limit = span - (span % bound);
        std::uint32_t value = 0;
        do {
            value = next_uint32();
        } while (static_cast<std::uint64_t>(value) >= limit);
        return value % bound;
    }

private:
    void twist() {
        static constexpr std::uint32_t kUpperMask = 0x80000000u;
        static constexpr std::uint32_t kLowerMask = 0x7fffffffu;
        static constexpr std::uint32_t kMatrixA = 0x9908b0dfu;

        for (std::size_t i = 0; i < state_.size(); ++i) {
            const std::uint32_t y =
                (state_[i] & kUpperMask) |
                (state_[(i + 1) % state_.size()] & kLowerMask);
            state_[i] = state_[(i + 397) % state_.size()] ^ (y >> 1);
            if ((y & 1u) != 0) {
                state_[i] ^= kMatrixA;
            }
        }
        index_ = 0;
    }

    std::array<std::uint32_t, 624> state_{};
    std::size_t index_ = 624;
};

inline const std::array<std::uint32_t, 256>& crc32_table() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < t.size(); ++i) {
            std::uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit) {
                c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    return table;
}

inline std::uint32_t zlib_crc32(std::string_view bytes) {
    std::uint32_t crc = 0xffffffffu;
    const auto& table = crc32_table();
    for (unsigned char ch : bytes) {
        crc = table[(crc ^ ch) & 0xffu] ^ (crc >> 8);
    }
    return crc ^ 0xffffffffu;
}

struct PromptDbConfig {
    int prompts_len = 4096;
    int num_hash_planes = 16;
    int vocab_size = 32000;
    int bucket_size = 8;
    int seed = 42;
    int num_buckets = 20000;
    int num_hashes = 16;
    int signature_tokens = 3;
};

struct FindResult {
    int best_idx = -1;
    double best_score = -1.0;
    std::vector<std::uint32_t> best_tokens;
};

inline bool contains_token(std::span<const std::uint32_t> values,
                           std::uint32_t needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

inline bool seen_before(std::span<const std::uint32_t> values,
                        std::size_t pos) {
    return std::find(values.begin(), values.begin() +
                                      static_cast<std::ptrdiff_t>(pos),
                     values[pos]) != values.begin() +
                                     static_cast<std::ptrdiff_t>(pos);
}

inline std::size_t unique_count(std::span<const std::uint32_t> values) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!seen_before(values, i)) {
            ++count;
        }
    }
    return count;
}

inline double overlap_score(std::span<const std::uint32_t> a,
                            std::span<const std::uint32_t> b) {
    std::size_t intersection = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (seen_before(a, i)) {
            continue;
        }
        if (contains_token(b, a[i])) {
            ++intersection;
        }
    }
    const std::size_t uni = unique_count(a) + unique_count(b) - intersection;
    return uni == 0 ? 0.0
                    : static_cast<double>(intersection) /
                          static_cast<double>(uni);
}

class PromptDbState {
public:
    explicit PromptDbState(PromptDbConfig config)
        : config_(config),
          hash_matrix_(static_cast<std::size_t>(config.num_hashes) *
                       static_cast<std::size_t>(config.vocab_size)),
          lsh_buckets_(static_cast<std::size_t>(config.num_buckets)),
          prompts_(static_cast<std::size_t>(config.prompts_len)),
          alive_(static_cast<std::size_t>(config.prompts_len), false),
          prompt_bucket_ids_(static_cast<std::size_t>(config.prompts_len), -1) {
        if (config_.prompts_len <= 0 || config_.vocab_size <= 0 ||
            config_.bucket_size <= 0 || config_.num_buckets <= 0 ||
            config_.num_hashes <= 0 || config_.signature_tokens <= 0) {
            throw std::runtime_error("invalid PromptDbConfig");
        }

        NumpyMt19937 rng(static_cast<std::uint32_t>(config_.seed));
        for (double& value : hash_matrix_) {
            value = rng.random_sample();
        }
    }

    const PromptDbConfig& config() const {
        return config_;
    }

    int num_prompts() const {
        return num_prompts_;
    }

    std::optional<int> add_prompt(int idx,
                                  std::vector<std::uint32_t> tokens) {
        if (idx < 0 || idx >= config_.prompts_len) {
            throw std::out_of_range("prompt idx out of range");
        }

        const int old_bucket_id =
            prompt_bucket_ids_[static_cast<std::size_t>(idx)];
        if (old_bucket_id != -1) {
            auto& old_bucket =
                lsh_buckets_[static_cast<std::size_t>(old_bucket_id)];
            old_bucket.erase(
                std::remove(old_bucket.begin(), old_bucket.end(), idx),
                old_bucket.end());
        }

        const bool was_empty = !alive_[static_cast<std::size_t>(idx)];
        prompts_[static_cast<std::size_t>(idx)] = std::move(tokens);
        alive_[static_cast<std::size_t>(idx)] = true;

        const int bucket_id =
            get_bucket_id(prompts_[static_cast<std::size_t>(idx)]);
        if (bucket_id == -1) {
            prompt_bucket_ids_[static_cast<std::size_t>(idx)] = -1;
            if (was_empty) {
                ++num_prompts_;
            }
            return std::nullopt;
        }

        auto& bucket = lsh_buckets_[static_cast<std::size_t>(bucket_id)];
        std::optional<int> evicted;
        if (static_cast<int>(bucket.size()) == config_.bucket_size) {
            const int old_idx = bucket.front();
            if (old_idx != idx) {
                alive_[static_cast<std::size_t>(old_idx)] = false;
                prompts_[static_cast<std::size_t>(old_idx)].clear();
                prompt_bucket_ids_[static_cast<std::size_t>(old_idx)] = -1;
                --num_prompts_;
                evicted = old_idx;
            }
            bucket.pop_front();
        }

        bucket.push_back(idx);
        prompt_bucket_ids_[static_cast<std::size_t>(idx)] = bucket_id;
        if (was_empty) {
            ++num_prompts_;
        }
        return evicted;
    }

    FindResult find_most_similar_prompt(
        const std::vector<std::uint32_t>& input_tokens) const {
        return find_most_similar_prompt(std::span<const std::uint32_t>(
            input_tokens.data(), input_tokens.size()));
    }

    FindResult find_most_similar_prompt(
        std::span<const std::uint32_t> input_tokens) const {
        const int bucket_id = get_bucket_id(input_tokens);

        const std::deque<int>* bucket_candidates = nullptr;
        if (bucket_id != -1) {
            const auto& bucket =
                lsh_buckets_[static_cast<std::size_t>(bucket_id)];
            if (!bucket.empty()) {
                bucket_candidates = &bucket;
            }
        }

        FindResult result;
        auto visit_candidate = [&](int idx) {
            if (idx < 0 || idx >= config_.prompts_len ||
                !alive_[static_cast<std::size_t>(idx)]) {
                return;
            }
            const auto& tokens = prompts_[static_cast<std::size_t>(idx)];
            if (tokens.size() != input_tokens.size()) {
                return;
            }
            const double score = overlap_score(
                input_tokens,
                std::span<const std::uint32_t>(tokens.data(), tokens.size()));
            if (score > result.best_score) {
                result.best_score = score;
                result.best_idx = idx;
                result.best_tokens = tokens;
            }
        };

        if (bucket_candidates != nullptr) {
            for (int idx : *bucket_candidates) {
                visit_candidate(idx);
            }
        } else {
            for (int idx = 0; idx < config_.prompts_len; ++idx) {
                if (alive_[static_cast<std::size_t>(idx)]) {
                    visit_candidate(idx);
                }
            }
        }

        return result;
    }

    int get_bucket_id(const std::vector<std::uint32_t>& token_ids) const {
        return get_bucket_id(
            std::span<const std::uint32_t>(token_ids.data(), token_ids.size()));
    }

    int get_bucket_id(std::span<const std::uint32_t> token_ids) const {
        const std::string py_string = signature_to_python_string(token_ids);
        if (py_string.empty()) {
            return -1;
        }
        return static_cast<int>(
            zlib_crc32(py_string) %
            static_cast<std::uint32_t>(config_.num_buckets));
    }

    std::vector<std::vector<std::uint32_t>> hash_tokens(
        const std::vector<std::uint32_t>& token_ids) const {
        std::vector<std::uint32_t> tokens;
        tokens.reserve(token_ids.size());
        for (std::uint32_t token : token_ids) {
            if (token < static_cast<std::uint32_t>(config_.vocab_size)) {
                tokens.push_back(token);
            }
        }
        if (tokens.empty()) {
            return {};
        }

        const std::size_t n = tokens.size();
        const std::size_t current_l = std::min<std::size_t>(
            static_cast<std::size_t>(config_.signature_tokens), n);
        std::vector<std::vector<std::uint32_t>> signature;
        signature.reserve(static_cast<std::size_t>(config_.num_hashes));

        for (int h = 0; h < config_.num_hashes; ++h) {
            std::vector<std::pair<double, std::size_t>> ranked;
            ranked.reserve(n);
            const std::size_t row =
                static_cast<std::size_t>(h) *
                static_cast<std::size_t>(config_.vocab_size);
            for (std::size_t pos = 0; pos < n; ++pos) {
                ranked.emplace_back(
                    hash_matrix_[row + static_cast<std::size_t>(tokens[pos])],
                    pos);
            }

            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& lhs, const auto& rhs) {
                          if (lhs.first == rhs.first) {
                              return lhs.second < rhs.second;
                          }
                          return lhs.first < rhs.first;
                      });

            std::vector<std::size_t> chosen_positions;
            chosen_positions.reserve(current_l);
            for (std::size_t i = 0; i < current_l; ++i) {
                chosen_positions.push_back(ranked[i].second);
            }
            std::sort(chosen_positions.begin(), chosen_positions.end());

            std::vector<std::uint32_t> row_tokens;
            row_tokens.reserve(current_l);
            for (std::size_t pos : chosen_positions) {
                row_tokens.push_back(tokens[pos]);
            }
            signature.push_back(std::move(row_tokens));
        }

        return signature;
    }

private:
    static void append_uint32(std::string& out, std::uint32_t value) {
        char buffer[16];
        auto [ptr, ec] = std::to_chars(
            buffer, buffer + sizeof(buffer), value);
        if (ec != std::errc()) {
            throw std::runtime_error("failed to format uint32");
        }
        out.append(buffer, ptr);
    }

    std::string signature_to_python_string(
        std::span<const std::uint32_t> token_ids) const {
        bool all_valid = true;
        for (std::uint32_t token : token_ids) {
            if (token >= static_cast<std::uint32_t>(config_.vocab_size)) {
                all_valid = false;
                break;
            }
        }

        if (all_valid) {
            return signature_to_python_string_for_valid_tokens(token_ids);
        }

        std::vector<std::uint32_t> filtered;
        filtered.reserve(token_ids.size());
        for (std::uint32_t token : token_ids) {
            if (token < static_cast<std::uint32_t>(config_.vocab_size)) {
                filtered.push_back(token);
            }
        }
        return signature_to_python_string_for_valid_tokens(
            std::span<const std::uint32_t>(filtered.data(), filtered.size()));
    }

    std::string signature_to_python_string_for_valid_tokens(
        std::span<const std::uint32_t> tokens) const {
        if (tokens.empty()) {
            return {};
        }

        const std::size_t n = tokens.size();
        const std::size_t current_l = std::min<std::size_t>(
            static_cast<std::size_t>(config_.signature_tokens), n);
        static constexpr std::size_t kStackSignatureTokens = 64;
        std::array<std::size_t, kStackSignatureTokens> stack_positions{};
        std::array<double, kStackSignatureTokens> stack_values{};
        std::vector<std::size_t> heap_positions;
        std::vector<double> heap_values;
        std::span<std::size_t> best_positions;
        std::span<double> best_values;
        if (current_l <= kStackSignatureTokens) {
            best_positions =
                std::span<std::size_t>(stack_positions.data(), current_l);
            best_values = std::span<double>(stack_values.data(), current_l);
        } else {
            heap_positions.resize(current_l);
            heap_values.resize(current_l);
            best_positions =
                std::span<std::size_t>(heap_positions.data(), current_l);
            best_values = std::span<double>(heap_values.data(), current_l);
        }

        std::string out;
        out.reserve(static_cast<std::size_t>(config_.num_hashes) *
                    (current_l * 12 + 4) + 2);
        out.push_back('(');

        for (int h = 0; h < config_.num_hashes; ++h) {
            if (h != 0) {
                out.append(", ");
            }

            const std::size_t row =
                static_cast<std::size_t>(h) *
                static_cast<std::size_t>(config_.vocab_size);
            std::size_t best_count = 0;
            for (std::size_t pos = 0; pos < n; ++pos) {
                const double value =
                    hash_matrix_[row + static_cast<std::size_t>(tokens[pos])];
                if (best_count < current_l) {
                    best_values[best_count] = value;
                    best_positions[best_count] = pos;
                    ++best_count;
                    continue;
                }

                std::size_t worst = 0;
                for (std::size_t i = 1; i < current_l; ++i) {
                    if (best_values[worst] < best_values[i] ||
                        (best_values[worst] == best_values[i] &&
                         best_positions[worst] < best_positions[i])) {
                        worst = i;
                    }
                }
                if (value < best_values[worst] ||
                    (value == best_values[worst] &&
                     pos < best_positions[worst])) {
                    best_values[worst] = value;
                    best_positions[worst] = pos;
                }
            }

            std::sort(best_positions.begin(), best_positions.end());

            out.push_back('(');
            for (std::size_t i = 0; i < current_l; ++i) {
                if (i != 0) {
                    out.append(", ");
                }
                append_uint32(out, tokens[best_positions[i]]);
            }
            if (current_l == 1) {
                out.push_back(',');
            }
            out.push_back(')');
        }

        if (config_.num_hashes == 1) {
            out.push_back(',');
        }
        out.push_back(')');
        return out;
    }

    static std::string signature_to_python_string(
        const std::vector<std::vector<std::uint32_t>>& signature) {
        std::ostringstream out;
        out << "(";
        for (std::size_t i = 0; i < signature.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << "(";
            const auto& row = signature[i];
            for (std::size_t j = 0; j < row.size(); ++j) {
                if (j != 0) {
                    out << ", ";
                }
                out << row[j];
            }
            if (row.size() == 1) {
                out << ",";
            }
            out << ")";
        }
        if (signature.size() == 1) {
            out << ",";
        }
        out << ")";
        return out.str();
    }

    PromptDbConfig config_;
    std::vector<double> hash_matrix_;
    std::vector<std::deque<int>> lsh_buckets_;
    std::vector<std::vector<std::uint32_t>> prompts_;
    std::vector<bool> alive_;
    std::vector<int> prompt_bucket_ids_;
    int num_prompts_ = 0;
};

inline std::vector<std::vector<std::uint32_t>> make_deterministic_prompts(
    int count,
    int token_len,
    int vocab_size,
    int seed) {
    if (count < 0 || token_len < 0 || vocab_size <= 0) {
        throw std::runtime_error("invalid deterministic prompt parameters");
    }
    NumpyMt19937 rng(static_cast<std::uint32_t>(seed));
    std::vector<std::vector<std::uint32_t>> prompts;
    prompts.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        std::vector<std::uint32_t> tokens;
        tokens.reserve(static_cast<std::size_t>(token_len));
        for (int j = 0; j < token_len; ++j) {
            tokens.push_back(
                rng.uniform_below(static_cast<std::uint32_t>(vocab_size)));
        }
        prompts.push_back(std::move(tokens));
    }
    return prompts;
}

inline PromptDbState make_prefilled_prompt_db(const PromptDbConfig& config,
                                              int token_len) {
    PromptDbState db(config);
    auto prompts = make_deterministic_prompts(
        config.prompts_len, token_len, config.vocab_size, config.seed);
    for (int idx = 0; idx < static_cast<int>(prompts.size()); ++idx) {
        db.add_prompt(idx, std::move(prompts[static_cast<std::size_t>(idx)]));
    }
    return db;
}

} // namespace promptdb_rdma
