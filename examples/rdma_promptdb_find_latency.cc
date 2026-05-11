#include "qp_connector.h"
#include "promptdb_lsh.hpp"

#include <asio/co_spawn.hpp>
#include <asio/use_future.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <rdmapp/executor.h>
#include <rdmapp/mr.h>
#include <rdmapp/rdmapp.h>
#include <spdlog/spdlog.h>

#include <infiniband/verbs.h>

#ifdef __linux__
#include <pthread.h>
#endif

namespace {

using promptdb_rdma::AckWire;
using promptdb_rdma::ClientHelloWire;
using promptdb_rdma::RequestOp;
using promptdb_rdma::RequestWire;
using promptdb_rdma::ResultHeaderWire;
using promptdb_rdma::ServerHelloWire;
using promptdb_rdma::Status;

struct Options {
    std::string host = "127.0.0.1";
    std::uint16_t port = 2333;
    int ib_device = 0;
    int ib_port = 1;
    int io_workers = 1;
    std::size_t cq_batch_size = 16;
    int prompts = 4096;
    int token_len = 32;
    int max_tokens = 0;
    int vocab_size = 32000;
    int seed = 42;
    int min_threads = 1;
    int max_threads = 16;
    int repeats = 1000;
    int warmup = 100;
    int trials = 3;
    std::int64_t pace_us = 0;
    std::int64_t stagger_us = 0;
    int thread_gap_ms = 50;
    bool sync_each_repeat = false;
    bool verbose = false;
    std::string rdma_log_level = "warn";
    std::vector<int> cpu_list;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::runtime_error("argument error: " + message);
}

long long parse_ll(const std::string& name,
                   const std::string& value,
                   long long min_value,
                   long long max_value) {
    try {
        std::size_t consumed = 0;
        long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size()) {
            usage_error("invalid " + name + ": " + value);
        }
        if (parsed < min_value || parsed > max_value) {
            usage_error(name + " outside allowed range: " + value);
        }
        return parsed;
    } catch (const std::invalid_argument&) {
        usage_error("invalid " + name + ": " + value);
    } catch (const std::out_of_range&) {
        usage_error(name + " out of range: " + value);
    }
}

std::string take_value(int& index,
                       int argc,
                       char* argv[],
                       const std::string& flag) {
    if (index + 1 >= argc) {
        usage_error(flag + " requires a value");
    }
    return argv[++index];
}

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [options]\n\n"
        << "RDMA options:\n"
        << "  --host HOST                 default: 127.0.0.1\n"
        << "  --port PORT                 default: 2333\n"
        << "  --ib-device DEV             default: 0\n"
        << "  --ib-port PORT              default: 1\n"
        << "  --io-workers N              default: 1 per connection\n"
        << "  --cq-batch-size N           default: 16\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n\n"
        << "Prompt workload:\n"
        << "  --prompts N                 default: 4096\n"
        << "  --token-len N               default: 32\n"
        << "  --max-tokens N              default: token-len\n"
        << "  --vocab-size N              default: 32000\n"
        << "  --seed N                    default: 42; must match server\n\n"
        << "Benchmark:\n"
        << "  --min-threads N             default: 1\n"
        << "  --max-threads N             default: 16\n"
        << "  --repeats N                 default: 1000\n"
        << "  --warmup N                  default: 100\n"
        << "  --trials N                  default: 3; each trial recreates QP/CQ\n"
        << "  --pace-us N                 default: 0\n"
        << "  --stagger-us N              default: 0\n"
        << "  --thread-gap-ms N           default: 50; pause after each thread count\n"
        << "  --sync-each-repeat          sync workers before each query; burst stress mode\n"
        << "  --round-sync                alias for --sync-each-repeat\n"
        << "  --no-round-sync             default; let each worker loop independently\n"
        << "  --cpu-list LIST             pin workers, e.g. 0-7,16-23\n"
        << "  --verbose                   print p50/p95/p99 too\n";
}

std::vector<int> parse_cpu_list(const std::string& text) {
    if (text.empty()) {
        usage_error("--cpu-list cannot be empty");
    }
    std::vector<int> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t comma = text.find(',', start);
        std::string item = text.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            usage_error("empty item in --cpu-list");
        }
        std::size_t dash = item.find('-');
        if (dash == std::string::npos) {
            values.push_back(static_cast<int>(
                parse_ll("--cpu-list item", item, 0,
                         std::numeric_limits<int>::max())));
        } else {
            if (item.find('-', dash + 1) != std::string::npos) {
                usage_error("invalid range in --cpu-list: " + item);
            }
            int first = static_cast<int>(
                parse_ll("--cpu-list range start", item.substr(0, dash), 0,
                         std::numeric_limits<int>::max()));
            int last = static_cast<int>(
                parse_ll("--cpu-list range end", item.substr(dash + 1), 0,
                         std::numeric_limits<int>::max()));
            if (first > last) {
                usage_error("descending range in --cpu-list: " + item);
            }
            for (int cpu = first; cpu <= last; ++cpu) {
                values.push_back(cpu);
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return values;
}

Options parse_args(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--host") {
            opts.host = take_value(i, argc, argv, arg);
        } else if (arg == "--port") {
            opts.port = static_cast<std::uint16_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1, 65535));
        } else if (arg == "--ib-device") {
            opts.ib_device = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--ib-port") {
            opts.ib_port = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--io-workers") {
            opts.io_workers = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--cq-batch-size") {
            opts.cq_batch_size = static_cast<std::size_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--prompts") {
            opts.prompts = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--token-len") {
            opts.token_len = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--max-tokens") {
            opts.max_tokens = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--vocab-size") {
            opts.vocab_size = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--seed") {
            opts.seed = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--min-threads") {
            opts.min_threads = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--max-threads") {
            opts.max_threads = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--repeats") {
            opts.repeats = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--warmup") {
            opts.warmup = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--trials") {
            opts.trials = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--pace-us") {
            opts.pace_us = parse_ll(arg, take_value(i, argc, argv, arg), 0,
                                    std::numeric_limits<std::int64_t>::max());
        } else if (arg == "--stagger-us") {
            opts.stagger_us = parse_ll(arg, take_value(i, argc, argv, arg), 0,
                                       std::numeric_limits<std::int64_t>::max());
        } else if (arg == "--thread-gap-ms") {
            opts.thread_gap_ms = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--sync-each-repeat" || arg == "--round-sync") {
            opts.sync_each_repeat = true;
        } else if (arg == "--no-round-sync") {
            opts.sync_each_repeat = false;
        } else if (arg == "--cpu-list") {
            opts.cpu_list = parse_cpu_list(take_value(i, argc, argv, arg));
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--rdma-log-level") {
            opts.rdma_log_level = take_value(i, argc, argv, arg);
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (opts.max_tokens == 0) {
        opts.max_tokens = opts.token_len;
    }
    if (opts.max_tokens < opts.token_len) {
        usage_error("--max-tokens must be >= --token-len");
    }
    if (opts.min_threads > opts.max_threads) {
        usage_error("--min-threads must be <= --max-threads");
    }
    if (!opts.cpu_list.empty() &&
        static_cast<int>(opts.cpu_list.size()) < opts.max_threads) {
        usage_error("--cpu-list must contain at least --max-threads CPUs");
    }
    return opts;
}

void set_spdlog_level_from_name(const std::string& level) {
    if (level == "trace") {
        spdlog::set_level(spdlog::level::trace);
    } else if (level == "debug") {
        spdlog::set_level(spdlog::level::debug);
    } else if (level == "info") {
        spdlog::set_level(spdlog::level::info);
    } else if (level == "warn" || level == "warning") {
        spdlog::set_level(spdlog::level::warn);
    } else if (level == "error") {
        spdlog::set_level(spdlog::level::err);
    } else if (level == "critical") {
        spdlog::set_level(spdlog::level::critical);
    } else if (level == "off") {
        spdlog::set_level(spdlog::level::off);
    } else {
        usage_error("unsupported RDMA log level: " + level);
    }
}

template <typename T>
asio::awaitable<void> send_pod(std::shared_ptr<rdmapp::qp> qp,
                               const T& value) {
    co_await qp->send(std::as_bytes(std::span(&value, 1)));
}

template <typename T>
asio::awaitable<T> recv_pod(std::shared_ptr<rdmapp::qp> qp,
                            const char* what) {
    T value{};
    auto [bytes, imm] = co_await qp->recv(
        std::as_writable_bytes(std::span(&value, 1)));
    (void)imm;
    if (bytes != sizeof(T)) {
        throw std::runtime_error(
            std::string("short recv for ") + what + ": got " +
            std::to_string(bytes) + " expected " + std::to_string(sizeof(T)));
    }
    co_return value;
}

asio::awaitable<void> send_local_mr(std::shared_ptr<rdmapp::qp> qp,
                                    const rdmapp::local_mr& mr) {
    auto serialized = mr.serialize();
    co_await qp->send(std::as_bytes(
        std::span(serialized.data(), serialized.size())));
}

void raw_send_mr_wait(std::shared_ptr<rdmapp::qp> qp,
                      const rdmapp::local_mr& local_mr) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(local_mr.addr());
    sge.length = static_cast<std::uint32_t>(local_mr.length());
    sge.lkey = local_mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.next = nullptr;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    send_wr.wr_id = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_wc wc = {};
    qp->post_send(send_wr, bad_send_wr);
    while (!qp->poll_send_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw send failed: ") +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
}

void raw_send_mr_nowait(std::shared_ptr<rdmapp::qp> qp,
                        const rdmapp::local_mr& local_mr) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(local_mr.addr());
    sge.length = static_cast<std::uint32_t>(local_mr.length());
    sge.lkey = local_mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.next = nullptr;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    send_wr.wr_id = 1;
    send_wr.send_flags = 0;

    qp->post_send(send_wr, bad_send_wr);
}

struct RawRecvState {
    std::shared_ptr<rdmapp::qp> qp;
    std::uint32_t expected_bytes = 0;
    const char* what = "recv";
    bool check_bytes = true;
    bool expect_imm = false;
};

RawRecvState raw_post_recv_mr(std::shared_ptr<rdmapp::qp> qp,
                              const rdmapp::local_mr& local_mr,
                              std::uint32_t expected_bytes,
                              const char* what) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(local_mr.addr());
    sge.length = expected_bytes;
    sge.lkey = local_mr.lkey();

    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr* bad_recv_wr = nullptr;
    recv_wr.next = nullptr;
    recv_wr.wr_id = 1;
    recv_wr.num_sge = 1;
    recv_wr.sg_list = &sge;

    qp->post_recv(recv_wr, bad_recv_wr);
    return RawRecvState{
        std::move(qp), expected_bytes, what, true, false};
}

RawRecvState raw_post_recv_imm(std::shared_ptr<rdmapp::qp> qp,
                               const char* what) {
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr* bad_recv_wr = nullptr;
    recv_wr.next = nullptr;
    recv_wr.wr_id = 2;
    recv_wr.num_sge = 0;
    recv_wr.sg_list = nullptr;

    qp->post_recv(recv_wr, bad_recv_wr);
    return RawRecvState{std::move(qp), 0, what, false, true};
}

void raw_wait_recv(const RawRecvState& state) {
    struct ibv_wc wc = {};
    while (!state.qp->poll_recv_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw recv failed for ") + state.what + ": " +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
    if (state.expect_imm &&
        (wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM ||
         !(wc.wc_flags & IBV_WC_WITH_IMM))) {
        throw std::runtime_error(
            std::string("expected RDMA write-with-imm completion for ") +
            state.what);
    }
    if (state.check_bytes && wc.byte_len != state.expected_bytes) {
        throw std::runtime_error(
            std::string("short raw recv for ") + state.what + ": got " +
            std::to_string(wc.byte_len) + " expected " +
            std::to_string(state.expected_bytes));
    }
}

template <typename T>
void raw_send_pod_wait(std::shared_ptr<rdmapp::qp> qp, const T& value) {
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(const_cast<T*>(&value), sizeof(T)));
    raw_send_mr_wait(qp, *mr);
}

void raw_send_bytes_wait(std::shared_ptr<rdmapp::qp> qp,
                         std::vector<std::uint8_t>& bytes) {
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(bytes.data(), bytes.size()));
    raw_send_mr_wait(qp, *mr);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    std::size_t idx = static_cast<std::size_t>(
        std::ceil((pct / 100.0) * static_cast<double>(values.size())));
    idx = std::min<std::size_t>(std::max<std::size_t>(idx, 1) - 1,
                                values.size() - 1);
    return values[idx];
}

struct QuerySample {
    double total_us = 0.0;
    double prompt_transfer_us = 0.0;
    double server_match_us = 0.0;
    double result_transfer_us = 0.0;
};

double client_overhead_us(const QuerySample& sample) {
    return sample.total_us -
           sample.prompt_transfer_us -
           sample.server_match_us -
           sample.result_transfer_us;
}

std::vector<double> collect_field(
    const std::vector<QuerySample>& samples,
    double (*field_fn)(const QuerySample&)) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        values.push_back(field_fn(sample));
    }
    return values;
}

double mean_field(const std::vector<QuerySample>& samples,
                  double (*field_fn)(const QuerySample&)) {
    return mean(collect_field(samples, field_fn));
}

double percentile_field(const std::vector<QuerySample>& samples,
                        double (*field_fn)(const QuerySample&),
                        double pct) {
    return percentile(collect_field(samples, field_fn), pct);
}

double sample_total_us(const QuerySample& sample) {
    return sample.total_us;
}

double sample_prompt_transfer_us(const QuerySample& sample) {
    return sample.prompt_transfer_us;
}

double sample_server_match_us(const QuerySample& sample) {
    return sample.server_match_us;
}

double sample_result_transfer_us(const QuerySample& sample) {
    return sample.result_transfer_us;
}

double sample_client_overhead_us(const QuerySample& sample) {
    return client_overhead_us(sample);
}

void pin_current_thread_to_cpu(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        throw std::runtime_error(
            "failed to pin worker to CPU " + std::to_string(cpu) +
            ": " + std::strerror(rc));
    }
#else
    (void)cpu;
    throw std::runtime_error("--cpu-list is only supported on Linux");
#endif
}

class StartGate {
public:
    explicit StartGate(int expected)
        : expected_(expected) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(mutex_);
        ++arrived_;
        if (arrived_ == expected_) {
            released_ = true;
            cv_.notify_all();
            return;
        }
        cv_.wait(lk, [this] { return released_; });
    }

private:
    int expected_;
    int arrived_ = 0;
    bool released_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class ReusableGate {
public:
    explicit ReusableGate(int expected)
        : expected_(expected) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(mutex_);
        const int generation = generation_;
        ++arrived_;
        if (arrived_ == expected_) {
            arrived_ = 0;
            ++generation_;
            cv_.notify_all();
            return;
        }
        cv_.wait(lk, [&] { return generation_ != generation; });
    }

private:
    int expected_;
    int arrived_ = 0;
    int generation_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class PromptDbConnection {
public:
    explicit PromptDbConnection(const Options& opts)
        : opts_(opts),
          device_(std::make_shared<rdmapp::device>(
              opts.ib_device, opts.ib_port)),
          pd_(std::make_shared<rdmapp::pd>(device_)),
          recv_cq_(std::make_shared<rdmapp::cq>(device_)),
          send_cq_(std::make_shared<rdmapp::cq>(device_)),
          io_ctx_(std::make_shared<asio::io_context>(opts.io_workers)),
          prompt_buffer_(static_cast<std::size_t>(opts.max_tokens), 0),
          result_buffer_(promptdb_rdma::result_bytes_for_tokens(
              static_cast<std::uint32_t>(opts.max_tokens)),
              0) {
        work_guard_.emplace(asio::make_work_guard(*io_ctx_));
        io_threads_.reserve(static_cast<std::size_t>(opts.io_workers));
        for (int i = 0; i < opts.io_workers; ++i) {
            io_threads_.emplace_back([ctx = io_ctx_] {
                ctx->run();
            });
        }

        prompt_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(prompt_buffer_.data(),
                        prompt_buffer_.size() * sizeof(std::uint32_t)));
        result_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(result_buffer_.data(), result_buffer_.size()));
        ack_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(&ack_buffer_, sizeof(ack_buffer_)));
        request_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(&request_buffer_, sizeof(request_buffer_)));

        auto connector = std::make_shared<rdmapp::qp_connector>(
            opts.host, opts.port, pd_, recv_cq_, send_cq_);
        auto connect_future = asio::co_spawn(
            *io_ctx_,
            [connector]() -> asio::awaitable<std::shared_ptr<rdmapp::qp>> {
                co_return co_await connector->connect();
            }(),
            asio::use_future);
        qp_ = connect_future.get();
        if (!qp_) {
            throw std::runtime_error("failed to connect RDMA QP");
        }
        handshake();
        work_guard_.reset();
        io_ctx_->stop();
        io_threads_.clear();
    }

    ~PromptDbConnection() {
        close_noexcept();
        work_guard_.reset();
        if (io_ctx_) {
            io_ctx_->stop();
        }
    }

    QuerySample query(const std::vector<std::uint32_t>& tokens,
                      std::uint64_t request_id) {
        if (tokens.empty() ||
            tokens.size() > static_cast<std::size_t>(opts_.max_tokens)) {
            throw std::runtime_error("query token count out of range");
        }
        std::copy(tokens.begin(), tokens.end(), prompt_buffer_.begin());

        request_buffer_ = RequestWire{};
        request_buffer_.request_id = request_id;
        request_buffer_.token_count =
            static_cast<std::uint32_t>(tokens.size());

        auto ack_state = raw_post_recv_imm(qp_, "ack write-with-imm");
        auto t0 = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_release);
        raw_send_mr_nowait(qp_, *request_mr_);
        raw_wait_recv(ack_state);
        std::atomic_thread_fence(std::memory_order_acquire);
        AckWire ack = ack_buffer_;
        auto t1 = std::chrono::steady_clock::now();

        validate_ack_and_result(ack, request_buffer_.request_id);
        QuerySample sample;
        sample.total_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        sample.prompt_transfer_us = ack.prompt_transfer_us;
        sample.server_match_us = ack.server_match_us;
        sample.result_transfer_us = ack.result_transfer_us;
        return sample;
    }

private:
    void handshake() {
        ClientHelloWire hello;
        hello.max_tokens = static_cast<std::uint32_t>(opts_.max_tokens);
        hello.result_bytes = static_cast<std::uint32_t>(result_buffer_.size());

        ServerHelloWire resp{};
        auto resp_mr = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(&resp, sizeof(resp)));
        auto resp_state =
            raw_post_recv_mr(qp_, *resp_mr, sizeof(resp), "server hello");

        raw_send_pod_wait(qp_, hello);
        auto prompt_serialized = prompt_mr_->serialize();
        raw_send_bytes_wait(qp_, prompt_serialized);
        auto result_serialized = result_mr_->serialize();
        raw_send_bytes_wait(qp_, result_serialized);
        auto ack_serialized = ack_mr_->serialize();
        raw_send_bytes_wait(qp_, ack_serialized);
        raw_wait_recv(resp_state);

        if (resp.magic != promptdb_rdma::kServerHelloMagic ||
            resp.version != promptdb_rdma::kProtocolVersion ||
            resp.status != static_cast<std::uint16_t>(Status::ok)) {
            throw std::runtime_error("server rejected PromptDB RDMA handshake");
        }
    }

    void close_noexcept() noexcept {
        if (closed_ || !qp_ || !io_ctx_) {
            return;
        }
        closed_ = true;
        try {
            request_buffer_ = RequestWire{};
            request_buffer_.op =
                static_cast<std::uint16_t>(RequestOp::shutdown);
            request_buffer_.request_id =
                std::numeric_limits<std::uint64_t>::max();
            raw_send_mr_wait(qp_, *request_mr_);
        } catch (...) {
        }
    }

    void validate_ack_and_result(const AckWire& ack,
                                 std::uint64_t request_id) const {
        if (ack.magic != promptdb_rdma::kAckMagic ||
            ack.version != promptdb_rdma::kProtocolVersion ||
            ack.request_id != request_id) {
            throw std::runtime_error("bad ack");
        }
        if (ack.status != static_cast<std::uint16_t>(Status::ok)) {
            throw std::runtime_error(
                "query failed with status=" + std::to_string(ack.status));
        }
        if (ack.result_bytes < sizeof(ResultHeaderWire) ||
            ack.result_bytes > result_buffer_.size()) {
            throw std::runtime_error("bad result byte count in ack");
        }

        ResultHeaderWire header;
        std::memcpy(&header, result_buffer_.data(), sizeof(header));
        if (header.magic != promptdb_rdma::kResultMagic ||
            header.version != promptdb_rdma::kProtocolVersion ||
            header.request_id != request_id ||
            header.status != static_cast<std::uint16_t>(Status::ok) ||
            header.best_idx < 0 ||
            header.result_bytes != ack.result_bytes ||
            header.result_bytes !=
                promptdb_rdma::result_bytes_for_tokens(header.token_count)) {
            throw std::runtime_error("bad result header");
        }
    }

    const Options& opts_;
    std::shared_ptr<rdmapp::device> device_;
    std::shared_ptr<rdmapp::pd> pd_;
    std::shared_ptr<rdmapp::cq> recv_cq_;
    std::shared_ptr<rdmapp::cq> send_cq_;
    std::shared_ptr<asio::io_context> io_ctx_;
    std::shared_ptr<rdmapp::qp> qp_;
    std::vector<std::uint32_t> prompt_buffer_;
    std::vector<std::uint8_t> result_buffer_;
    std::shared_ptr<rdmapp::local_mr> prompt_mr_;
    std::shared_ptr<rdmapp::local_mr> result_mr_;
    AckWire ack_buffer_{};
    std::shared_ptr<rdmapp::local_mr> ack_mr_;
    RequestWire request_buffer_{};
    std::shared_ptr<rdmapp::local_mr> request_mr_;
    std::vector<std::jthread> io_threads_;
    std::optional<asio::executor_work_guard<asio::io_context::executor_type>>
        work_guard_;
    bool closed_ = false;
};

std::vector<QuerySample> run_trial(
    const Options& opts,
    const std::vector<std::vector<std::uint32_t>>& prompts,
    int threads,
    std::uint64_t trial_base_request_id) {
    std::vector<std::unique_ptr<PromptDbConnection>> connections;
    connections.reserve(static_cast<std::size_t>(threads));
    for (int i = 0; i < threads; ++i) {
        connections.push_back(std::make_unique<PromptDbConnection>(opts));
    }

    std::vector<std::vector<QuerySample>> per_worker(
        static_cast<std::size_t>(threads));
    StartGate start_gate(threads);
    ReusableGate round_gate(threads);
    std::vector<std::thread> worker_threads;
    std::mutex exception_mutex;
    std::exception_ptr first_exception;
    worker_threads.reserve(static_cast<std::size_t>(threads));

    for (int worker_id = 0; worker_id < threads; ++worker_id) {
        worker_threads.emplace_back([&, worker_id] {
            try {
                if (!opts.cpu_list.empty()) {
                    pin_current_thread_to_cpu(
                        opts.cpu_list[static_cast<std::size_t>(worker_id)]);
                }
                start_gate.arrive_and_wait();
                if (opts.stagger_us > 0) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(
                            opts.stagger_us *
                            static_cast<std::int64_t>(worker_id)));
                }

                auto& latencies =
                    per_worker[static_cast<std::size_t>(worker_id)];
                latencies.reserve(static_cast<std::size_t>(opts.repeats));
                const int total_iters = opts.warmup + opts.repeats;
                auto next_issue = std::chrono::steady_clock::now();

                for (int iteration = 0; iteration < total_iters; ++iteration) {
                    if (opts.sync_each_repeat) {
                        round_gate.arrive_and_wait();
                    }
                    if (opts.pace_us > 0 && iteration > 0) {
                        next_issue += std::chrono::microseconds(opts.pace_us);
                        auto now = std::chrono::steady_clock::now();
                        if (next_issue > now) {
                            std::this_thread::sleep_until(next_issue);
                        } else {
                            next_issue = now;
                        }
                    }

                    const std::size_t prompt_id =
                        (static_cast<std::size_t>(worker_id) *
                             static_cast<std::size_t>(opts.repeats) +
                         static_cast<std::size_t>(iteration)) %
                        prompts.size();
                    const std::uint64_t request_id =
                        trial_base_request_id +
                        static_cast<std::uint64_t>(worker_id) *
                            1'000'000ull +
                        static_cast<std::uint64_t>(iteration);
                    const QuerySample sample =
                        connections[static_cast<std::size_t>(worker_id)]
                            ->query(prompts[prompt_id], request_id);
                    if (iteration >= opts.warmup) {
                        latencies.push_back(sample);
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lk(exception_mutex);
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        });
    }

    for (auto& worker : worker_threads) {
        worker.join();
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

    std::vector<QuerySample> flat;
    flat.reserve(static_cast<std::size_t>(threads * opts.repeats));
    for (auto& latencies : per_worker) {
        flat.insert(flat.end(), latencies.begin(), latencies.end());
    }
    return flat;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        Options opts = parse_args(argc, argv);
        set_spdlog_level_from_name(opts.rdma_log_level);

        auto prompts = promptdb_rdma::make_deterministic_prompts(
            opts.prompts, opts.token_len, opts.vocab_size, opts.seed);

        std::cout << std::fixed << std::setprecision(4);
        std::cout
            << "# host=" << opts.host << ":" << opts.port
            << " prompts=" << opts.prompts
            << " token_len=" << opts.token_len
            << " repeats=" << opts.repeats
            << " warmup=" << opts.warmup
            << " trials=" << opts.trials
            << " thread_gap_ms=" << opts.thread_gap_ms
            << " sync_each_repeat=" << (opts.sync_each_repeat ? "true" : "false")
            << " data_path=send_request_rdma_read_prompt_rdma_write_result_ack_write_imm\n";
        if (opts.verbose) {
            std::cout
                << "threads,total_avg_us,total_p50_us,total_p95_us,total_p99_us,"
                << "prompt_transfer_avg_us,prompt_transfer_p50_us,"
                << "server_match_avg_us,server_match_p50_us,"
                << "result_transfer_avg_us,result_transfer_p50_us,"
                << "client_overhead_avg_us\n";
        } else {
            std::cout
                << "threads,total_avg_us,prompt_transfer_avg_us,"
                << "server_match_avg_us,result_transfer_avg_us,"
                << "client_overhead_avg_us\n";
        }

        for (int threads = opts.min_threads;
             threads <= opts.max_threads;
             ++threads) {
            std::vector<QuerySample> samples;
            for (int trial = 0; trial < opts.trials; ++trial) {
                auto trial_latencies = run_trial(
                    opts,
                    prompts,
                    threads,
                    static_cast<std::uint64_t>(threads) * 100'000'000ull +
                        static_cast<std::uint64_t>(trial) * 10'000'000ull);
                samples.insert(samples.end(),
                               trial_latencies.begin(),
                               trial_latencies.end());
            }

            if (opts.verbose) {
                std::cout << threads << ","
                          << mean_field(samples, sample_total_us) << ","
                          << percentile_field(samples, sample_total_us, 50) << ","
                          << percentile_field(samples, sample_total_us, 95) << ","
                          << percentile_field(samples, sample_total_us, 99) << ","
                          << mean_field(samples, sample_prompt_transfer_us) << ","
                          << percentile_field(samples, sample_prompt_transfer_us, 50) << ","
                          << mean_field(samples, sample_server_match_us) << ","
                          << percentile_field(samples, sample_server_match_us, 50) << ","
                          << mean_field(samples, sample_result_transfer_us) << ","
                          << percentile_field(samples, sample_result_transfer_us, 50) << ","
                          << mean_field(samples, sample_client_overhead_us)
                          << "\n";
            } else {
                std::cout << threads << ","
                          << mean_field(samples, sample_total_us) << ","
                          << mean_field(samples, sample_prompt_transfer_us) << ","
                          << mean_field(samples, sample_server_match_us) << ","
                          << mean_field(samples, sample_result_transfer_us) << ","
                          << mean_field(samples, sample_client_overhead_us)
                          << "\n";
            }
            std::cout.flush();
            if (opts.thread_gap_ms > 0 && threads != opts.max_threads) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(opts.thread_gap_ms));
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_promptdb_find_latency failed: " << e.what() << "\n";
        return 1;
    }
}
