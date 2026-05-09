#include "rdma_session.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#ifdef RDMAPY_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * kKiB;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

#ifdef RDMAPY_HAS_CUDA
constexpr const char* kDefaultMemory = "cuda";
#else
constexpr const char* kDefaultMemory = "host";
#endif

struct Options {
    std::string host = "127.0.0.1";
    std::uint16_t port = 2333;
    int ib_port = 1;
    int ib_device = 1;
    int cuda_device = 0;
    std::string memory = kDefaultMemory;
    std::uint64_t offset = 0;
    std::uint64_t prompt_stride = 4 * kKiB;
    int prompt_tokens = 32;
    int token_bytes = 4;
    bool prompt_bytes_set = false;
    std::uint64_t prompt_bytes = 0;
    int min_threads = 1;
    int max_threads = 16;
    int repeats = 5000;
    int warmup = 500;
    int trials = 5;
    int cpu_scan_repeats = 2000;
    int cpu_scan_warmup = 200;
    std::int64_t pace_us = 0;
    std::int64_t stagger_us = 0;
    int io_workers = 1;
    std::size_t cq_batch_size = 16;
    std::size_t mr_cache_limit = 16;
    std::vector<int> cpu_list;
    std::vector<int> cpu_candidates;
    bool auto_cpu_list = false;
    bool round_sync = true;
    bool endpoint_per_thread = true;
    bool verbose = false;
    bool debug = false;
    std::string rdma_log_level = "warn";
    std::string completion_mode = "raw-poll";
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::runtime_error("argument error: " + message);
}

std::string lower_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return value;
}

bool is_blank_arg(const std::string& value) {
    return !value.empty() &&
           std::all_of(
               value.begin(),
               value.end(),
               [](unsigned char ch) {
                   return std::isspace(ch) != 0;
               });
}

std::uint64_t parse_size(const std::string& text) {
    std::string value = lower_copy(text);
    if (value.empty()) {
        usage_error("empty size");
    }

    std::uint64_t scale = 1;
    auto strip_suffix = [&](const std::string& suffix, std::uint64_t suffix_scale) {
        if (value.size() >= suffix.size() &&
            value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) {
            value.resize(value.size() - suffix.size());
            scale = suffix_scale;
            return true;
        }
        return false;
    };

    if (!strip_suffix("gb", kGiB) &&
        !strip_suffix("g", kGiB) &&
        !strip_suffix("mb", kMiB) &&
        !strip_suffix("m", kMiB) &&
        !strip_suffix("kb", kKiB)) {
        strip_suffix("k", kKiB);
    }

    try {
        std::size_t consumed = 0;
        long double parsed = std::stold(value, &consumed);
        if (consumed != value.size()) {
            usage_error("invalid size: " + text);
        }
        if (parsed < 0.0L) {
            usage_error("size must be non-negative: " + text);
        }
        long double bytes = parsed * static_cast<long double>(scale);
        if (bytes > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
            usage_error("size overflows uint64: " + text);
        }
        return static_cast<std::uint64_t>(bytes);
    } catch (const std::invalid_argument&) {
        usage_error("invalid size: " + text);
    } catch (const std::out_of_range&) {
        usage_error("size out of range: " + text);
    }
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

std::string take_value(int& index, int argc, char* argv[], const std::string& flag) {
    if (index + 1 >= argc) {
        usage_error(flag + " requires a value");
    }
    return argv[++index];
}

int parse_cpu_id(const std::string& text) {
    return static_cast<int>(
        parse_ll("CPU id", text, 0, std::numeric_limits<int>::max()));
}

std::vector<int> parse_cpu_list(const std::string& text) {
    if (text.empty()) {
        usage_error("--cpu-list cannot be empty");
    }

    std::vector<int> cpus;
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
            cpus.push_back(parse_cpu_id(item));
        } else {
            if (item.find('-', dash + 1) != std::string::npos) {
                usage_error("invalid CPU range in --cpu-list: " + item);
            }
            int first = parse_cpu_id(item.substr(0, dash));
            int last = parse_cpu_id(item.substr(dash + 1));
            if (first > last) {
                usage_error("descending CPU range in --cpu-list: " + item);
            }
            for (int cpu = first; cpu <= last; ++cpu) {
                cpus.push_back(cpu);
            }
        }

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    return cpus;
}

bool has_duplicates(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    return std::adjacent_find(values.begin(), values.end()) != values.end();
}

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [options]\n\n"
        << "Core options:\n"
        << "  --host HOST                 default: 127.0.0.1\n"
        << "  --port PORT                 default: 2333\n"
        << "  --ib-port PORT              default: 1\n"
        << "  --ib-device DEVICE          default: 1\n"
        << "  --memory cuda|host          default: " << kDefaultMemory << "\n"
        << "  --cuda-device DEVICE        default: 0\n"
        << "  --offset BYTES              default: 0\n"
        << "  --prompt-stride BYTES       default: 4k\n"
        << "  --prompt-tokens N           default: 32\n"
        << "  --token-bytes N             default: 4\n"
        << "  --prompt-bytes BYTES        overrides tokens * token-bytes\n\n"
        << "Benchmark options:\n"
        << "  --min-threads N             default: 1\n"
        << "  --max-threads N             default: 16\n"
        << "  --repeats N                 default: 5000\n"
        << "  --warmup N                  default: 500\n"
        << "  --trials N                  default: 5; each trial recreates QP/CQ\n"
        << "  --pace-us N                 default: 0\n"
        << "  --stagger-us N              default: 0\n"
        << "  --io-workers N              default: 1\n"
        << "  --cq-batch-size N           default: 16\n"
        << "  --mr-cache-limit N          default: 16, 0 disables cache\n"
        << "  --cpu-list LIST             pin workers, e.g. 0-7,16-23\n"
        << "  --auto-cpu-list             benchmark candidate CPUs and use the fastest\n"
        << "  --cpu-candidates LIST       candidates for --auto-cpu-list\n"
        << "  --cpu-scan-repeats N        default: 2000\n"
        << "  --cpu-scan-warmup N         default: 200\n"
        << "  --round-sync                default; sync workers before every READ\n"
        << "  --no-round-sync             let each worker loop independently\n"
        << "  --endpoint-per-thread       default; one endpoint/CQ/QP per worker\n"
        << "  --verbose                   accepted for compatibility; output remains avg only\n"
        << "  --debug\n"
        << "  --completion-mode MODE      raw-poll|callback, default: raw-poll\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n";
}

Options parse_args(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (is_blank_arg(arg)) {
            usage_error(
                "blank argument detected. In shell commands, a line-continuation "
                "backslash must be the final character on the line; remove any "
                "spaces after '\\'.");
        }
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--host") {
            opts.host = take_value(i, argc, argv, arg);
        } else if (arg == "--port") {
            opts.port = static_cast<std::uint16_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1, 65535));
        } else if (arg == "--ib-port") {
            opts.ib_port = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--ib-device") {
            opts.ib_device = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--memory") {
            opts.memory = lower_copy(take_value(i, argc, argv, arg));
        } else if (arg == "--cuda-device") {
            opts.cuda_device = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--offset") {
            opts.offset = parse_size(take_value(i, argc, argv, arg));
        } else if (arg == "--prompt-stride") {
            opts.prompt_stride = parse_size(take_value(i, argc, argv, arg));
        } else if (arg == "--prompt-tokens") {
            opts.prompt_tokens = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--token-bytes") {
            opts.token_bytes = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--prompt-bytes") {
            opts.prompt_bytes = parse_size(take_value(i, argc, argv, arg));
            opts.prompt_bytes_set = true;
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
        } else if (arg == "--cpu-scan-repeats") {
            opts.cpu_scan_repeats = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--cpu-scan-warmup") {
            opts.cpu_scan_warmup = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--pace-us") {
            opts.pace_us = parse_ll(arg, take_value(i, argc, argv, arg), 0,
                                    std::numeric_limits<std::int64_t>::max());
        } else if (arg == "--stagger-us") {
            opts.stagger_us = parse_ll(arg, take_value(i, argc, argv, arg), 0,
                                       std::numeric_limits<std::int64_t>::max());
        } else if (arg == "--io-workers") {
            opts.io_workers = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--cq-batch-size") {
            opts.cq_batch_size = static_cast<std::size_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--mr-cache-limit") {
            opts.mr_cache_limit = static_cast<std::size_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--cpu-list") {
            opts.cpu_list = parse_cpu_list(take_value(i, argc, argv, arg));
        } else if (arg == "--auto-cpu-list") {
            opts.auto_cpu_list = true;
        } else if (arg == "--cpu-candidates") {
            opts.cpu_candidates = parse_cpu_list(take_value(i, argc, argv, arg));
        } else if (arg == "--round-sync") {
            opts.round_sync = true;
        } else if (arg == "--no-round-sync") {
            opts.round_sync = false;
        } else if (arg == "--endpoint-per-thread") {
            opts.endpoint_per_thread = true;
        } else if (arg == "--shared-endpoint") {
            usage_error(
                "--shared-endpoint is no longer supported; this benchmark "
                "requires one endpoint/CQ/QP per worker");
        } else if (arg == "--recreate-per-thread-count" ||
                   arg == "--reuse-max-threads") {
            usage_error(
                arg + " is no longer supported; this benchmark now recreates "
                "QP/CQ for every thread-count trial");
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--debug") {
            opts.debug = true;
        } else if (arg == "--completion-mode") {
            opts.completion_mode = lower_copy(take_value(i, argc, argv, arg));
        } else if (arg == "--rdma-log-level") {
            opts.rdma_log_level = lower_copy(take_value(i, argc, argv, arg));
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (!opts.prompt_bytes_set) {
        if (static_cast<std::uint64_t>(opts.prompt_tokens) >
            std::numeric_limits<std::uint64_t>::max() /
                static_cast<std::uint64_t>(opts.token_bytes)) {
            usage_error("prompt_tokens * token_bytes overflows uint64");
        }
        opts.prompt_bytes =
            static_cast<std::uint64_t>(opts.prompt_tokens) *
            static_cast<std::uint64_t>(opts.token_bytes);
    }

    if (opts.memory != "cuda" && opts.memory != "host") {
        usage_error("--memory must be cuda or host");
    }
    if (opts.prompt_bytes == 0) {
        usage_error("prompt size must be positive");
    }
    if (opts.prompt_bytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        usage_error("prompt size exceeds size_t");
    }
    if (opts.min_threads > opts.max_threads) {
        usage_error("--min-threads must be <= --max-threads");
    }
    if (opts.prompt_stride == 0) {
        usage_error("--prompt-stride must be positive");
    }
    if (opts.completion_mode != "raw-poll" &&
        opts.completion_mode != "callback") {
        usage_error("--completion-mode must be raw-poll or callback");
    }
    if (opts.auto_cpu_list && !opts.cpu_list.empty()) {
        usage_error("--auto-cpu-list cannot be used with --cpu-list");
    }
    if (has_duplicates(opts.cpu_list)) {
        usage_error("--cpu-list contains duplicate CPUs");
    }
    if (has_duplicates(opts.cpu_candidates)) {
        usage_error("--cpu-candidates contains duplicate CPUs");
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

class LocalBuffer {
public:
    virtual ~LocalBuffer() = default;
    virtual std::uintptr_t addr() const = 0;
    virtual std::uint64_t length() const = 0;
    virtual void activate() const {}
};

class HostBuffer final : public LocalBuffer {
public:
    explicit HostBuffer(std::uint64_t nbytes)
        : bytes_(static_cast<std::size_t>(nbytes), 0) {}

    std::uintptr_t addr() const override {
        return reinterpret_cast<std::uintptr_t>(bytes_.data());
    }

    std::uint64_t length() const override {
        return static_cast<std::uint64_t>(bytes_.size());
    }

private:
    std::vector<std::uint8_t> bytes_;
};

#ifdef RDMAPY_HAS_CUDA
void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

class CudaBuffer final : public LocalBuffer {
public:
    CudaBuffer(std::uint64_t nbytes, int device)
        : ptr_(nullptr),
          nbytes_(nbytes),
          device_(device) {
        activate();
        check_cuda(cudaMalloc(&ptr_, static_cast<std::size_t>(nbytes_)),
                   "cudaMalloc");
        check_cuda(cudaMemset(ptr_, 0, static_cast<std::size_t>(nbytes_)),
                   "cudaMemset");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    ~CudaBuffer() override {
        if (ptr_ != nullptr) {
            cudaSetDevice(device_);
            cudaFree(ptr_);
        }
    }

    std::uintptr_t addr() const override {
        return reinterpret_cast<std::uintptr_t>(ptr_);
    }

    std::uint64_t length() const override {
        return nbytes_;
    }

    void activate() const override {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
    }

private:
    void* ptr_;
    std::uint64_t nbytes_;
    int device_;
};
#endif

std::unique_ptr<LocalBuffer> make_buffer(const Options& opts) {
    if (opts.memory == "host") {
        return std::make_unique<HostBuffer>(opts.prompt_bytes);
    }

#ifdef RDMAPY_HAS_CUDA
    return std::make_unique<CudaBuffer>(opts.prompt_bytes, opts.cuda_device);
#else
    throw std::runtime_error(
        "this binary was built without CUDA support; rerun with --memory host "
        "or rebuild after CMake can find CUDAToolkit");
#endif
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

struct WorkerState {
    std::shared_ptr<RdmaSession> session;
    std::unique_ptr<LocalBuffer> buffer;
    std::uint64_t base_offset = 0;
};

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

double cv_pct(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    double avg = mean(values);
    if (avg <= 0.0) {
        return 0.0;
    }
    double accum = 0.0;
    for (double value : values) {
        double diff = value - avg;
        accum += diff * diff;
    }
    double variance = accum / static_cast<double>(values.size() - 1);
    return std::sqrt(variance) / avg * 100.0;
}

class PromptReadBench {
public:
    PromptReadBench(const Options& opts, int workers)
        : opts_(opts) {
        if (!opts_.cpu_list.empty()) {
            pin_current_thread_to_cpu(opts_.cpu_list.front());
        }
        workers_.reserve(static_cast<std::size_t>(workers));
        if (opts_.endpoint_per_thread) {
            endpoints_.reserve(static_cast<std::size_t>(workers));
            for (int worker_id = 0; worker_id < workers; ++worker_id) {
                endpoints_.push_back(make_endpoint());
                add_worker(worker_id, *endpoints_.back());
            }
        } else {
            shared_endpoint_ = make_endpoint();
            for (int worker_id = 0; worker_id < workers; ++worker_id) {
                add_worker(worker_id, *shared_endpoint_);
            }
        }

        for (auto& worker : workers_) {
            prime_worker(worker);
        }
        if (opts_.completion_mode == "raw-poll") {
            stop_cq_pollers();
        }
    }

    std::vector<double> measure_threads(int threads) {
        if (threads < 1 ||
            threads > static_cast<int>(workers_.size())) {
            throw std::runtime_error("requested thread count exceeds bench workers");
        }

        std::vector<std::vector<double>> per_worker(
            static_cast<std::size_t>(threads));
        StartGate gate(threads);
        ReusableGate round_gate(threads);
        std::vector<std::thread> thread_pool;
        std::mutex exception_mutex;
        std::exception_ptr first_exception;
        thread_pool.reserve(static_cast<std::size_t>(threads));

        for (int worker_id = 0; worker_id < threads; ++worker_id) {
            thread_pool.emplace_back([&, worker_id] {
                try {
                    if (!opts_.cpu_list.empty()) {
                        pin_current_thread_to_cpu(
                            opts_.cpu_list[static_cast<std::size_t>(worker_id)]);
                    }
                    gate.arrive_and_wait();

                    auto& worker = workers_[static_cast<std::size_t>(worker_id)];
                    worker.buffer->activate();
                    if (opts_.stagger_us > 0) {
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(
                                opts_.stagger_us *
                                static_cast<std::int64_t>(worker_id)));
                    }

                    const int total_iters = opts_.warmup + opts_.repeats;
                    auto next_issue = std::chrono::steady_clock::now();
                    auto& latencies =
                        per_worker[static_cast<std::size_t>(worker_id)];
                    latencies.reserve(static_cast<std::size_t>(opts_.repeats));

                    for (int iteration = 0; iteration < total_iters; ++iteration) {
                        if (opts_.round_sync) {
                            round_gate.arrive_and_wait();
                        }
                        if (opts_.pace_us > 0 && iteration > 0) {
                            next_issue += std::chrono::microseconds(opts_.pace_us);
                            auto now = std::chrono::steady_clock::now();
                            if (next_issue > now) {
                                std::this_thread::sleep_until(next_issue);
                            } else {
                                next_issue = now;
                            }
                        }

                        double latency_us = 0.0;
                        if (opts_.completion_mode == "raw-poll") {
                            latency_us =
                                worker.session->read_into_transfer_latency_us_raw_poll(
                                    worker.base_offset,
                                    worker.buffer->addr(),
                                    opts_.prompt_bytes,
                                    opts_.prompt_bytes,
                                    nullptr,
                                    worker.buffer->length(),
                                    1,
                                    opts_.debug);
                        } else {
                            latency_us =
                                worker.session->read_into_transfer_latency_us(
                                    worker.base_offset,
                                    worker.buffer->addr(),
                                    opts_.prompt_bytes,
                                    opts_.prompt_bytes,
                                    nullptr,
                                    worker.buffer->length(),
                                    1,
                                    opts_.debug);
                        }
                        if (iteration >= opts_.warmup) {
                            latencies.push_back(latency_us);
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

        for (auto& thread : thread_pool) {
            thread.join();
        }
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        std::vector<double> flat;
        flat.reserve(static_cast<std::size_t>(threads * opts_.repeats));
        for (auto& worker_latencies : per_worker) {
            flat.insert(flat.end(), worker_latencies.begin(), worker_latencies.end());
        }
        return flat;
    }

    std::string mr_cache_summary() const {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::size_t size = 0;
        for (const auto& worker : workers_) {
            hits += worker.session->mr_cache_hits();
            misses += worker.session->mr_cache_misses();
            size += worker.session->mr_cache_size();
        }
        return "mr_cache_total_size=" + std::to_string(size) +
               " hits=" + std::to_string(hits) +
               " misses=" + std::to_string(misses);
    }

private:
    std::unique_ptr<RdmaEndpoint> make_endpoint() const {
        return std::make_unique<RdmaEndpoint>(
            opts_.ib_port,
            opts_.ib_device,
            static_cast<std::uint32_t>(opts_.io_workers),
            opts_.cq_batch_size);
    }

    void add_worker(int worker_id, RdmaEndpoint& endpoint) {
        auto session = endpoint.connect(opts_.host, opts_.port);
        session->set_mr_cache_limit(opts_.mr_cache_limit);
        WorkerState worker;
        worker.session = std::move(session);
        worker.buffer = make_buffer(opts_);
        worker.base_offset =
            opts_.offset +
            static_cast<std::uint64_t>(worker_id) * opts_.prompt_stride;
        workers_.push_back(std::move(worker));
    }

    void prime_worker(WorkerState& worker) const {
        worker.buffer->activate();
        worker.session->read_into(
            worker.base_offset,
            worker.buffer->addr(),
            opts_.prompt_bytes,
            opts_.prompt_bytes,
            nullptr,
            worker.buffer->length(),
            1,
            opts_.debug);
    }

    void stop_cq_pollers() {
        if (shared_endpoint_) {
            shared_endpoint_->stop_cq_poller();
        }
        for (auto& endpoint : endpoints_) {
            endpoint->stop_cq_poller();
        }
    }

    const Options& opts_;
    std::vector<std::unique_ptr<RdmaEndpoint>> endpoints_;
    std::unique_ptr<RdmaEndpoint> shared_endpoint_;
    std::vector<WorkerState> workers_;
};

std::vector<int> allowed_cpus() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    int rc = sched_getaffinity(0, sizeof(set), &set);
    if (rc != 0) {
        throw std::runtime_error(
            std::string("sched_getaffinity failed: ") + std::strerror(errno));
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        throw std::runtime_error("sched_getaffinity returned no allowed CPUs");
    }
    return cpus;
#else
    throw std::runtime_error("--auto-cpu-list is only supported on Linux");
#endif
}

std::vector<int> select_fastest_cpus(const Options& opts) {
    std::vector<int> candidates =
        opts.cpu_candidates.empty() ? allowed_cpus() : opts.cpu_candidates;
    if (static_cast<int>(candidates.size()) < opts.max_threads) {
        usage_error("--cpu-candidates must contain at least --max-threads CPUs");
    }

    std::vector<std::pair<double, int>> scores;
    scores.reserve(candidates.size());
    for (int cpu : candidates) {
        Options scan_opts = opts;
        scan_opts.cpu_list = {cpu};
        scan_opts.cpu_candidates.clear();
        scan_opts.auto_cpu_list = false;
        scan_opts.repeats = opts.cpu_scan_repeats;
        scan_opts.warmup = opts.cpu_scan_warmup;
        scan_opts.trials = 1;

        PromptReadBench bench(scan_opts, 1);
        std::vector<double> latencies = bench.measure_threads(1);
        scores.emplace_back(mean(latencies), cpu);
    }

    std::sort(scores.begin(), scores.end(),
              [](const auto& lhs, const auto& rhs) {
                  if (lhs.first == rhs.first) {
                      return lhs.second < rhs.second;
                  }
                  return lhs.first < rhs.first;
              });

    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(opts.max_threads));
    for (int i = 0; i < opts.max_threads; ++i) {
        selected.push_back(scores[static_cast<std::size_t>(i)].second);
    }
    return selected;
}

void print_header(const Options& opts) {
    (void)opts;
    std::cout << "threads,avg_us\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        Options opts = parse_args(argc, argv);
        set_spdlog_level_from_name(opts.rdma_log_level);
        if (opts.auto_cpu_list) {
            opts.cpu_list = select_fastest_cpus(opts);
        }

        std::cout << std::fixed << std::setprecision(4);
        print_header(opts);

        for (int threads = opts.min_threads;
             threads <= opts.max_threads;
             ++threads) {
            std::vector<double> latencies;
            for (int trial = 0; trial < opts.trials; ++trial) {
                PromptReadBench bench(opts, threads);
                auto trial_latencies = bench.measure_threads(threads);
                latencies.insert(
                    latencies.end(),
                    trial_latencies.begin(),
                    trial_latencies.end());
            }

            std::cout << threads << "," << mean(latencies) << "\n";
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_prompt_read_latency failed: " << e.what() << "\n";
        return 1;
    }
}
