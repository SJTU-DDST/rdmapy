#pragma once

#include "rdma_session.hpp"

#include <algorithm>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
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
#endif

#ifdef RDMAPY_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace rdma_read_bench {

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
    int ib_device = 0;
    std::vector<std::string> server_hosts;
    std::vector<std::uint16_t> server_ports;
    std::vector<int> ib_ports;
    std::vector<int> ib_devices;
    int client_id = 0;
    int clients = 1;
    int min_threads_per_server = 1;
    int threads_per_server = 16;
    int cuda_device = 0;
    std::string memory = kDefaultMemory;
    std::uint64_t offset = 0;
    std::uint64_t stride = 0;
    std::uint64_t payload_bytes = 128;
    std::uint64_t chunk_bytes = 0;
    std::uint32_t max_inflight = 16;
    std::uint32_t pipeline_depth = 1;
    bool reuse_local_buffer = false;
    int repeats = 5000;
    int warmup = 500;
    int trials = 5;
    int warmup_ops = 8;
    double duration_sec = 10.0;
    double start_delay_sec = 0.0;
    std::int64_t pace_us = 0;
    std::int64_t stagger_us = 0;
    int io_workers = 1;
    std::size_t cq_batch_size = 16;
    std::size_t mr_cache_limit = 16;
    std::vector<int> cpu_list;
    bool round_sync = true;
    bool debug = false;
    std::string rdma_log_level = "warn";
    std::string completion_mode = "raw-poll";
};

struct LatencySample {
    int server_index = 0;
    double latency_us = 0.0;
};

struct ThroughputWorkerResult {
    int server_index = 0;
    std::uint64_t ops = 0;
    std::uint64_t bytes = 0;
    double elapsed_sec = 0.0;
};

struct ThroughputRunResult {
    std::uint64_t ops = 0;
    std::uint64_t bytes = 0;
    double elapsed_sec = 0.0;
    std::vector<std::uint64_t> per_server_bytes;
    std::vector<double> per_server_elapsed_sec;
};

struct WorkerMapRow {
    int worker_id = 0;
    int server_index = 0;
    int server_local_worker = 0;
    int cpu = -1;
    std::uint64_t base_offset = 0;
};

template <typename T>
const T& pick_round_robin(const std::vector<T>& values, std::size_t index) {
    return values.size() == 1 ? values[0] : values[index % values.size()];
}

[[noreturn]] inline void usage_error(const std::string& message) {
    throw std::runtime_error("argument error: " + message);
}

inline std::string lower_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return value;
}

inline bool is_blank_arg(const std::string& value) {
    return !value.empty() &&
           std::all_of(
               value.begin(),
               value.end(),
               [](unsigned char ch) {
                   return std::isspace(ch) != 0;
               });
}

inline std::uint64_t parse_size(const std::string& text) {
    std::string value = lower_copy(text);
    if (value.empty()) {
        usage_error("empty size");
    }

    std::uint64_t scale = 1;
    auto strip_suffix = [&](const std::string& suffix,
                            std::uint64_t suffix_scale) {
        if (value.size() >= suffix.size() &&
            value.compare(value.size() - suffix.size(),
                          suffix.size(),
                          suffix) == 0) {
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
        if (bytes > static_cast<long double>(
                std::numeric_limits<std::uint64_t>::max())) {
            usage_error("size overflows uint64: " + text);
        }
        return static_cast<std::uint64_t>(bytes);
    } catch (const std::invalid_argument&) {
        usage_error("invalid size: " + text);
    } catch (const std::out_of_range&) {
        usage_error("size out of range: " + text);
    }
}

inline long long parse_ll(const std::string& name,
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

inline double parse_double(const std::string& name,
                           const std::string& value,
                           double min_value) {
    try {
        std::size_t consumed = 0;
        double parsed = std::stod(value, &consumed);
        if (consumed != value.size()) {
            usage_error("invalid " + name + ": " + value);
        }
        if (parsed < min_value) {
            usage_error(name + " outside allowed range: " + value);
        }
        return parsed;
    } catch (const std::invalid_argument&) {
        usage_error("invalid " + name + ": " + value);
    } catch (const std::out_of_range&) {
        usage_error(name + " out of range: " + value);
    }
}

inline std::string take_value(int& index,
                              int argc,
                              char* argv[],
                              const std::string& flag) {
    if (index + 1 >= argc) {
        usage_error(flag + " requires a value");
    }
    return argv[++index];
}

inline int parse_nonnegative_id(const std::string& name,
                                const std::string& text) {
    return static_cast<int>(
        parse_ll(name, text, 0, std::numeric_limits<int>::max()));
}

inline std::vector<int> parse_nonnegative_int_list(const std::string& text,
                                                   const std::string& flag) {
    if (text.empty()) {
        usage_error(flag + " cannot be empty");
    }

    std::vector<int> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t comma = text.find(',', start);
        std::string item = text.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            usage_error("empty item in " + flag);
        }

        std::size_t dash = item.find('-');
        if (dash == std::string::npos) {
            values.push_back(parse_nonnegative_id(flag + " item", item));
        } else {
            if (item.find('-', dash + 1) != std::string::npos) {
                usage_error("invalid range in " + flag + ": " + item);
            }
            int first = parse_nonnegative_id(flag + " range start",
                                             item.substr(0, dash));
            int last = parse_nonnegative_id(flag + " range end",
                                            item.substr(dash + 1));
            if (first > last) {
                usage_error("descending range in " + flag + ": " + item);
            }
            for (int value = first; value <= last; ++value) {
                values.push_back(value);
            }
        }

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    return values;
}

inline void parse_server_endpoints(const std::string& text,
                                   std::vector<std::string>& hosts,
                                   std::vector<std::uint16_t>& ports) {
    if (text.empty()) {
        usage_error("--server-endpoints cannot be empty");
    }

    hosts.clear();
    ports.clear();
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t comma = text.find(',', start);
        std::string item = text.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            usage_error("empty item in --server-endpoints");
        }

        std::size_t colon = item.rfind(':');
        if (colon == std::string::npos || colon == 0 ||
            colon + 1 == item.size()) {
            usage_error(
                "--server-endpoints items must be HOST:PORT: " + item);
        }
        hosts.push_back(item.substr(0, colon));
        ports.push_back(static_cast<std::uint16_t>(
            parse_ll("--server-endpoints port",
                     item.substr(colon + 1),
                     1,
                     65535)));

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
}

inline bool has_duplicates(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    return std::adjacent_find(values.begin(), values.end()) != values.end();
}

inline std::size_t endpoint_config_count(const Options& opts) {
    return std::max(
        std::max(opts.server_hosts.size(), opts.ib_devices.size()),
        opts.ib_ports.size());
}

inline std::string endpoint_label(const Options& opts, int server_index) {
    const std::size_t config_id =
        static_cast<std::size_t>(server_index) % endpoint_config_count(opts);
    return pick_round_robin(opts.server_hosts, config_id) + ":" +
           std::to_string(pick_round_robin(opts.server_ports, config_id));
}

inline std::vector<WorkerMapRow> expected_worker_map(const Options& opts) {
    std::vector<WorkerMapRow> rows;
    const int servers = static_cast<int>(opts.server_hosts.size());
    rows.reserve(static_cast<std::size_t>(servers * opts.threads_per_server));
    int worker_id = 0;
    for (int server_index = 0; server_index < servers; ++server_index) {
        for (int server_local_worker = 0;
             server_local_worker < opts.threads_per_server;
             ++server_local_worker) {
            const int cpu = opts.cpu_list.empty()
                ? -1
                : opts.cpu_list[static_cast<std::size_t>(worker_id)];
            const std::uint64_t global_worker_on_server =
                static_cast<std::uint64_t>(opts.client_id) *
                    static_cast<std::uint64_t>(opts.threads_per_server) +
                static_cast<std::uint64_t>(server_local_worker);
            rows.push_back(WorkerMapRow{
                worker_id,
                server_index,
                server_local_worker,
                cpu,
                opts.offset + global_worker_on_server * opts.stride});
            ++worker_id;
        }
    }
    return rows;
}

inline void finalize_options(Options& opts,
                             std::uint64_t default_payload_bytes,
                             std::uint64_t default_stride) {
    if (opts.server_hosts.empty()) {
        opts.server_hosts.push_back(opts.host);
        opts.server_ports.push_back(opts.port);
    }
    if (opts.ib_devices.empty()) {
        opts.ib_devices.push_back(opts.ib_device);
    }
    if (opts.ib_ports.empty()) {
        opts.ib_ports.push_back(opts.ib_port);
    }
    for (int port : opts.ib_ports) {
        if (port < 1) {
            usage_error("--ib-ports values must be >= 1");
        }
    }
    std::size_t configs = endpoint_config_count(opts);
    auto validate_broadcastable = [&](std::size_t size,
                                      const std::string& name) {
        if (size != 1 && size != configs) {
            usage_error(
                name + " must contain either 1 item or the same number of "
                "items as the largest endpoint list");
        }
    };
    validate_broadcastable(opts.server_hosts.size(), "--server-endpoints");
    validate_broadcastable(opts.server_ports.size(), "--server-endpoints");
    validate_broadcastable(opts.ib_devices.size(), "--ib-devices");
    validate_broadcastable(opts.ib_ports.size(), "--ib-ports");

    if (opts.payload_bytes == 0) {
        opts.payload_bytes = default_payload_bytes;
    }
    if (opts.chunk_bytes == 0) {
        opts.chunk_bytes = opts.payload_bytes;
    }
    if (opts.stride == 0) {
        opts.stride = default_stride == 0 ? opts.payload_bytes
                                          : default_stride;
    }
    if (opts.payload_bytes == 0) {
        usage_error("payload size must be positive");
    }
    if (opts.payload_bytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        usage_error("payload size exceeds size_t");
    }
    if (opts.chunk_bytes == 0) {
        usage_error("--chunk-bytes must be positive");
    }
    if (opts.chunk_bytes > opts.payload_bytes) {
        usage_error("--chunk-bytes must be <= payload size");
    }
    if (opts.max_inflight == 0) {
        usage_error("--max-inflight must be positive");
    }
    if (opts.pipeline_depth == 0) {
        usage_error("--pipeline-depth must be positive");
    }
    if (opts.stride < opts.payload_bytes) {
        usage_error("--stride must be >= payload size");
    }
    if (opts.memory != "cuda" && opts.memory != "host") {
        usage_error("--memory must be cuda or host");
    }
    if (opts.client_id < 0 || opts.client_id >= opts.clients) {
        usage_error("--client-id must be in [0, --clients)");
    }
    if (opts.threads_per_server < 1) {
        usage_error("--threads-per-server must be positive");
    }
    if (opts.min_threads_per_server < 1) {
        usage_error("--min-threads must be positive");
    }
    if (opts.min_threads_per_server > opts.threads_per_server) {
        usage_error("--min-threads must be <= --max-threads/--threads-per-server");
    }
    if (opts.repeats < 1) {
        usage_error("--repeats must be positive");
    }
    if (opts.trials < 1) {
        usage_error("--trials must be positive");
    }
    if (opts.warmup < 0) {
        usage_error("--warmup must be non-negative");
    }
    if (opts.warmup_ops < 0) {
        usage_error("--warmup-ops must be non-negative");
    }
    if (opts.duration_sec <= 0.0) {
        usage_error("--duration-sec must be positive");
    }
    if (opts.start_delay_sec < 0.0) {
        usage_error("--start-delay-sec must be non-negative");
    }
    if (opts.completion_mode != "raw-poll" &&
        opts.completion_mode != "callback") {
        usage_error("--completion-mode must be raw-poll or callback");
    }
    if (has_duplicates(opts.cpu_list)) {
        usage_error("--cpu-list contains duplicate CPUs");
    }

    const int total_threads =
        static_cast<int>(opts.server_hosts.size()) * opts.threads_per_server;
    if (!opts.cpu_list.empty() &&
        static_cast<int>(opts.cpu_list.size()) < total_threads) {
        usage_error("--cpu-list must contain at least total local threads");
    }
}

inline void parse_common_arg(Options& opts,
                             int& i,
                             int argc,
                             char* argv[],
                             bool& handled) {
    std::string arg = argv[i];
    handled = true;
    if (arg == "--host") {
        opts.host = take_value(i, argc, argv, arg);
    } else if (arg == "--port") {
        opts.port = static_cast<std::uint16_t>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1, 65535));
    } else if (arg == "--server-endpoints") {
        parse_server_endpoints(
            take_value(i, argc, argv, arg),
            opts.server_hosts,
            opts.server_ports);
    } else if (arg == "--ib-port") {
        opts.ib_port = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--ib-device") {
        opts.ib_device = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 0,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--ib-ports") {
        opts.ib_ports =
            parse_nonnegative_int_list(take_value(i, argc, argv, arg), arg);
    } else if (arg == "--ib-devices") {
        opts.ib_devices =
            parse_nonnegative_int_list(take_value(i, argc, argv, arg), arg);
    } else if (arg == "--client-id") {
        opts.client_id = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 0,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--clients") {
        opts.clients = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--min-threads" ||
               arg == "--min-threads-per-server") {
        opts.min_threads_per_server = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--max-threads" ||
               arg == "--max-threads-per-server" ||
               arg == "--threads-per-server") {
        opts.threads_per_server = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--memory") {
        opts.memory = lower_copy(take_value(i, argc, argv, arg));
    } else if (arg == "--cuda-device") {
        opts.cuda_device = static_cast<int>(
            parse_ll(arg, take_value(i, argc, argv, arg), 0,
                     std::numeric_limits<int>::max()));
    } else if (arg == "--offset") {
        opts.offset = parse_size(take_value(i, argc, argv, arg));
    } else if (arg == "--stride") {
        opts.stride = parse_size(take_value(i, argc, argv, arg));
    } else if (arg == "--payload-bytes" || arg == "--prompt-bytes") {
        opts.payload_bytes = parse_size(take_value(i, argc, argv, arg));
    } else if (arg == "--chunk-bytes") {
        opts.chunk_bytes = parse_size(take_value(i, argc, argv, arg));
    } else if (arg == "--max-inflight") {
        opts.max_inflight = static_cast<std::uint32_t>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1,
                     std::numeric_limits<std::uint32_t>::max()));
    } else if (arg == "--pipeline-depth") {
        opts.pipeline_depth = static_cast<std::uint32_t>(
            parse_ll(arg, take_value(i, argc, argv, arg), 1,
                     std::numeric_limits<std::uint32_t>::max()));
    } else if (arg == "--reuse-local-buffer") {
        opts.reuse_local_buffer = true;
    } else if (arg == "--no-reuse-local-buffer") {
        opts.reuse_local_buffer = false;
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
        opts.cpu_list =
            parse_nonnegative_int_list(take_value(i, argc, argv, arg), arg);
    } else if (arg == "--round-sync") {
        opts.round_sync = true;
    } else if (arg == "--no-round-sync") {
        opts.round_sync = false;
    } else if (arg == "--pace-us") {
        opts.pace_us = parse_ll(arg, take_value(i, argc, argv, arg), 0,
                                std::numeric_limits<std::int64_t>::max());
    } else if (arg == "--stagger-us") {
        opts.stagger_us = parse_ll(arg, take_value(i, argc, argv, arg), 0,
                                   std::numeric_limits<std::int64_t>::max());
    } else if (arg == "--start-delay-sec") {
        opts.start_delay_sec =
            parse_double(arg, take_value(i, argc, argv, arg), 0.0);
    } else if (arg == "--debug") {
        opts.debug = true;
    } else if (arg == "--completion-mode") {
        opts.completion_mode = lower_copy(take_value(i, argc, argv, arg));
    } else if (arg == "--rdma-log-level") {
        opts.rdma_log_level = lower_copy(take_value(i, argc, argv, arg));
    } else {
        handled = false;
    }
}

inline void set_spdlog_level_from_name(const std::string& level) {
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
inline void check_cuda(cudaError_t status, const char* what) {
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

inline std::unique_ptr<LocalBuffer> make_buffer(const Options& opts) {
    std::uint64_t buffer_bytes = opts.payload_bytes;
    if (!opts.reuse_local_buffer) {
        if (opts.pipeline_depth >
            std::numeric_limits<std::uint64_t>::max() / opts.payload_bytes) {
            throw std::runtime_error("local buffer size overflows uint64");
        }
        buffer_bytes =
            opts.payload_bytes * static_cast<std::uint64_t>(opts.pipeline_depth);
    }
    if (opts.memory == "host") {
        return std::make_unique<HostBuffer>(buffer_bytes);
    }

#ifdef RDMAPY_HAS_CUDA
    return std::make_unique<CudaBuffer>(buffer_bytes, opts.cuda_device);
#else
    throw std::runtime_error(
        "this binary was built without CUDA support; rerun with --memory host "
        "or rebuild after CMake can find CUDAToolkit");
#endif
}

inline void pin_current_thread_to_cpu(int cpu) {
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
    int server_index = 0;
    int server_local_worker = 0;
    std::shared_ptr<RdmaSession> session;
    std::unique_ptr<LocalBuffer> buffer;
    std::uint64_t base_offset = 0;
};

struct EndpointSharedResource {
    std::shared_ptr<rdmapp::device> device;
    std::shared_ptr<rdmapp::pd> pd;
};

inline double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

inline double percentile(std::vector<double> values, double pct) {
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

class MultiServerReadBench {
public:
    explicit MultiServerReadBench(const Options& opts)
        : opts_(opts) {
        if (!opts_.cpu_list.empty()) {
            pin_current_thread_to_cpu(opts_.cpu_list.front());
        }
        const int total_workers = total_threads();
        const std::size_t configs = endpoint_config_count(opts_);
        shared_resources_.reserve(configs);
        for (std::size_t config_id = 0; config_id < configs; ++config_id) {
            const int ib_device = pick_round_robin(opts_.ib_devices, config_id);
            const int ib_port = pick_round_robin(opts_.ib_ports, config_id);
            auto device = std::make_shared<rdmapp::device>(ib_device, ib_port);
            auto pd = std::make_shared<rdmapp::pd>(device);
            shared_resources_.push_back(
                EndpointSharedResource{std::move(device), std::move(pd)});
        }

        endpoints_.reserve(static_cast<std::size_t>(total_workers));
        workers_.reserve(static_cast<std::size_t>(total_workers));
        for (int server_index = 0; server_index < num_servers(); ++server_index) {
            for (int local_worker = 0;
                 local_worker < opts_.threads_per_server;
                 ++local_worker) {
                endpoints_.push_back(make_endpoint(server_index));
                add_worker(server_index, local_worker, *endpoints_.back());
            }
        }

        for (auto& worker : workers_) {
            prime_worker(worker);
        }
        if (opts_.completion_mode == "raw-poll") {
            stop_background_workers();
        }
    }

    int num_servers() const {
        return static_cast<int>(opts_.server_hosts.size());
    }

    int total_threads() const {
        return num_servers() * opts_.threads_per_server;
    }

    std::vector<WorkerMapRow> worker_map() const {
        std::vector<WorkerMapRow> rows;
        rows.reserve(workers_.size());
        for (std::size_t worker_id = 0; worker_id < workers_.size();
             ++worker_id) {
            const auto& worker = workers_[worker_id];
            const int cpu = opts_.cpu_list.empty()
                ? -1
                : opts_.cpu_list[worker_id];
            rows.push_back(WorkerMapRow{
                static_cast<int>(worker_id),
                worker.server_index,
                worker.server_local_worker,
                cpu,
                worker.base_offset});
        }
        return rows;
    }

    std::vector<LatencySample> measure_latency() {
        const int threads = total_threads();
        std::vector<std::vector<LatencySample>> per_worker(
            static_cast<std::size_t>(threads));
        StartGate start_gate(threads);
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
                    start_gate.arrive_and_wait();
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
                    auto& samples =
                        per_worker[static_cast<std::size_t>(worker_id)];
                    samples.reserve(static_cast<std::size_t>(opts_.repeats));

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

                        const double latency_us = read_once(worker);
                        if (iteration >= opts_.warmup) {
                            samples.push_back(
                                LatencySample{worker.server_index, latency_us});
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

        std::vector<LatencySample> flat;
        flat.reserve(static_cast<std::size_t>(threads * opts_.repeats));
        for (auto& worker_samples : per_worker) {
            flat.insert(flat.end(), worker_samples.begin(), worker_samples.end());
        }
        return flat;
    }

    std::vector<ThroughputWorkerResult> measure_throughput() {
        const int threads = total_threads();
        std::vector<ThroughputWorkerResult> results(
            static_cast<std::size_t>(threads));
        StartGate start_gate(threads);
        StartGate measure_gate(threads);
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
                    start_gate.arrive_and_wait();
                    auto& worker = workers_[static_cast<std::size_t>(worker_id)];
                    worker.buffer->activate();
                    if (opts_.stagger_us > 0) {
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(
                                opts_.stagger_us *
                                static_cast<std::int64_t>(worker_id)));
                    }
                    for (int i = 0; i < opts_.warmup_ops; ++i) {
                        read_once(worker);
                    }

                    measure_gate.arrive_and_wait();
                    const auto start = std::chrono::steady_clock::now();
                    const auto deadline = start +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(opts_.duration_sec));
                    std::uint64_t ops = 0;
                    auto now = start;
                    do {
                        read_once(worker);
                        ops += opts_.pipeline_depth;
                        now = std::chrono::steady_clock::now();
                    } while (now < deadline);

                    const double elapsed_sec =
                        std::chrono::duration<double>(now - start).count();
                    results[static_cast<std::size_t>(worker_id)] =
                        ThroughputWorkerResult{
                            worker.server_index,
                            ops,
                            ops * opts_.payload_bytes,
                            elapsed_sec};
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
        return results;
    }

    std::vector<ThroughputWorkerResult> measure_read_count(
        std::uint64_t count,
        const std::function<void()>& before_measure = {}) {
        auto run = measure_read_count_total(count, before_measure);
        std::vector<ThroughputWorkerResult> results(
            static_cast<std::size_t>(total_threads()));
        const int threads = total_threads();
        const std::uint64_t per_worker_ops = count;
        const std::uint64_t per_worker_bytes = count * opts_.payload_bytes;
        for (int worker_id = 0; worker_id < threads; ++worker_id) {
            auto& worker = workers_[static_cast<std::size_t>(worker_id)];
            results[static_cast<std::size_t>(worker_id)] =
                ThroughputWorkerResult{
                    worker.server_index,
                    per_worker_ops,
                    per_worker_bytes,
                    run.elapsed_sec};
        }
        return results;
    }

    ThroughputRunResult measure_read_count_total(
        std::uint64_t count,
        const std::function<void()>& before_measure = {},
        int active_threads_per_server = 0,
        bool send_done = true) {
        if (count == 0) {
            throw std::runtime_error("read count must be positive");
        }
        if (active_threads_per_server == 0) {
            active_threads_per_server = opts_.threads_per_server;
        }
        if (active_threads_per_server < 1 ||
            active_threads_per_server > opts_.threads_per_server) {
            throw std::runtime_error(
                "active_threads_per_server is outside the configured range");
        }
        const auto active_workers =
            active_worker_indices(active_threads_per_server);
        const int threads = static_cast<int>(active_workers.size());
        std::vector<std::uint64_t> per_server_bytes(
            static_cast<std::size_t>(num_servers()),
            0);
        std::vector<std::chrono::steady_clock::time_point> per_server_end(
            static_cast<std::size_t>(num_servers()));
        std::mutex per_server_end_mutex;
        for (int worker_id : active_workers) {
            const auto& worker = workers_[static_cast<std::size_t>(worker_id)];
            per_server_bytes[static_cast<std::size_t>(worker.server_index)] +=
                count * opts_.payload_bytes;
        }
        StartGate start_gate(threads);
        std::vector<std::thread> thread_pool;
        std::mutex exception_mutex;
        std::exception_ptr first_exception;
        std::chrono::steady_clock::time_point run_start;
        std::chrono::steady_clock::time_point run_end;
        std::barrier measure_start_gate(
            static_cast<std::ptrdiff_t>(threads),
            [&] {
                if (before_measure) {
                    before_measure();
                }
                run_start = std::chrono::steady_clock::now();
            });
        std::barrier measure_end_gate(
            static_cast<std::ptrdiff_t>(threads),
            [&run_end] {
                run_end = std::chrono::steady_clock::now();
            });
        thread_pool.reserve(static_cast<std::size_t>(threads));

        for (int active_id = 0; active_id < threads; ++active_id) {
            thread_pool.emplace_back([&, active_id] {
                try {
                    const int worker_id =
                        active_workers[static_cast<std::size_t>(active_id)];
                    if (!opts_.cpu_list.empty()) {
                        pin_current_thread_to_cpu(
                            opts_.cpu_list[static_cast<std::size_t>(worker_id)]);
                    }
                    start_gate.arrive_and_wait();
                    auto& worker = workers_[static_cast<std::size_t>(worker_id)];
                    auto local_buffer = make_buffer(opts_);
                    local_buffer->activate();
                    if (opts_.stagger_us > 0) {
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(
                                opts_.stagger_us *
                                static_cast<std::int64_t>(worker_id)));
                    }
                    for (int i = 0; i < opts_.warmup_ops; ++i) {
                        read_payload_once(worker, *local_buffer);
                    }

                    measure_start_gate.arrive_and_wait();
                    for (std::uint64_t i = 0; i < count; ++i) {
                        read_payload_once(worker, *local_buffer);
                    }
                    const auto worker_end = std::chrono::steady_clock::now();
                    {
                        std::lock_guard<std::mutex> lk(per_server_end_mutex);
                        auto& server_end =
                            per_server_end[static_cast<std::size_t>(
                                worker.server_index)];
                        if (server_end.time_since_epoch().count() == 0 ||
                            worker_end > server_end) {
                            server_end = worker_end;
                        }
                    }
                    measure_end_gate.arrive_and_wait();
                    if (send_done && opts_.completion_mode == "raw-poll") {
                        try {
                            worker.session->send_done_raw_poll();
                        } catch (const std::exception& e) {
                            spdlog::debug(
                                "best-effort done send failed after measured "
                                "READs: {}",
                                e.what());
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

        const double elapsed_sec =
            std::chrono::duration<double>(run_end - run_start).count();
        std::vector<double> per_server_elapsed_sec(per_server_end.size(), 0.0);
        for (std::size_t server_index = 0;
             server_index < per_server_end.size();
             ++server_index) {
            if (per_server_bytes[server_index] == 0 ||
                per_server_end[server_index].time_since_epoch().count() == 0) {
                continue;
            }
            per_server_elapsed_sec[server_index] =
                std::chrono::duration<double>(
                    per_server_end[server_index] - run_start).count();
        }
        return ThroughputRunResult{
            count * static_cast<std::uint64_t>(threads),
            count * opts_.payload_bytes * static_cast<std::uint64_t>(threads),
            elapsed_sec,
            std::move(per_server_bytes),
            std::move(per_server_elapsed_sec)};
    }

    void close_raw_poll_sessions() {
        if (opts_.completion_mode != "raw-poll") {
            return;
        }
        for (auto& worker : workers_) {
            try {
                worker.session->send_done_raw_poll();
            } catch (const std::exception& e) {
                spdlog::debug(
                    "best-effort final done send failed after throughput "
                    "sweep: {}",
                    e.what());
            }
        }
    }

private:
    std::vector<int> active_worker_indices(int active_threads_per_server) const {
        std::vector<int> indices;
        indices.reserve(
            static_cast<std::size_t>(num_servers() * active_threads_per_server));
        for (int server_index = 0; server_index < num_servers(); ++server_index) {
            const int server_base = server_index * opts_.threads_per_server;
            for (int local_worker = 0;
                 local_worker < active_threads_per_server;
                 ++local_worker) {
                indices.push_back(server_base + local_worker);
            }
        }
        return indices;
    }

    std::unique_ptr<RdmaEndpoint> make_endpoint(int server_index) const {
        const std::size_t config_id =
            static_cast<std::size_t>(server_index) % endpoint_config_count(opts_);
        int ib_device = pick_round_robin(opts_.ib_devices, config_id);
        int ib_port = pick_round_robin(opts_.ib_ports, config_id);
        const auto& resource = shared_resources_[config_id];
        return std::make_unique<RdmaEndpoint>(
            ib_port,
            ib_device,
            static_cast<std::uint32_t>(opts_.io_workers),
            opts_.cq_batch_size,
            resource.device,
            resource.pd);
    }

    void add_worker(int server_index,
                    int server_local_worker,
                    RdmaEndpoint& endpoint) {
        const std::size_t config_id =
            static_cast<std::size_t>(server_index) % endpoint_config_count(opts_);
        const std::string& host =
            pick_round_robin(opts_.server_hosts, config_id);
        std::uint16_t port =
            pick_round_robin(opts_.server_ports, config_id);
        auto session = endpoint.connect(host, port);
        session->set_mr_cache_limit(opts_.mr_cache_limit);

        WorkerState worker;
        worker.server_index = server_index;
        worker.server_local_worker = server_local_worker;
        worker.session = std::move(session);
        worker.buffer = make_buffer(opts_);
        const std::uint64_t global_worker_on_server =
            static_cast<std::uint64_t>(opts_.client_id) *
                static_cast<std::uint64_t>(opts_.threads_per_server) +
            static_cast<std::uint64_t>(server_local_worker);
        worker.base_offset =
            opts_.offset + global_worker_on_server * opts_.stride;
        workers_.push_back(std::move(worker));
    }

    void prime_worker(WorkerState& worker) const {
        worker.buffer->activate();
        worker.session->read_into(
            worker.base_offset,
            worker.buffer->addr(),
            opts_.payload_bytes,
            opts_.payload_bytes,
            nullptr,
            worker.buffer->length(),
            1,
            opts_.debug);
    }

    double read_once(WorkerState& worker) const {
        if (opts_.completion_mode == "raw-poll") {
            if (opts_.pipeline_depth == 1 &&
                opts_.chunk_bytes >= opts_.payload_bytes) {
                return worker.session->read_into_transfer_latency_us_raw_poll(
                    worker.base_offset,
                    worker.buffer->addr(),
                    opts_.payload_bytes,
                    opts_.payload_bytes,
                    nullptr,
                    worker.buffer->length(),
                    1,
                    opts_.debug);
            }
            return worker.session->read_repeated_into_transfer_latency_us_raw_poll(
                worker.base_offset,
                worker.buffer->addr(),
                opts_.payload_bytes,
                opts_.chunk_bytes,
                opts_.pipeline_depth,
                opts_.reuse_local_buffer,
                nullptr,
                worker.buffer->length(),
                opts_.max_inflight,
                opts_.debug);
        }
        if (opts_.pipeline_depth != 1) {
            throw std::runtime_error(
                "--pipeline-depth > 1 is only supported with raw-poll");
        }
        return worker.session->read_into_transfer_latency_us(
            worker.base_offset,
            worker.buffer->addr(),
            opts_.payload_bytes,
            opts_.chunk_bytes,
            nullptr,
            worker.buffer->length(),
            opts_.max_inflight,
            opts_.debug);
    }

    double read_payload_once(WorkerState& worker) const {
        return read_payload_once(worker, *worker.buffer);
    }

    double read_payload_once(WorkerState& worker, LocalBuffer& buffer) const {
        if (opts_.completion_mode == "raw-poll") {
            return worker.session->read_into_transfer_latency_us_raw_poll(
                worker.base_offset,
                buffer.addr(),
                opts_.payload_bytes,
                opts_.payload_bytes,
                nullptr,
                buffer.length(),
                1,
                opts_.debug);
        }
        return worker.session->read_into_transfer_latency_us(
            worker.base_offset,
            buffer.addr(),
            opts_.payload_bytes,
            opts_.payload_bytes,
            nullptr,
            buffer.length(),
            1,
            opts_.debug);
    }

    void stop_background_workers() {
        for (auto& endpoint : endpoints_) {
            endpoint->stop_background_workers();
        }
    }

    const Options& opts_;
    std::vector<EndpointSharedResource> shared_resources_;
    std::vector<std::unique_ptr<RdmaEndpoint>> endpoints_;
    std::vector<WorkerState> workers_;
};

inline std::vector<double> latency_values(
    const std::vector<LatencySample>& samples,
    int server_index) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        if (server_index < 0 || sample.server_index == server_index) {
            values.push_back(sample.latency_us);
        }
    }
    return values;
}

inline double bytes_to_gbps(std::uint64_t bytes, double seconds) {
    return seconds > 0.0
        ? static_cast<double>(bytes) * 8.0 / seconds / 1.0e9
        : 0.0;
}

inline double bytes_to_gibps(std::uint64_t bytes, double seconds) {
    return seconds > 0.0
        ? static_cast<double>(bytes) / seconds /
              static_cast<double>(kGiB)
        : 0.0;
}

} // namespace rdma_read_bench
