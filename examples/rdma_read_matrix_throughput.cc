#include "rdma_read_bench_common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <asio/connect.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/read.hpp>
#include <asio/write.hpp>

namespace {

using rdma_read_bench::Options;

constexpr std::uint64_t kSyncMagic = 0x52444d4153594e43ull; // RDMASYNC
constexpr std::uint32_t kSyncKindBarrier = 1;
constexpr std::uint32_t kSyncKindStats = 2;

struct SyncWire {
    std::uint64_t magic = kSyncMagic;
    std::uint32_t kind = 0;
    std::uint32_t reserved = 0;
    std::uint64_t phase = 0;
    std::uint64_t threads = 0;
    std::uint64_t bytes = 0;
    std::uint64_t elapsed_ns = 0;
};

struct ReadBwOptions {
    Options common;
    std::uint64_t count = 3000;
    std::uint64_t prewarm_count = 0;
    std::string trial_reducer = "mean";
    int only_total_threads = 0;
    bool prewarm_count_set = false;
    std::string sync_role = "none";
    std::string sync_host = "127.0.0.1";
    std::uint16_t sync_port = 2444;
    int sync_clients = 2;
    std::string sync_output = "local";
    bool server_breakdown = false;
};

class CrossClientSync {
public:
    explicit CrossClientSync(const ReadBwOptions& opts)
        : role_(opts.sync_role),
          output_(opts.sync_output) {
        if (role_ == "none") {
            return;
        }
        if (role_ == "server") {
            asio::ip::tcp::acceptor acceptor(
                io_ctx_,
                {asio::ip::tcp::v4(), opts.sync_port});
            const int peers = opts.sync_clients - 1;
            sockets_.reserve(static_cast<std::size_t>(peers));
            for (int i = 0; i < peers; ++i) {
                sockets_.emplace_back(io_ctx_);
                acceptor.accept(sockets_.back());
            }
            return;
        }
        if (role_ == "client") {
            asio::ip::tcp::resolver resolver(io_ctx_);
            auto endpoints = resolver.resolve(
                opts.sync_host,
                std::to_string(opts.sync_port));
            sockets_.emplace_back(io_ctx_);
            asio::connect(sockets_.back(), endpoints);
            return;
        }
        throw std::runtime_error("--sync-role must be none, server, or client");
    }

    bool should_print() const {
        if (role_ == "none") {
            return true;
        }
        if (output_ == "local") {
            return true;
        }
        return role_ == "server" && output_ == "aggregate";
    }

    void begin_phase(std::uint64_t phase) {
        if (role_ == "none") {
            return;
        }
        if (role_ == "server") {
            for (auto& socket : sockets_) {
                SyncWire wire{};
                read_wire(socket, wire);
                validate_wire(wire, kSyncKindBarrier, phase);
            }
            SyncWire release{};
            release.kind = kSyncKindBarrier;
            release.phase = phase;
            phase_start_ = std::chrono::steady_clock::now();
            for (auto& socket : sockets_) {
                write_wire(socket, release);
            }
            return;
        }

        SyncWire wire{};
        wire.kind = kSyncKindBarrier;
        wire.phase = phase;
        write_wire(sockets_.front(), wire);
        SyncWire release{};
        read_wire(sockets_.front(), release);
        validate_wire(release, kSyncKindBarrier, phase);
    }

    std::optional<double> report(std::uint64_t phase,
                                 std::uint64_t threads,
                                 std::uint64_t bytes,
                                 double elapsed_sec) {
        if (role_ == "none") {
            return rdma_read_bench::bytes_to_gbps(bytes, elapsed_sec);
        }

        const std::uint64_t elapsed_ns =
            static_cast<std::uint64_t>(elapsed_sec * 1.0e9);
        if (role_ == "client") {
            SyncWire stats{};
            stats.kind = kSyncKindStats;
            stats.phase = phase;
            stats.threads = threads;
            stats.bytes = bytes;
            stats.elapsed_ns = elapsed_ns;
            write_wire(sockets_.front(), stats);
            SyncWire release{};
            read_wire(sockets_.front(), release);
            validate_wire(release, kSyncKindStats, phase);
            if (output_ == "local") {
                return rdma_read_bench::bytes_to_gbps(bytes, elapsed_sec);
            }
            return std::nullopt;
        }

        std::uint64_t total_bytes = bytes;
        for (auto& socket : sockets_) {
            SyncWire stats{};
            read_wire(socket, stats);
            validate_wire(stats, kSyncKindStats, phase);
            if (stats.threads != threads) {
                throw std::runtime_error(
                    "sync peer reported a different thread count");
            }
            total_bytes += stats.bytes;
        }
        const auto phase_end = std::chrono::steady_clock::now();
        SyncWire release{};
        release.kind = kSyncKindStats;
        release.phase = phase;
        for (auto& socket : sockets_) {
            write_wire(socket, release);
        }
        if (output_ == "aggregate") {
            return rdma_read_bench::bytes_to_gbps(
                total_bytes,
                std::chrono::duration<double>(phase_end - phase_start_).count());
        }
        if (output_ == "local") {
            return rdma_read_bench::bytes_to_gbps(bytes, elapsed_sec);
        }
        return std::nullopt;
    }

private:
    static void write_wire(asio::ip::tcp::socket& socket,
                           const SyncWire& wire) {
        asio::write(socket, asio::buffer(&wire, sizeof(wire)));
    }

    static void read_wire(asio::ip::tcp::socket& socket,
                          SyncWire& wire) {
        asio::read(socket, asio::buffer(&wire, sizeof(wire)));
    }

    static void validate_wire(const SyncWire& wire,
                              std::uint32_t expected_kind,
                              std::uint64_t expected_phase) {
        if (wire.magic != kSyncMagic ||
            wire.kind != expected_kind ||
            wire.phase != expected_phase) {
            throw std::runtime_error("bad cross-client sync message");
        }
    }

    std::string role_;
    std::string output_;
    std::chrono::steady_clock::time_point phase_start_;
    asio::io_context io_ctx_;
    std::vector<asio::ip::tcp::socket> sockets_;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [options]\n\n"
        << "Topology options:\n"
        << "  --server-endpoints LIST     HOST:PORT[,HOST:PORT...]\n"
        << "  --host HOST                 default: 127.0.0.1\n"
        << "  --port PORT                 default: 2333\n"
        << "  --clients N                 metadata for 2-client runs, default: 1\n"
        << "  --client-id ID              metadata/offset partition, default: 0\n"
        << "  --min-threads N             per-server min, default: 1\n"
        << "  --max-threads N             per-server max, default: 16\n"
        << "  --threads-per-server N      alias for --max-threads\n"
        << "  --only-total-threads N      run exactly N total local threads\n"
        << "  --ib-devices LIST           local RDMA device list, e.g. 0,1\n"
        << "  --ib-ports LIST             local RDMA port list, e.g. 1,1\n"
        << "  --ib-device DEV             default: 0\n"
        << "  --ib-port PORT              default: 1\n\n"
        << "Payload options:\n"
        << "  --payload-bytes BYTES       default: 4m; accepts k/m/g suffix\n"
        << "  --prompt-bytes BYTES        alias for --payload-bytes\n"
        << "  --memory host               read_bw-compatible mode is host-memory only\n"
        << "  --offset BYTES              default: 0\n"
        << "  --stride BYTES              default: payload size\n\n"
        << "Benchmark options:\n"
        << "  --count N                   4MB READs per worker, default: 3000\n"
        << "  --prewarm-count N           hidden prewarm READs before output\n"
        << "  --warmup-ops N              warmup READs per worker, default: 8\n"
        << "  --trials N                  default: 1\n"
        << "  --trial-reducer MODE        mean|median|max, default: mean\n"
        << "  --start-delay-sec SEC       sleep before connecting, default: 0\n"
        << "  --stagger-us N              default: 0\n"
        << "  --cpu-list LIST             pin local workers, e.g. 0,2,4,6\n"
        << "  --completion-mode MODE      raw-poll|callback, default: raw-poll\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n"
        << "Cross-client aggregate options:\n"
        << "  --sync-role none|server|client  default: none\n"
        << "  --sync-host HOST                sync server host for client role\n"
        << "  --sync-port PORT                default: 2444\n"
        << "  --sync-clients N                default: 2\n"
        << "  --sync-output local|aggregate|none  default: local\n"
        << "  --server-breakdown              print local per-server bandwidth columns\n";
}

ReadBwOptions parse_args(int argc, char* argv[]) {
    ReadBwOptions parsed;
    Options& opts = parsed.common;
    opts.memory = "host";
    opts.payload_bytes = 4 * rdma_read_bench::kMiB;
    opts.chunk_bytes = 4 * rdma_read_bench::kMiB;
    opts.max_inflight = 1;
    opts.pipeline_depth = 1;
    opts.reuse_local_buffer = true;
    opts.stride = 0;
    opts.trials = 1;
    opts.round_sync = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (rdma_read_bench::is_blank_arg(arg)) {
            rdma_read_bench::usage_error(
                "blank argument detected. In shell commands, a "
                "line-continuation backslash must be the final character on "
                "the line");
        }
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        bool handled = false;
        rdma_read_bench::parse_common_arg(opts, i, argc, argv, handled);
        if (handled) {
            continue;
        }

        if (arg == "--count") {
            parsed.count = static_cast<std::uint64_t>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    1,
                    std::numeric_limits<long long>::max()));
        } else if (arg == "--prewarm-count") {
            parsed.prewarm_count = static_cast<std::uint64_t>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    0,
                    std::numeric_limits<long long>::max()));
            parsed.prewarm_count_set = true;
        } else if (arg == "--duration-sec") {
            opts.duration_sec = rdma_read_bench::parse_double(
                arg,
                rdma_read_bench::take_value(i, argc, argv, arg),
                0.001);
        } else if (arg == "--warmup-ops") {
            opts.warmup_ops = static_cast<int>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    0,
                    std::numeric_limits<int>::max()));
        } else if (arg == "--trials") {
            opts.trials = static_cast<int>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    1,
                    std::numeric_limits<int>::max()));
        } else if (arg == "--trial-reducer") {
            parsed.trial_reducer = rdma_read_bench::lower_copy(
                rdma_read_bench::take_value(i, argc, argv, arg));
        } else if (arg == "--only-total-threads") {
            parsed.only_total_threads = static_cast<int>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    1,
                    std::numeric_limits<int>::max()));
        } else if (arg == "--sync-role") {
            parsed.sync_role = rdma_read_bench::lower_copy(
                rdma_read_bench::take_value(i, argc, argv, arg));
        } else if (arg == "--sync-host") {
            parsed.sync_host = rdma_read_bench::take_value(i, argc, argv, arg);
        } else if (arg == "--sync-port") {
            parsed.sync_port = static_cast<std::uint16_t>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    1,
                    65535));
        } else if (arg == "--sync-clients") {
            parsed.sync_clients = static_cast<int>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    2,
                    std::numeric_limits<int>::max()));
        } else if (arg == "--sync-output") {
            parsed.sync_output = rdma_read_bench::lower_copy(
                rdma_read_bench::take_value(i, argc, argv, arg));
        } else if (arg == "--server-breakdown") {
            parsed.server_breakdown = true;
        } else {
            rdma_read_bench::usage_error("unknown argument: " + arg);
        }
    }

    rdma_read_bench::finalize_options(
        opts, 4 * rdma_read_bench::kMiB, 0);
    if (opts.memory != "host") {
        rdma_read_bench::usage_error(
            "this read_bw-compatible throughput test is host-memory only; "
            "remove --memory cuda or use --memory host");
    }
    if (parsed.trial_reducer != "max" &&
        parsed.trial_reducer != "median" &&
        parsed.trial_reducer != "mean") {
        rdma_read_bench::usage_error(
            "--trial-reducer must be max, median, or mean");
    }
    if (parsed.sync_role != "none" &&
        parsed.sync_role != "server" &&
        parsed.sync_role != "client") {
        rdma_read_bench::usage_error(
            "--sync-role must be none, server, or client");
    }
    if (parsed.sync_output != "local" &&
        parsed.sync_output != "aggregate" &&
        parsed.sync_output != "none") {
        rdma_read_bench::usage_error(
            "--sync-output must be local, aggregate, or none");
    }
    if (parsed.only_total_threads > 0) {
        const int servers = static_cast<int>(opts.server_hosts.size());
        if (parsed.only_total_threads % servers != 0) {
            rdma_read_bench::usage_error(
                "--only-total-threads must be divisible by the number of servers");
        }
        const int per_server = parsed.only_total_threads / servers;
        if (per_server < 1) {
            rdma_read_bench::usage_error("--only-total-threads is too small");
        }
        opts.min_threads_per_server = per_server;
        opts.threads_per_server = per_server;
        const int total_threads = per_server * servers;
        if (!opts.cpu_list.empty() &&
            static_cast<int>(opts.cpu_list.size()) < total_threads) {
            rdma_read_bench::usage_error(
                "--cpu-list must contain at least --only-total-threads CPUs");
        }
        if (!parsed.prewarm_count_set) {
            parsed.prewarm_count = std::min<std::uint64_t>(parsed.count, 300);
        }
    }
    return parsed;
}

double reduce_trials(std::vector<double> values, const std::string& reducer) {
    if (values.empty()) {
        return 0.0;
    }
    if (reducer == "max") {
        return *std::max_element(values.begin(), values.end());
    }
    if (reducer == "median") {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        ReadBwOptions parsed = parse_args(argc, argv);
        Options opts = parsed.common;
        rdma_read_bench::set_spdlog_level_from_name(opts.rdma_log_level);
        CrossClientSync sync(parsed);

        std::cout << std::fixed << std::setprecision(4);
        if (sync.should_print()) {
            std::cout
                << "# data_path=rdma_read_bw_count_" << opts.completion_mode
                << " clients=" << opts.clients
                << " client_id=" << opts.client_id
                << " servers=" << opts.server_hosts.size()
                << " min_threads_per_server=" << opts.min_threads_per_server
                << " max_threads_per_server=" << opts.threads_per_server
                << " max_total_local_threads="
                << opts.threads_per_server *
                       static_cast<int>(opts.server_hosts.size())
                << " only_total_threads=" << parsed.only_total_threads
                << " payload_bytes=" << opts.payload_bytes
                << " memory=" << opts.memory
                << " count=" << parsed.count
                << " prewarm_count=" << parsed.prewarm_count
                << " warmup_ops=" << opts.warmup_ops
                << " trials=" << opts.trials
                << " trial_reducer=" << parsed.trial_reducer
                << " persistent_qps=true"
                << " shared_device_pd=true"
                << " server_breakdown="
                << (parsed.server_breakdown ? "true" : "false")
                << " server_breakdown_time=per_server_last_worker"
                << " sync_role=" << parsed.sync_role
                << " sync_output=" << parsed.sync_output
                << "\n";
            std::cout.flush();
        }
        if (opts.start_delay_sec > 0.0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(opts.start_delay_sec));
        }
        if (sync.should_print()) {
            if (parsed.server_breakdown) {
                std::cout << "threads,total_gbps";
                for (std::size_t server_index = 0;
                     server_index < opts.server_hosts.size();
                     ++server_index) {
                    std::cout << ",server" << server_index << "_gbps";
                }
                std::cout << "\n";
            } else {
                std::cout << "threads,gbps\n";
            }
        }
        std::uint64_t phase = 0;
        rdma_read_bench::MultiServerReadBench bench(opts);

        for (int threads_per_server = opts.min_threads_per_server;
             threads_per_server <= opts.threads_per_server;
             ++threads_per_server) {
            const int total_threads =
                threads_per_server *
                static_cast<int>(opts.server_hosts.size());

            if (parsed.prewarm_count > 0) {
                const std::uint64_t warmup_phase = phase++;
                auto warmup_result = bench.measure_read_count_total(
                    parsed.prewarm_count,
                    [&sync, warmup_phase] {
                        sync.begin_phase(warmup_phase);
                    },
                    threads_per_server,
                    false);
                (void)sync.report(
                    warmup_phase,
                    static_cast<std::uint64_t>(total_threads),
                    warmup_result.bytes,
                    warmup_result.elapsed_sec);
            }

            std::vector<double> trial_gbps_values;
            trial_gbps_values.reserve(static_cast<std::size_t>(opts.trials));
            std::vector<std::vector<double>> trial_server_gbps_values(
                opts.server_hosts.size());
            for (auto& values : trial_server_gbps_values) {
                values.reserve(static_cast<std::size_t>(opts.trials));
            }

            for (int trial = 0; trial < opts.trials; ++trial) {
                const std::uint64_t trial_phase = phase++;
                auto result = bench.measure_read_count_total(
                    parsed.count,
                    [&sync, trial_phase] {
                        sync.begin_phase(trial_phase);
                    },
                    threads_per_server,
                    false);
                auto gbps = sync.report(
                    trial_phase,
                    static_cast<std::uint64_t>(total_threads),
                    result.bytes,
                    result.elapsed_sec);
                if (gbps.has_value()) {
                    trial_gbps_values.push_back(*gbps);
                }
                if (gbps.has_value() && parsed.server_breakdown) {
                    for (std::size_t server_index = 0;
                         server_index < result.per_server_bytes.size();
                         ++server_index) {
                        trial_server_gbps_values[server_index].push_back(
                            rdma_read_bench::bytes_to_gbps(
                                result.per_server_bytes[server_index],
                                result.per_server_elapsed_sec[server_index]));
                    }
                }
            }

            if (!sync.should_print()) {
                continue;
            }
            std::cout << total_threads << ","
                      << reduce_trials(
                             std::move(trial_gbps_values),
                             parsed.trial_reducer);
            if (parsed.server_breakdown) {
                for (auto& values : trial_server_gbps_values) {
                    std::cout << ","
                              << reduce_trials(
                                     std::move(values),
                                     parsed.trial_reducer);
                }
            }
            std::cout << "\n";
            std::cout.flush();
        }
        bench.close_raw_poll_sessions();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_read_matrix_throughput failed: "
                  << e.what() << "\n";
        return 1;
    }
}
