#include "rdma_read_bench_common.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct SyncWire {
    std::uint64_t magic = kSyncMagic;
    std::uint32_t kind = 0;
    std::uint32_t reserved = 0;
    std::uint64_t phase = 0;
};

struct MatrixLatencyOptions {
    Options common;
    std::string sync_role = "none";
    std::string sync_host = "127.0.0.1";
    std::uint16_t sync_port = 2444;
    int sync_clients = 2;
    std::string sync_output = "local";
};

class CrossClientBarrier {
public:
    explicit CrossClientBarrier(const MatrixLatencyOptions& opts)
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
        return role_ == "none" || output_ == "local";
    }

    void begin_phase(std::uint64_t phase) {
        if (role_ == "none") {
            return;
        }
        if (role_ == "server") {
            for (auto& socket : sockets_) {
                SyncWire wire{};
                read_wire(socket, wire);
                validate_wire(wire, phase);
            }
            SyncWire release{};
            release.kind = kSyncKindBarrier;
            release.phase = phase;
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
        validate_wire(release, phase);
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
                              std::uint64_t expected_phase) {
        if (wire.magic != kSyncMagic ||
            wire.kind != kSyncKindBarrier ||
            wire.phase != expected_phase) {
            throw std::runtime_error("bad cross-client sync message");
        }
    }

    std::string role_;
    std::string output_;
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
        << "  --ib-devices LIST           local RDMA device list, e.g. 0,1\n"
        << "  --ib-ports LIST             local RDMA port list, e.g. 1,1\n"
        << "  --ib-device DEV             default: 0\n"
        << "  --ib-port PORT              default: 1\n\n"
        << "Payload options:\n"
        << "  --payload-bytes BYTES       default: 1kB; accepts kB/MB/GB suffix\n"
        << "  --prompt-bytes BYTES        alias for --payload-bytes\n"
        << "  --memory cuda|host          default: "
        << rdma_read_bench::kDefaultMemory << "\n"
        << "  --cuda-device DEV           default: 0\n"
        << "  --offset BYTES              default: 0\n"
        << "  --stride BYTES              default: 4k\n\n"
        << "Benchmark options:\n"
        << "  --repeats N                 default: 5000\n"
        << "  --warmup N                  default: 500\n"
        << "  --trials N                  default: 5\n"
        << "  --round-sync                default; sync local workers before each READ\n"
        << "  --no-round-sync             let each worker loop independently\n"
        << "  --start-delay-sec SEC       sleep before connecting, default: 0\n"
        << "  --pace-us N                 default: 0\n"
        << "  --stagger-us N              default: 0\n"
        << "  --cpu-list LIST             pin local workers, e.g. 0,2,4,6\n"
        << "  --completion-mode MODE      raw-poll|callback, default: raw-poll\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n"
        << "Cross-client sync options:\n"
        << "  --sync-role none|server|client  default: none\n"
        << "  --sync-host HOST                sync server host for client role\n"
        << "  --sync-port PORT                default: 2444\n"
        << "  --sync-clients N                default: 2\n"
        << "  --sync-output local|none        default: local\n";
}

MatrixLatencyOptions parse_args(int argc, char* argv[]) {
    MatrixLatencyOptions parsed;
    Options& opts = parsed.common;
    opts.payload_bytes = rdma_read_bench::kKiB;
    opts.stride = 4 * rdma_read_bench::kKiB;

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

        if (arg == "--repeats") {
            opts.repeats = static_cast<int>(
                rdma_read_bench::parse_ll(
                    arg,
                    rdma_read_bench::take_value(i, argc, argv, arg),
                    1,
                    std::numeric_limits<int>::max()));
        } else if (arg == "--warmup") {
            opts.warmup = static_cast<int>(
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
        } else if (arg == "--sync-role") {
            parsed.sync_role = rdma_read_bench::lower_copy(
                rdma_read_bench::take_value(i, argc, argv, arg));
        } else if (arg == "--sync-host") {
            parsed.sync_host =
                rdma_read_bench::take_value(i, argc, argv, arg);
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
        } else {
            rdma_read_bench::usage_error("unknown argument: " + arg);
        }
    }

    rdma_read_bench::finalize_options(
        opts, rdma_read_bench::kKiB, 4 * rdma_read_bench::kKiB);
    if (parsed.sync_role != "none" &&
        parsed.sync_role != "server" &&
        parsed.sync_role != "client") {
        rdma_read_bench::usage_error(
            "--sync-role must be none, server, or client");
    }
    if (parsed.sync_output != "local" &&
        parsed.sync_output != "none") {
        rdma_read_bench::usage_error(
            "--sync-output must be local or none");
    }
    return parsed;
}

void print_latency_row(int threads,
                       const std::vector<double>& values) {
    std::cout << threads << ","
              << rdma_read_bench::mean(values)
              << "\n";
}

void print_worker_map(const Options& opts) {
    for (const auto& row : rdma_read_bench::expected_worker_map(opts)) {
        std::cout << "# worker_map"
                  << " worker_id=" << row.worker_id
                  << " server_idx=" << row.server_index
                  << " server_worker=" << row.server_local_worker
                  << " endpoint="
                  << rdma_read_bench::endpoint_label(opts, row.server_index)
                  << " cpu=";
        if (row.cpu >= 0) {
            std::cout << row.cpu;
        } else {
            std::cout << "unbound";
        }
        std::cout << " offset=" << row.base_offset << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        MatrixLatencyOptions parsed = parse_args(argc, argv);
        Options opts = parsed.common;
        rdma_read_bench::set_spdlog_level_from_name(opts.rdma_log_level);
        CrossClientBarrier sync(parsed);

        std::cout << std::fixed << std::setprecision(4);
        if (sync.should_print()) {
            std::cout
                << "# data_path=rdma_read_latency_raw_poll"
                << " clients=" << opts.clients
                << " client_id=" << opts.client_id
                << " servers=" << opts.server_hosts.size()
                << " min_threads_per_server=" << opts.min_threads_per_server
                << " max_threads_per_server=" << opts.threads_per_server
                << " max_total_local_threads="
                << opts.threads_per_server *
                       static_cast<int>(opts.server_hosts.size())
                << " payload_bytes=" << opts.payload_bytes
                << " repeats=" << opts.repeats
                << " warmup=" << opts.warmup
                << " trials=" << opts.trials
                << " round_sync=" << (opts.round_sync ? "true" : "false")
                << " sync_role=" << parsed.sync_role
                << " sync_output=" << parsed.sync_output
                << "\n";
            print_worker_map(opts);
            std::cout.flush();
        }
        if (opts.start_delay_sec > 0.0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(opts.start_delay_sec));
        }
        if (sync.should_print()) {
            std::cout << "threads,avg_us\n";
        }

        std::uint64_t phase = 0;
        for (int threads_per_server = opts.min_threads_per_server;
             threads_per_server <= opts.threads_per_server;
             ++threads_per_server) {
            Options run_opts = opts;
            run_opts.threads_per_server = threads_per_server;

            std::vector<rdma_read_bench::LatencySample> all_samples;
            for (int trial = 0; trial < run_opts.trials; ++trial) {
                const std::uint64_t trial_phase = phase++;
                rdma_read_bench::MultiServerReadBench bench(run_opts);
                auto samples = bench.measure_latency(
                    [&sync, trial_phase] {
                        sync.begin_phase(trial_phase);
                    });
                bench.close_raw_poll_sessions();
                all_samples.insert(all_samples.end(),
                                   samples.begin(),
                                   samples.end());
            }

            if (!sync.should_print()) {
                continue;
            }
            auto overall =
                rdma_read_bench::latency_values(all_samples, -1);
            print_latency_row(
                run_opts.threads_per_server *
                    static_cast<int>(run_opts.server_hosts.size()),
                overall);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_read_matrix_latency failed: " << e.what() << "\n";
        return 1;
    }
}
