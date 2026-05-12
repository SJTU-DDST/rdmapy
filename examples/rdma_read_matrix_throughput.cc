#include "rdma_read_bench_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using rdma_read_bench::Options;

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
        << "  --payload-bytes BYTES       default: 4m; accepts k/m/g suffix\n"
        << "  --prompt-bytes BYTES        alias for --payload-bytes\n"
        << "  --chunk-bytes BYTES         default: 4m\n"
        << "  --pipeline-depth N          default: 16 full payload READs per worker\n"
        << "  --max-inflight N            default: 16 outstanding READs per worker\n"
        << "  --memory cuda|host          default: "
        << rdma_read_bench::kDefaultMemory << "\n"
        << "  --cuda-device DEV           default: 0\n"
        << "  --offset BYTES              default: 0\n"
        << "  --stride BYTES              default: payload size\n\n"
        << "Benchmark options:\n"
        << "  --duration-sec SEC          measured duration, default: 10\n"
        << "  --warmup-ops N              warmup READs per worker, default: 8\n"
        << "  --trials N                  default: 1\n"
        << "  --start-delay-sec SEC       sleep before connecting, default: 0\n"
        << "  --stagger-us N              default: 0\n"
        << "  --cpu-list LIST             pin local workers, e.g. 0,2,4,6\n"
        << "  --completion-mode MODE      raw-poll|callback, default: raw-poll\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n";
}

Options parse_args(int argc, char* argv[]) {
    Options opts;
    opts.payload_bytes = 4 * rdma_read_bench::kMiB;
    opts.chunk_bytes = 4 * rdma_read_bench::kMiB;
    opts.max_inflight = 16;
    opts.pipeline_depth = 16;
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

        if (arg == "--duration-sec") {
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
        } else {
            rdma_read_bench::usage_error("unknown argument: " + arg);
        }
    }

    rdma_read_bench::finalize_options(
        opts, 4 * rdma_read_bench::kMiB, 0);
    return opts;
}

void print_throughput_row(int threads,
                          std::uint64_t bytes,
                          double seconds) {
    const double gbps = rdma_read_bench::bytes_to_gbps(bytes, seconds);
    std::cout << threads << ","
              << gbps
              << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        Options opts = parse_args(argc, argv);
        rdma_read_bench::set_spdlog_level_from_name(opts.rdma_log_level);

        std::cout << std::fixed << std::setprecision(4);
        std::cout
            << "# data_path=rdma_read_throughput_raw_poll"
            << " clients=" << opts.clients
            << " client_id=" << opts.client_id
            << " servers=" << opts.server_hosts.size()
            << " min_threads_per_server=" << opts.min_threads_per_server
            << " max_threads_per_server=" << opts.threads_per_server
            << " max_total_local_threads="
            << opts.threads_per_server *
                   static_cast<int>(opts.server_hosts.size())
            << " payload_bytes=" << opts.payload_bytes
            << " chunk_bytes=" << opts.chunk_bytes
            << " max_inflight=" << opts.max_inflight
            << " pipeline_depth=" << opts.pipeline_depth
            << " duration_sec=" << opts.duration_sec
            << " warmup_ops=" << opts.warmup_ops
            << " trials=" << opts.trials
            << "\n";
        std::cout.flush();
        if (opts.start_delay_sec > 0.0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(opts.start_delay_sec));
        }
        std::cout
            << "threads,gbps\n";

        for (int threads_per_server = opts.min_threads_per_server;
             threads_per_server <= opts.threads_per_server;
             ++threads_per_server) {
            Options run_opts = opts;
            run_opts.threads_per_server = threads_per_server;

            double accumulated_seconds = 0.0;
            std::uint64_t total_bytes = 0;

            for (int trial = 0; trial < run_opts.trials; ++trial) {
                rdma_read_bench::MultiServerReadBench bench(run_opts);
                auto results = bench.measure_throughput();

                double trial_seconds = 0.0;
                for (const auto& result : results) {
                    trial_seconds = std::max(
                        trial_seconds, result.elapsed_sec);
                }
                accumulated_seconds += trial_seconds;

                for (const auto& result : results) {
                    total_bytes += result.bytes;
                }
            }

            print_throughput_row(
                run_opts.threads_per_server *
                    static_cast<int>(run_opts.server_hosts.size()),
                total_bytes,
                accumulated_seconds);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_read_matrix_throughput failed: "
                  << e.what() << "\n";
        return 1;
    }
}
