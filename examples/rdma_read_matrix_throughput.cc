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

struct ReadBwOptions {
    Options common;
    std::uint64_t count = 3000;
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
        << "  --payload-bytes BYTES       default: 4m; accepts k/m/g suffix\n"
        << "  --prompt-bytes BYTES        alias for --payload-bytes\n"
        << "  --memory host               read_bw-compatible mode is host-memory only\n"
        << "  --offset BYTES              default: 0\n"
        << "  --stride BYTES              default: payload size\n\n"
        << "Benchmark options:\n"
        << "  --count N                   4MB READs per worker, default: 3000\n"
        << "  --warmup-ops N              warmup READs per worker, default: 8\n"
        << "  --trials N                  default: 1\n"
        << "  --start-delay-sec SEC       sleep before connecting, default: 0\n"
        << "  --stagger-us N              default: 0\n"
        << "  --cpu-list LIST             pin local workers, e.g. 0,2,4,6\n"
        << "  --completion-mode MODE      raw-poll|callback, default: raw-poll\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n";
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
    return parsed;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        ReadBwOptions parsed = parse_args(argc, argv);
        Options opts = parsed.common;
        rdma_read_bench::set_spdlog_level_from_name(opts.rdma_log_level);

        std::cout << std::fixed << std::setprecision(4);
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
            << " payload_bytes=" << opts.payload_bytes
            << " memory=" << opts.memory
            << " count=" << parsed.count
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
            run_opts.chunk_bytes = run_opts.payload_bytes;
            run_opts.max_inflight = 1;
            run_opts.pipeline_depth = 1;
            run_opts.reuse_local_buffer = true;

            double accumulated_gbps = 0.0;

            for (int trial = 0; trial < run_opts.trials; ++trial) {
                rdma_read_bench::MultiServerReadBench bench(run_opts);
                auto results = bench.measure_read_count(parsed.count);

                double trial_gbps = 0.0;
                for (const auto& result : results) {
                    trial_gbps += rdma_read_bench::bytes_to_gbps(
                        result.bytes,
                        result.elapsed_sec);
                }
                accumulated_gbps += trial_gbps;
            }

            const int total_threads =
                run_opts.threads_per_server *
                static_cast<int>(run_opts.server_hosts.size());
            std::cout << total_threads << ","
                      << accumulated_gbps /
                             static_cast<double>(run_opts.trials)
                      << "\n";
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_read_matrix_throughput failed: "
                  << e.what() << "\n";
        return 1;
    }
}
