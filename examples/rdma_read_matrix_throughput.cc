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
    std::uint64_t prewarm_count = 0;
    std::string trial_reducer = "mean";
    int only_total_threads = 0;
    bool prewarm_count_set = false;
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
            << " only_total_threads=" << parsed.only_total_threads
            << " payload_bytes=" << opts.payload_bytes
            << " memory=" << opts.memory
            << " count=" << parsed.count
            << " prewarm_count=" << parsed.prewarm_count
            << " warmup_ops=" << opts.warmup_ops
            << " trials=" << opts.trials
            << " trial_reducer=" << parsed.trial_reducer
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

            if (parsed.prewarm_count > 0) {
                rdma_read_bench::MultiServerReadBench warmup_bench(run_opts);
                (void)warmup_bench.measure_read_count_total(parsed.prewarm_count);
            }

            std::vector<double> trial_gbps_values;
            trial_gbps_values.reserve(static_cast<std::size_t>(run_opts.trials));

            for (int trial = 0; trial < run_opts.trials; ++trial) {
                rdma_read_bench::MultiServerReadBench bench(run_opts);
                auto result = bench.measure_read_count_total(parsed.count);
                trial_gbps_values.push_back(
                    rdma_read_bench::bytes_to_gbps(
                        result.bytes,
                        result.elapsed_sec));
            }

            const int total_threads =
                run_opts.threads_per_server *
                static_cast<int>(run_opts.server_hosts.size());
            std::cout << total_threads << ","
                      << reduce_trials(
                             std::move(trial_gbps_values),
                             parsed.trial_reducer)
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
