#include "rdma_read_bench_common.hpp"

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
        << "  --payload-bytes BYTES       default: 128; accepts k/m/g suffix\n"
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
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n";
}

Options parse_args(int argc, char* argv[]) {
    Options opts;
    opts.payload_bytes = 128;
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
        } else {
            rdma_read_bench::usage_error("unknown argument: " + arg);
        }
    }

    rdma_read_bench::finalize_options(
        opts, 128, 4 * rdma_read_bench::kKiB);
    return opts;
}

void print_latency_row(const Options& opts,
                       const std::string& section,
                       int server_index,
                       const std::string& endpoint,
                       int threads,
                       const std::vector<double>& values) {
    std::cout << section << ","
              << opts.client_id << ","
              << opts.clients << ","
              << opts.server_hosts.size() << ","
              << server_index << ","
              << endpoint << ","
              << threads << ","
              << opts.payload_bytes << ","
              << rdma_read_bench::mean(values) << ","
              << rdma_read_bench::percentile(values, 50) << ","
              << rdma_read_bench::percentile(values, 95) << ","
              << rdma_read_bench::percentile(values, 99)
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
        Options opts = parse_args(argc, argv);
        rdma_read_bench::set_spdlog_level_from_name(opts.rdma_log_level);

        std::cout << std::fixed << std::setprecision(4);
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
            << "\n";
        print_worker_map(opts);
        std::cout.flush();
        if (opts.start_delay_sec > 0.0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(opts.start_delay_sec));
        }
        std::cout
            << "section,client_id,clients,servers,server_idx,endpoint,"
            << "threads,payload_bytes,avg_us,p50_us,p95_us,p99_us\n";

        for (int threads_per_server = opts.min_threads_per_server;
             threads_per_server <= opts.threads_per_server;
             ++threads_per_server) {
            Options run_opts = opts;
            run_opts.threads_per_server = threads_per_server;

            std::vector<rdma_read_bench::LatencySample> all_samples;
            for (int trial = 0; trial < run_opts.trials; ++trial) {
                rdma_read_bench::MultiServerReadBench bench(run_opts);
                auto samples = bench.measure_latency();
                all_samples.insert(all_samples.end(),
                                   samples.begin(),
                                   samples.end());
            }

            auto overall =
                rdma_read_bench::latency_values(all_samples, -1);
            print_latency_row(
                run_opts,
                "overall",
                -1,
                "all",
                run_opts.threads_per_server *
                    static_cast<int>(run_opts.server_hosts.size()),
                overall);

            for (int server_index = 0;
                 server_index < static_cast<int>(run_opts.server_hosts.size());
                 ++server_index) {
                auto values =
                    rdma_read_bench::latency_values(all_samples, server_index);
                print_latency_row(
                    run_opts,
                    "server",
                    server_index,
                    rdma_read_bench::endpoint_label(run_opts, server_index),
                    run_opts.threads_per_server,
                    values);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_read_matrix_latency failed: " << e.what() << "\n";
        return 1;
    }
}
