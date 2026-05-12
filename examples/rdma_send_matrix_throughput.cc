#include "qp_transmission.h"
#include "qp_connector.h"
#include "rdma_read_bench_common.hpp"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/executor_work_guard.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/use_future.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rdmapp/executor.h>
#include <rdmapp/mr.h>
#include <rdmapp/rdmapp.h>
#include <spdlog/spdlog.h>

namespace {

constexpr std::uint32_t kConfigMagic = 0x57425343u; // CSBW
constexpr std::uint32_t kReadyMagic = 0x57425352u;  // RSBW
constexpr std::uint32_t kGoMagic = 0x57425347u;     // GSBW
constexpr std::uint16_t kVersion = 1;

#pragma pack(push, 1)
struct ConfigWire {
    std::uint32_t magic = kConfigMagic;
    std::uint16_t version = kVersion;
    std::uint16_t reserved = 0;
    std::uint32_t payload_bytes = 0;
    std::uint32_t count = 0;
};

struct ReadyWire {
    std::uint32_t magic = kReadyMagic;
    std::uint16_t version = kVersion;
    std::uint16_t status = 0;
};

struct GoWire {
    std::uint32_t magic = kGoMagic;
    std::uint16_t version = kVersion;
    std::uint16_t reserved = 0;
};
#pragma pack(pop)

using rdma_read_bench::Options;

struct SendOptions {
    Options common;
    std::string role = "client";
    std::uint64_t total_sends = 60000;
    int recv_window = 16;
    int max_connections = 4096;
};

struct AcceptedClient {
    std::shared_ptr<rdmapp::qp> qp;
    std::shared_ptr<rdmapp::cq> recv_cq;
    std::shared_ptr<rdmapp::cq> send_cq;
};

struct WorkerResult {
    std::uint64_t sends = 0;
    std::uint64_t bytes = 0;
    double seconds = 0.0;
};

std::atomic<int> g_live_connections = 0;

class PerClientAcceptor {
public:
    PerClientAcceptor(std::shared_ptr<asio::io_context> io_ctx,
                      std::uint16_t port,
                      std::shared_ptr<rdmapp::pd> pd,
                      std::shared_ptr<rdmapp::device> device)
        : io_ctx_(std::move(io_ctx)),
          pd_(std::move(pd)),
          device_(std::move(device)),
          acceptor_(*io_ctx_, {asio::ip::tcp::v4(), port}) {}

    asio::awaitable<std::shared_ptr<AcceptedClient>> accept() {
        asio::ip::tcp::socket socket =
            co_await acceptor_.async_accept(asio::use_awaitable);
        auto remote_qp = co_await rdmapp::recv_qp(socket);

        auto client = std::make_shared<AcceptedClient>();
        client->recv_cq = std::make_shared<rdmapp::cq>(device_);
        client->send_cq = std::make_shared<rdmapp::cq>(device_);
        client->qp = std::make_shared<rdmapp::qp>(
            remote_qp.header.lid,
            remote_qp.header.qp_num,
            remote_qp.header.sq_psn,
            remote_qp.header.gid,
            pd_,
            client->recv_cq,
            client->send_cq);
        client->qp->user_data() = std::move(remote_qp.user_data);
        co_await rdmapp::send_qp(*client->qp, socket);
        co_return client;
    }

private:
    std::shared_ptr<asio::io_context> io_ctx_;
    std::shared_ptr<rdmapp::pd> pd_;
    std::shared_ptr<rdmapp::device> device_;
    asio::ip::tcp::acceptor acceptor_;
};

template <typename T>
void raw_recv_pod_wait(std::shared_ptr<rdmapp::qp> qp,
                       rdmapp::local_mr& mr,
                       T& value,
                       const char* what) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(&value);
    sge.length = sizeof(T);
    sge.lkey = mr.lkey();

    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr* bad_recv_wr = nullptr;
    recv_wr.wr_id = 1;
    recv_wr.num_sge = 1;
    recv_wr.sg_list = &sge;
    qp->post_recv(recv_wr, bad_recv_wr);

    struct ibv_wc wc = {};
    while (!qp->poll_recv_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS || wc.byte_len != sizeof(T)) {
        throw std::runtime_error(
            std::string("bad recv for ") + what + ": " +
            ibv_wc_status_str(wc.status));
    }
}

template <typename T>
void raw_send_pod_wait(std::shared_ptr<rdmapp::qp> qp,
                       rdmapp::local_mr& mr,
                       T& value) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(&value);
    sge.length = sizeof(T);
    sge.lkey = mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.wr_id = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    qp->post_send(send_wr, bad_send_wr);

    struct ibv_wc wc = {};
    while (!qp->poll_send_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("bad send: ") + ibv_wc_status_str(wc.status));
    }
}

void raw_send_payload_wait(std::shared_ptr<rdmapp::qp> qp,
                           const rdmapp::local_mr& mr,
                           std::uint32_t bytes) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(mr.addr());
    sge.length = bytes;
    sge.lkey = mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.wr_id = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    qp->post_send(send_wr, bad_send_wr);

    struct ibv_wc wc = {};
    while (!qp->poll_send_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("payload send failed: ") +
            ibv_wc_status_str(wc.status));
    }
}

void post_payload_recv(std::shared_ptr<rdmapp::qp> qp,
                       const rdmapp::local_mr& mr,
                       std::uintptr_t addr,
                       std::uint32_t bytes,
                       std::uint64_t wr_id) {
    struct ibv_sge sge = {};
    sge.addr = static_cast<std::uint64_t>(addr);
    sge.length = bytes;
    sge.lkey = mr.lkey();

    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr* bad_recv_wr = nullptr;
    recv_wr.wr_id = wr_id;
    recv_wr.num_sge = 1;
    recv_wr.sg_list = &sge;
    qp->post_recv(recv_wr, bad_recv_wr);
}

void server_worker(std::shared_ptr<AcceptedClient> client) {
    ++g_live_connections;
    try {
        auto qp = client->qp;
        ConfigWire config{};
        GoWire go{};
        ReadyWire ready{};

        auto config_mr = std::make_shared<rdmapp::local_mr>(
            qp->pd_ptr()->reg_mr(&config, sizeof(config)));
        auto go_mr = std::make_shared<rdmapp::local_mr>(
            qp->pd_ptr()->reg_mr(&go, sizeof(go)));
        auto ready_mr = std::make_shared<rdmapp::local_mr>(
            qp->pd_ptr()->reg_mr(&ready, sizeof(ready)));

        raw_recv_pod_wait(qp, *config_mr, config, "config");
        if (config.magic != kConfigMagic || config.version != kVersion ||
            config.payload_bytes == 0 || config.count == 0) {
            throw std::runtime_error("bad send-bw config");
        }

        std::vector<std::uint8_t> payload(config.payload_bytes, 0);
        auto payload_mr = std::make_shared<rdmapp::local_mr>(
            qp->pd_ptr()->reg_mr(payload.data(), payload.size()));

        ready.status = 0;
        raw_send_pod_wait(qp, *ready_mr, ready);
        raw_recv_pod_wait(qp, *go_mr, go, "go");
        if (go.magic != kGoMagic || go.version != kVersion) {
            throw std::runtime_error("bad go message");
        }

        for (std::uint32_t i = 0; i < config.count; ++i) {
            raw_send_payload_wait(qp, *payload_mr, config.payload_bytes);
        }
    } catch (const std::exception& e) {
        spdlog::warn("send-bw server worker failed: {}", e.what());
    }
    --g_live_connections;
}

asio::awaitable<void> server_loop(std::shared_ptr<PerClientAcceptor> acceptor,
                                  int max_connections) {
    while (true) {
        auto client = co_await acceptor->accept();
        if (!client || !client->qp) {
            continue;
        }
        if (g_live_connections.load() >= max_connections) {
            spdlog::warn("rejecting send-bw connection: max_connections reached");
            continue;
        }
        std::thread(server_worker, client).detach();
    }
}

std::shared_ptr<rdmapp::qp> connect_qp(const Options& opts,
                                       int worker_id,
                                       std::shared_ptr<rdmapp::device>& device,
                                       std::shared_ptr<rdmapp::pd>& pd,
                                       std::shared_ptr<rdmapp::cq>& recv_cq,
                                       std::shared_ptr<rdmapp::cq>& send_cq,
                                       std::shared_ptr<asio::io_context>& io_ctx) {
    const std::size_t config_id =
        static_cast<std::size_t>(worker_id / opts.threads_per_server) %
        rdma_read_bench::endpoint_config_count(opts);
    int ib_device = rdma_read_bench::pick_round_robin(opts.ib_devices, config_id);
    int ib_port = rdma_read_bench::pick_round_robin(opts.ib_ports, config_id);
    const auto& host =
        rdma_read_bench::pick_round_robin(opts.server_hosts, config_id);
    std::uint16_t port =
        rdma_read_bench::pick_round_robin(opts.server_ports, config_id);

    device = std::make_shared<rdmapp::device>(ib_device, ib_port);
    pd = std::make_shared<rdmapp::pd>(device);
    recv_cq = std::make_shared<rdmapp::cq>(device);
    send_cq = std::make_shared<rdmapp::cq>(device);
    io_ctx = std::make_shared<asio::io_context>(1);
    auto connector = std::make_shared<rdmapp::qp_connector>(
        host, port, pd, recv_cq, send_cq);
    auto future = asio::co_spawn(
        *io_ctx,
        [connector]() -> asio::awaitable<std::shared_ptr<rdmapp::qp>> {
            co_return co_await connector->connect();
        }(),
        asio::use_future);
    std::jthread io_thread([&] {
        auto guard = asio::make_work_guard(*io_ctx);
        io_ctx->run();
    });
    auto qp = future.get();
    io_ctx->stop();
    return qp;
}

WorkerResult client_worker(const SendOptions& opts,
                           int worker_id,
                           std::uint32_t count,
                           rdma_read_bench::StartGate& start_gate) {
    if (!opts.common.cpu_list.empty()) {
        rdma_read_bench::pin_current_thread_to_cpu(
            opts.common.cpu_list[static_cast<std::size_t>(worker_id)]);
    }

    std::shared_ptr<rdmapp::device> device;
    std::shared_ptr<rdmapp::pd> pd;
    std::shared_ptr<rdmapp::cq> recv_cq;
    std::shared_ptr<rdmapp::cq> send_cq;
    std::shared_ptr<asio::io_context> io_ctx;
    auto qp = connect_qp(opts.common, worker_id, device, pd, recv_cq, send_cq, io_ctx);

    ConfigWire config;
    config.payload_bytes =
        static_cast<std::uint32_t>(opts.common.payload_bytes);
    config.count = count;
    ReadyWire ready{};
    GoWire go{};

    auto config_mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(&config, sizeof(config)));
    auto ready_mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(&ready, sizeof(ready)));
    auto go_mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(&go, sizeof(go)));

    raw_send_pod_wait(qp, *config_mr, config);
    raw_recv_pod_wait(qp, *ready_mr, ready, "ready");
    if (ready.magic != kReadyMagic || ready.version != kVersion ||
        ready.status != 0) {
        throw std::runtime_error("bad ready message");
    }

    const std::uint64_t recv_bytes =
        opts.common.payload_bytes * static_cast<std::uint64_t>(opts.recv_window);
    auto recv_buffer = rdma_read_bench::HostBuffer(recv_bytes);
    auto recv_mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(
            reinterpret_cast<void*>(recv_buffer.addr()),
            static_cast<std::size_t>(recv_buffer.length())));

    std::uint32_t posted = 0;
    std::uint32_t completed = 0;
    const std::uint32_t initial = std::min<std::uint32_t>(
        count, static_cast<std::uint32_t>(opts.recv_window));
    for (; posted < initial; ++posted) {
        const std::uint64_t slot = posted % opts.recv_window;
        post_payload_recv(
            qp,
            *recv_mr,
            recv_buffer.addr() + slot * opts.common.payload_bytes,
            static_cast<std::uint32_t>(opts.common.payload_bytes),
            slot);
    }

    start_gate.arrive_and_wait();
    raw_send_pod_wait(qp, *go_mr, go);
    const auto t0 = std::chrono::steady_clock::now();

    while (completed < count) {
        struct ibv_wc wc = {};
        while (!qp->poll_recv_cq(wc)) {
        }
        if (wc.status != IBV_WC_SUCCESS ||
            wc.byte_len != opts.common.payload_bytes) {
            throw std::runtime_error(
                std::string("payload recv failed: ") +
                ibv_wc_status_str(wc.status));
        }
        ++completed;
        if (posted < count) {
            const std::uint64_t slot = wc.wr_id;
            post_payload_recv(
                qp,
                *recv_mr,
                recv_buffer.addr() + slot * opts.common.payload_bytes,
                static_cast<std::uint32_t>(opts.common.payload_bytes),
                slot);
            ++posted;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    return WorkerResult{
        count,
        static_cast<std::uint64_t>(count) * opts.common.payload_bytes,
        std::chrono::duration<double>(t1 - t0).count()};
}

double run_client_thread_count(const SendOptions& opts, int threads_per_server) {
    SendOptions run_opts = opts;
    run_opts.common.threads_per_server = threads_per_server;
    const int total_threads =
        threads_per_server *
        static_cast<int>(run_opts.common.server_hosts.size());
    const std::uint32_t count_per_thread =
        static_cast<std::uint32_t>(
            std::max<std::uint64_t>(
                1,
                (run_opts.total_sends +
                 static_cast<std::uint64_t>(threads_per_server) - 1) /
                    static_cast<std::uint64_t>(threads_per_server)));

    rdma_read_bench::StartGate start_gate(total_threads);
    std::vector<WorkerResult> results(static_cast<std::size_t>(total_threads));
    std::vector<std::thread> threads;
    std::mutex exception_mutex;
    std::exception_ptr first_exception;
    threads.reserve(static_cast<std::size_t>(total_threads));

    for (int worker_id = 0; worker_id < total_threads; ++worker_id) {
        threads.emplace_back([&, worker_id] {
            try {
                results[static_cast<std::size_t>(worker_id)] =
                    client_worker(run_opts, worker_id, count_per_thread, start_gate);
            } catch (...) {
                std::lock_guard<std::mutex> lk(exception_mutex);
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

    std::uint64_t total_bytes = 0;
    double seconds = 0.0;
    for (const auto& result : results) {
        total_bytes += result.bytes;
        seconds = std::max(seconds, result.seconds);
    }
    return rdma_read_bench::bytes_to_gbps(total_bytes, seconds);
}

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " --role server --port PORT [server options]\n"
        << "  " << prog << " --role client --server-endpoints HOST:PORT[,HOST:PORT...] [client options]\n\n"
        << "Common/client options:\n"
        << "  --server-endpoints LIST     HOST:PORT[,HOST:PORT...]\n"
        << "  --ib-device DEV             default: 0\n"
        << "  --ib-port PORT              default: 1\n"
        << "  --ib-devices LIST           local device list for multi-server\n"
        << "  --ib-ports LIST             local port list for multi-server\n"
        << "  --min-threads N             per-server min, default: 1\n"
        << "  --max-threads N             per-server max, default: 16\n"
        << "  --threads-per-server N      alias for --max-threads\n"
        << "  --payload-bytes BYTES       default: 4m\n"
        << "  --total-sends N             default: 60000 sends per server per thread-count\n"
        << "  --recv-window N             default: 16 posted receives per worker\n"
        << "  --memory host               this send/recv benchmark is host-memory only\n"
        << "  --cpu-list LIST             pin client workers\n"
        << "  --start-delay-sec SEC       sleep before connecting, default: 0\n\n"
        << "Server options:\n"
        << "  --port PORT                 default: 2333\n"
        << "  --max-connections N         default: 4096\n";
}

SendOptions parse_args(int argc, char* argv[]) {
    SendOptions opts;
    opts.common.memory = "host";
    opts.common.payload_bytes = 4 * rdma_read_bench::kMiB;
    opts.common.threads_per_server = 16;
    opts.common.min_threads_per_server = 1;
    opts.common.trials = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (rdma_read_bench::is_blank_arg(arg)) {
            rdma_read_bench::usage_error("blank argument detected");
        }
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        bool handled = false;
        rdma_read_bench::parse_common_arg(opts.common, i, argc, argv, handled);
        if (handled) {
            continue;
        }

        if (arg == "--role") {
            opts.role =
                rdma_read_bench::lower_copy(
                    rdma_read_bench::take_value(i, argc, argv, arg));
        } else if (arg == "--total-sends") {
            opts.total_sends =
                static_cast<std::uint64_t>(
                    rdma_read_bench::parse_ll(
                        arg,
                        rdma_read_bench::take_value(i, argc, argv, arg),
                        1,
                        std::numeric_limits<long long>::max()));
        } else if (arg == "--recv-window") {
            opts.recv_window =
                static_cast<int>(
                    rdma_read_bench::parse_ll(
                        arg,
                        rdma_read_bench::take_value(i, argc, argv, arg),
                        1,
                        128));
        } else if (arg == "--max-connections") {
            opts.max_connections =
                static_cast<int>(
                    rdma_read_bench::parse_ll(
                        arg,
                        rdma_read_bench::take_value(i, argc, argv, arg),
                        1,
                        std::numeric_limits<int>::max()));
        } else {
            rdma_read_bench::usage_error("unknown argument: " + arg);
        }
    }

    if (opts.role != "server" && opts.role != "client") {
        rdma_read_bench::usage_error("--role must be server or client");
    }
    rdma_read_bench::finalize_options(
        opts.common, 4 * rdma_read_bench::kMiB, 0);
    if (opts.common.memory != "host") {
        rdma_read_bench::usage_error(
            "send/recv throughput benchmark is host-memory only; omit "
            "--memory or use --memory host");
    }
    if (opts.common.payload_bytes > std::numeric_limits<std::uint32_t>::max()) {
        rdma_read_bench::usage_error("--payload-bytes exceeds uint32");
    }
    return opts;
}

int run_server(const SendOptions& opts) {
    rdma_read_bench::set_spdlog_level_from_name(opts.common.rdma_log_level);
    auto device = std::make_shared<rdmapp::device>(
        opts.common.ib_device, opts.common.ib_port);
    auto pd = std::make_shared<rdmapp::pd>(device);
    auto io_ctx = std::make_shared<asio::io_context>(1);
    auto acceptor = std::make_shared<PerClientAcceptor>(
        io_ctx, opts.common.port, pd, device);

    spdlog::info(
        "send-bw server listening on port {}, ib_device={}, ib_port={}",
        opts.common.port,
        opts.common.ib_device,
        opts.common.ib_port);
    auto guard = asio::make_work_guard(*io_ctx);
    asio::co_spawn(
        *io_ctx,
        server_loop(acceptor, opts.max_connections),
        asio::detached);
    io_ctx->run();
    return 0;
}

int run_client(const SendOptions& opts) {
    rdma_read_bench::set_spdlog_level_from_name(opts.common.rdma_log_level);
    if (opts.common.start_delay_sec > 0.0) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(opts.common.start_delay_sec));
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout
        << "# data_path=send_recv_throughput_raw_poll"
        << " clients=" << opts.common.clients
        << " client_id=" << opts.common.client_id
        << " servers=" << opts.common.server_hosts.size()
        << " min_threads_per_server=" << opts.common.min_threads_per_server
        << " max_threads_per_server=" << opts.common.threads_per_server
        << " payload_bytes=" << opts.common.payload_bytes
        << " total_sends_per_server=" << opts.total_sends
        << " recv_window=" << opts.recv_window
        << "\n";
    std::cout << "threads,gbps\n";

    for (int threads_per_server = opts.common.min_threads_per_server;
         threads_per_server <= opts.common.threads_per_server;
         ++threads_per_server) {
        const int total_threads =
            threads_per_server *
            static_cast<int>(opts.common.server_hosts.size());
        double gbps = run_client_thread_count(opts, threads_per_server);
        std::cout << total_threads << "," << gbps << "\n";
        std::cout.flush();
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        SendOptions opts = parse_args(argc, argv);
        return opts.role == "server" ? run_server(opts) : run_client(opts);
    } catch (const std::exception& e) {
        std::cerr << "rdma_send_matrix_throughput failed: "
                  << e.what() << "\n";
        return 1;
    }
}
