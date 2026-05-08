#include "qp_acceptor.h"
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/this_coro.hpp>
#include <asio/steady_timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include <rdmapp/rdmapp.h>
#include <rdmapp/executor.h>
#include <rdmapp/mr.h>

// ==============================
// 全局资源池
// ==============================

static std::vector<std::shared_ptr<rdmapp::qp>> g_qps;

// 只分配/注册一次：全局 buffer + MR 池，所有 QP 共享这些 MR
static std::vector<std::shared_ptr<std::vector<std::uint8_t>>> g_buffers;
static std::vector<std::shared_ptr<rdmapp::local_mr>> g_mrs;

constexpr int NUM_GLOBAL_MR = 64;
constexpr std::size_t KV_BUFFER_SIZE = 2ull << 30;

// Maximum live QPs. Multi-QP bandwidth tests create one connection per QP.
static int g_max_connections = 64;

// 简单背压：同一时间只允许一个 handle_qp 在跑
static bool g_handling = false;

// ==============================
// 释放最旧连接（仅断开 QP，不再动 MR 池）
// ==============================
void release_oldest_connection() {
    if (g_qps.empty()) return;
    spdlog::warn("Max connections reached. Releasing oldest connection...");
    g_qps.erase(g_qps.begin()); // shared_ptr 减到 0 -> QP 析构断连
    spdlog::warn("Oldest connection released successfully.");
}

// ==============================
// 启动时预分配 + 预注册 MR（一次性）
// ==============================
static void init_global_mr_pool(std::shared_ptr<rdmapp::pd> pd) {
    spdlog::info("Pre-allocating {} buffers and registering MRs ({} bytes each)...",
                 NUM_GLOBAL_MR, KV_BUFFER_SIZE);

    g_buffers.reserve(NUM_GLOBAL_MR);
    g_mrs.reserve(NUM_GLOBAL_MR);

    for (int i = 0; i < NUM_GLOBAL_MR; i++) {
        auto kv_buf = std::make_shared<std::vector<uint8_t>>(KV_BUFFER_SIZE);

        // 注意：这里用 pd 注册一次即可；后续连接只复用 MR 信息
        auto local_mr = std::make_shared<rdmapp::local_mr>(
            pd->reg_mr(kv_buf->data(), kv_buf->size()));

        spdlog::info("Pool MR[{}]: addr={:p} length={} rkey={}",
                     i,
                     static_cast<void*>(local_mr->addr()),
                     local_mr->length(),
                     local_mr->rkey());

        g_buffers.push_back(kv_buf);
        g_mrs.push_back(local_mr);
    }

    spdlog::info("Global MR pool ready.");
}

// ==============================
// 处理单个 QP：不再分配/注册 MR，只发送元数据
// ==============================
asio::awaitable<void> handle_qp(std::shared_ptr<rdmapp::qp> qp) {
    spdlog::info("Handling qp (KV server)");

    // 发送 MR 数量
    {
        uint32_t n = static_cast<uint32_t>(g_mrs.size());
        co_await qp->send(std::as_bytes(std::span(&n, 1)));
    }

    // 发送每个 MR 的序列化信息（addr/len/rkey 等）
    for (std::size_t i = 0; i < g_mrs.size(); i++) {
        auto serialized = g_mrs[i]->serialize();
        co_await qp->send(std::as_bytes(std::span(serialized)));
    }

    g_qps.push_back(qp);
    co_return;
}

// ==============================
// Server 主循环：加一个非常轻量的背压
// ==============================
asio::awaitable<void> server(std::shared_ptr<rdmapp::qp_acceptor> acceptor) {
    auto ex = co_await asio::this_coro::executor;

    while (true) {
        auto qp = co_await acceptor->accept();

        if (g_qps.size() >= static_cast<std::size_t>(g_max_connections)) {
            release_oldest_connection();
        }

        // 简单背压：如果上一个 handle_qp 还没结束，就稍微等一下
        // （避免瞬间堆很多协程，CPU 抖动很大）
        while (g_handling) {
            asio::steady_timer t(ex);
            t.expires_after(std::chrono::milliseconds(1));
            co_await t.async_wait(asio::use_awaitable);
        }

        g_handling = true;

        asio::co_spawn(
            ex,
            [qp]() -> asio::awaitable<void> {
                try {
                    co_await handle_qp(qp);
                } catch (const std::exception &e) {
                    spdlog::error("handle_qp failed: {}", e.what());
                } catch (...) {
                    spdlog::error("handle_qp failed with unknown exception");
                }
                g_handling = false;
                co_return;
            }(),
            asio::detached
        );
    }
}

// ==============================
// main
// ==============================
struct ServerOptions {
    uint16_t port = 2333;
    int ib_device = 0;
    int ib_port = 1;
    int max_connections = 64;
};

static void print_usage(const char *prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [port]\n"
        << "  " << prog << " --port PORT [--ib-device DEV] [--ib-port PORT] "
        << "[--max-connections N]\n\n"
        << "Defaults:\n"
        << "  --port 2333 --ib-device 0 --ib-port 1 --max-connections 64\n";
}

static int parse_int_arg(const std::string &name,
                         const std::string &value,
                         int min_value,
                         int max_value) {
    try {
        size_t consumed = 0;
        int parsed = std::stoi(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        if (parsed < min_value || parsed > max_value) {
            throw std::out_of_range("outside allowed range");
        }
        return parsed;
    } catch (const std::exception &e) {
        throw std::runtime_error(
            "invalid " + name + "=" + value + ": " + e.what());
    }
}

static ServerOptions parse_args(int argc, char *argv[]) {
    ServerOptions opts;
    bool positional_port_seen = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        auto require_value = [&](const std::string &flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(flag + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--port") {
            opts.port = static_cast<uint16_t>(
                parse_int_arg(arg, require_value(arg), 1, 65535));
        } else if (arg == "--ib-device") {
            opts.ib_device = parse_int_arg(
                arg, require_value(arg), 0, std::numeric_limits<int>::max());
        } else if (arg == "--ib-port") {
            opts.ib_port = parse_int_arg(
                arg, require_value(arg), 1, std::numeric_limits<int>::max());
        } else if (arg == "--max-connections") {
            opts.max_connections = parse_int_arg(
                arg, require_value(arg), 1, std::numeric_limits<int>::max());
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown argument: " + arg);
        } else {
            if (positional_port_seen) {
                throw std::runtime_error("multiple positional ports provided");
            }
            opts.port = static_cast<uint16_t>(
                parse_int_arg("port", arg, 1, 65535));
            positional_port_seen = true;
        }
    }

    return opts;
}

int main(int argc, char *argv[]) {
    ServerOptions opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return -1;
    }

    g_max_connections = opts.max_connections;
    spdlog::set_level(spdlog::level::debug);
    spdlog::info(
        "Starting RDMA server on port {}, ib_device={}, ib_port={}, max_connections={}",
        opts.port,
        opts.ib_device,
        opts.ib_port,
        opts.max_connections);

    auto device = std::make_shared<rdmapp::device>(opts.ib_device, opts.ib_port);
    auto pd     = std::make_shared<rdmapp::pd>(device);
    auto cq     = std::make_shared<rdmapp::cq>(device);

    // ⚠️ 如果你现在 io_context(4) CPU 会更高（更多线程参与调度/唤醒）
    // 先降到 1，通常能明显降 idle CPU 和抖动
    auto io_ctx = std::make_shared<asio::io_context>(1);

    auto executor  = std::make_shared<rdmapp::executor>(io_ctx);

    // CQ poller：如果它是 busy-poll，idle CPU 仍可能偏高
    // 但至少我们已经把 reg_mr 的巨大 CPU 峰值拿掉了
    auto cq_poller = std::make_shared<rdmapp::cq_poller>(cq, executor);

    // ✅ 启动时一次性注册 MR（核心改动）
    init_global_mr_pool(pd);

    auto acceptor = std::make_shared<rdmapp::qp_acceptor>(
        io_ctx, opts.port, pd, cq);

    auto work_guard = asio::make_work_guard(*io_ctx);
    asio::co_spawn(*io_ctx, server(acceptor), asio::detached);
    io_ctx->run();

    return 0;
}
