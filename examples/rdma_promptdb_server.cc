#include "qp_transmission.h"
#include "promptdb_lsh.hpp"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/this_coro.hpp>
#include <asio/use_awaitable.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
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

using promptdb_rdma::AckWire;
using promptdb_rdma::ClientHelloWire;
using promptdb_rdma::FindResult;
using promptdb_rdma::PromptDbConfig;
using promptdb_rdma::PromptDbState;
using promptdb_rdma::RequestOp;
using promptdb_rdma::RequestWire;
using promptdb_rdma::ResultHeaderWire;
using promptdb_rdma::ServerHelloWire;
using promptdb_rdma::Status;

struct ServerOptions {
    std::uint16_t port = 2333;
    int ib_device = 0;
    int ib_port = 1;
    int io_threads = 4;
    std::size_t cq_batch_size = 32;
    int max_connections = 64;
    int prompts = 4096;
    int token_len = 32;
    int max_tokens = 0;
    int vocab_size = 32000;
    int num_hash_planes = 16;
    int bucket_size = 8;
    int seed = 42;
    std::string rdma_log_level = "warn";
};

std::atomic<int> g_live_connections = 0;

struct AcceptedClient {
    std::shared_ptr<rdmapp::qp> qp;
    std::shared_ptr<rdmapp::cq> recv_cq;
    std::shared_ptr<rdmapp::cq> send_cq;
};

class PromptDbAcceptor {
public:
    PromptDbAcceptor(std::shared_ptr<asio::io_context> io_ctx,
                     std::uint16_t port,
                     std::shared_ptr<rdmapp::pd> pd,
                     std::shared_ptr<rdmapp::device> device)
        : io_ctx_(std::move(io_ctx)),
          pd_(std::move(pd)),
          device_(std::move(device)),
          acceptor_(*io_ctx_, {asio::ip::tcp::v4(), port}) {}

    asio::awaitable<std::shared_ptr<AcceptedClient>> accept() {
        try {
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
        } catch (const std::exception& e) {
            spdlog::error("accept promptdb qp failed: {}", e.what());
            co_return nullptr;
        }
    }

private:
    std::shared_ptr<asio::io_context> io_ctx_;
    std::shared_ptr<rdmapp::pd> pd_;
    std::shared_ptr<rdmapp::device> device_;
    asio::ip::tcp::acceptor acceptor_;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::runtime_error("argument error: " + message);
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

std::string take_value(int& index,
                       int argc,
                       char* argv[],
                       const std::string& flag) {
    if (index + 1 >= argc) {
        usage_error(flag + " requires a value");
    }
    return argv[++index];
}

void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [options]\n\n"
        << "RDMA options:\n"
        << "  --port PORT                 default: 2333\n"
        << "  --ib-device DEV             default: 0\n"
        << "  --ib-port PORT              default: 1\n"
        << "  --io-threads N              default: 4\n"
        << "  --cq-batch-size N           default: 32\n"
        << "  --max-connections N         default: 64\n"
        << "  --rdma-log-level LEVEL      trace/debug/info/warn/error/critical/off\n\n"
        << "Prompt DB options:\n"
        << "  --prompts N                 default: 4096\n"
        << "  --token-len N               default: 32\n"
        << "  --max-tokens N              default: token-len\n"
        << "  --vocab-size N              default: 32000\n"
        << "  --num-hash-planes N         default: 16 (kept for TCP config compatibility)\n"
        << "  --bucket-size N             default: 8\n"
        << "  --seed N                    default: 42\n";
}

ServerOptions parse_args(int argc, char* argv[]) {
    ServerOptions opts;
    bool positional_port_seen = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--port") {
            opts.port = static_cast<std::uint16_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1, 65535));
        } else if (arg == "--ib-device") {
            opts.ib_device = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--ib-port") {
            opts.ib_port = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--io-threads") {
            opts.io_threads = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--cq-batch-size") {
            opts.cq_batch_size = static_cast<std::size_t>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--max-connections") {
            opts.max_connections = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--prompts") {
            opts.prompts = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--token-len") {
            opts.token_len = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--max-tokens") {
            opts.max_tokens = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--vocab-size") {
            opts.vocab_size = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--num-hash-planes") {
            opts.num_hash_planes = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--bucket-size") {
            opts.bucket_size = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 1,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--seed") {
            opts.seed = static_cast<int>(
                parse_ll(arg, take_value(i, argc, argv, arg), 0,
                         std::numeric_limits<int>::max()));
        } else if (arg == "--rdma-log-level") {
            opts.rdma_log_level = take_value(i, argc, argv, arg);
        } else if (!arg.empty() && arg[0] == '-') {
            usage_error("unknown argument: " + arg);
        } else {
            if (positional_port_seen) {
                usage_error("multiple positional ports provided");
            }
            opts.port = static_cast<std::uint16_t>(
                parse_ll("port", arg, 1, 65535));
            positional_port_seen = true;
        }
    }

    if (opts.max_tokens == 0) {
        opts.max_tokens = opts.token_len;
    }
    if (opts.max_tokens < opts.token_len) {
        usage_error("--max-tokens must be >= --token-len");
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

template <typename T>
asio::awaitable<T> recv_pod(std::shared_ptr<rdmapp::qp> qp,
                            const char* what) {
    T value{};
    auto span = std::as_writable_bytes(std::span(&value, 1));
    auto [bytes, imm] = co_await qp->recv(span);
    (void)imm;
    if (bytes != sizeof(T)) {
        throw std::runtime_error(
            std::string("short recv for ") + what + ": got " +
            std::to_string(bytes) + " expected " + std::to_string(sizeof(T)));
    }
    co_return value;
}

template <typename T>
asio::awaitable<void> send_pod(std::shared_ptr<rdmapp::qp> qp,
                               const T& value) {
    auto span = std::as_bytes(std::span(&value, 1));
    co_await qp->send(span);
}

double elapsed_us(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

void raw_send_mr_wait(std::shared_ptr<rdmapp::qp> qp,
                      const rdmapp::local_mr& local_mr) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(local_mr.addr());
    sge.length = static_cast<std::uint32_t>(local_mr.length());
    sge.lkey = local_mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.next = nullptr;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    send_wr.wr_id = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    qp->post_send(send_wr, bad_send_wr);
    struct ibv_wc wc = {};
    while (!qp->poll_send_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw send failed: ") +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
}

void raw_recv_mr_wait(std::shared_ptr<rdmapp::qp> qp,
                      const rdmapp::local_mr& local_mr,
                      std::uint32_t expected_bytes,
                      const char* what) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(local_mr.addr());
    sge.length = expected_bytes;
    sge.lkey = local_mr.lkey();

    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr* bad_recv_wr = nullptr;
    recv_wr.next = nullptr;
    recv_wr.wr_id = 1;
    recv_wr.num_sge = 1;
    recv_wr.sg_list = &sge;

    struct ibv_wc wc = {};
    qp->post_recv(recv_wr, bad_recv_wr);
    while (!qp->poll_recv_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw recv failed for ") + what + ": " +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
    if (wc.byte_len != expected_bytes) {
        throw std::runtime_error(
            std::string("short raw recv for ") + what + ": got " +
            std::to_string(wc.byte_len) + " expected " +
            std::to_string(expected_bytes));
    }
}

template <typename T>
void raw_send_pod_wait(std::shared_ptr<rdmapp::qp> qp, const T& value) {
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(
            const_cast<T*>(&value),
            sizeof(T)));
    raw_send_mr_wait(qp, *mr);
}

template <typename T>
T raw_recv_pod_wait(std::shared_ptr<rdmapp::qp> qp, const char* what) {
    T value{};
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(&value, sizeof(T)));
    raw_recv_mr_wait(qp, *mr, sizeof(T), what);
    return value;
}

rdmapp::remote_mr raw_recv_remote_mr_wait(std::shared_ptr<rdmapp::qp> qp,
                                          const char* what) {
    char serialized[rdmapp::remote_mr::kSerializedSize];
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(serialized, sizeof(serialized)));
    raw_recv_mr_wait(
        qp,
        *mr,
        static_cast<std::uint32_t>(sizeof(serialized)),
        what);
    return rdmapp::remote_mr::deserialize(serialized);
}

double raw_rdma_wait(std::shared_ptr<rdmapp::qp> qp,
                     enum ibv_wr_opcode opcode,
                     rdmapp::remote_mr remote_mr,
                     const rdmapp::local_mr& local_mr,
                     void* local_addr,
                     std::uint32_t length) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(local_addr);
    sge.length = length;
    sge.lkey = local_mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = opcode;
    send_wr.next = nullptr;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    send_wr.wr_id = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.wr.rdma.remote_addr =
        reinterpret_cast<std::uint64_t>(remote_mr.addr());
    send_wr.wr.rdma.rkey = remote_mr.rkey();

    struct ibv_wc wc = {};
    const auto t0 = std::chrono::steady_clock::now();
    qp->post_send(send_wr, bad_send_wr);
    while (!qp->poll_send_cq(wc)) {
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw RDMA failed: ") +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
    return elapsed_us(t0, t1);
}

asio::awaitable<rdmapp::remote_mr>
recv_remote_mr(std::shared_ptr<rdmapp::qp> qp, const char* what) {
    char serialized[rdmapp::remote_mr::kSerializedSize];
    auto span = std::as_writable_bytes(std::span(serialized));
    auto [bytes, imm] = co_await qp->recv(span);
    (void)imm;
    if (bytes != sizeof(serialized)) {
        throw std::runtime_error(
            std::string("short recv for ") + what + " MR: got " +
            std::to_string(bytes));
    }
    co_return rdmapp::remote_mr::deserialize(serialized);
}

rdmapp::remote_mr sub_remote_mr(rdmapp::remote_mr mr,
                                std::uint64_t offset,
                                std::uint32_t length,
                                const char* name) {
    if (offset > mr.length() ||
        static_cast<std::uint64_t>(length) >
            static_cast<std::uint64_t>(mr.length()) - offset) {
        throw std::runtime_error(std::string(name) + " remote MR range overflow");
    }
    auto addr = reinterpret_cast<std::uintptr_t>(mr.addr());
    return rdmapp::remote_mr(
        reinterpret_cast<void*>(addr + offset), length, mr.rkey());
}

void validate_client_hello(const ClientHelloWire& hello,
                           const ServerOptions& opts) {
    if (hello.magic != promptdb_rdma::kClientHelloMagic ||
        hello.version != promptdb_rdma::kProtocolVersion) {
        throw std::runtime_error("bad client hello");
    }
    if (hello.max_tokens == 0 ||
        hello.max_tokens > static_cast<std::uint32_t>(opts.max_tokens)) {
        throw std::runtime_error("client max_tokens exceeds server limit");
    }
    const std::uint32_t min_result_bytes =
        promptdb_rdma::result_bytes_for_tokens(hello.max_tokens);
    const std::uint32_t max_result_bytes =
        promptdb_rdma::result_bytes_for_tokens(
            static_cast<std::uint32_t>(opts.max_tokens));
    if (hello.result_bytes < min_result_bytes) {
        throw std::runtime_error("client result buffer is too small");
    }
    if (hello.result_bytes > max_result_bytes) {
        throw std::runtime_error("client result buffer exceeds server limit");
    }
}

void fill_result_buffer(std::vector<std::uint8_t>& result_buffer,
                        std::uint64_t request_id,
                        const FindResult& result) {
    const bool found = result.best_idx >= 0;
    const std::uint32_t token_count = found
        ? promptdb_rdma::checked_u32_size(result.best_tokens.size(),
                                          "best token count")
        : 0;
    const std::uint32_t result_bytes =
        promptdb_rdma::result_bytes_for_tokens(token_count);
    if (result_bytes > result_buffer.size()) {
        throw std::runtime_error("result buffer is too small for best prompt");
    }

    ResultHeaderWire header;
    header.status = static_cast<std::uint16_t>(
        found ? Status::ok : Status::not_found);
    header.request_id = request_id;
    header.best_idx = result.best_idx;
    header.best_score = result.best_score;
    header.token_count = token_count;
    header.result_bytes = result_bytes;

    std::memcpy(result_buffer.data(), &header, sizeof(header));
    if (token_count != 0) {
        std::memcpy(result_buffer.data() + sizeof(header),
                    result.best_tokens.data(),
                    static_cast<std::size_t>(token_count) *
                        sizeof(std::uint32_t));
    }
}

void post_ack_without_wait(std::shared_ptr<rdmapp::qp> qp,
                           const rdmapp::local_mr& ack_mr) {
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<std::uint64_t>(ack_mr.addr());
    sge.length = static_cast<std::uint32_t>(ack_mr.length());
    sge.lkey = ack_mr.lkey();

    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr* bad_send_wr = nullptr;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.next = nullptr;
    send_wr.num_sge = 1;
    send_wr.sg_list = &sge;
    send_wr.wr_id = 0;
    send_wr.send_flags = 0;

    qp->post_send(send_wr, bad_send_wr);
}

asio::awaitable<void> handle_client(std::shared_ptr<AcceptedClient> client,
                                    std::shared_ptr<const PromptDbState> db,
                                    std::shared_ptr<const ServerOptions> opts) {
    auto qp = client->qp;
    ClientHelloWire hello =
        raw_recv_pod_wait<ClientHelloWire>(qp, "client hello");
    rdmapp::remote_mr prompt_remote =
        raw_recv_remote_mr_wait(qp, "prompt");
    rdmapp::remote_mr result_remote =
        raw_recv_remote_mr_wait(qp, "result");

    ServerHelloWire server_hello;
    server_hello.max_tokens = static_cast<std::uint32_t>(opts->max_tokens);
    server_hello.result_header_bytes = sizeof(ResultHeaderWire);
    std::exception_ptr hello_error;
    try {
        validate_client_hello(hello, *opts);
    } catch (...) {
        server_hello.status = static_cast<std::uint16_t>(Status::bad_request);
        hello_error = std::current_exception();
    }
    raw_send_pod_wait(qp, server_hello);
    if (hello_error) {
        std::rethrow_exception(hello_error);
    }

    std::vector<std::uint32_t> prompt_buffer(hello.max_tokens);
    std::vector<std::uint8_t> result_buffer(hello.result_bytes);
    auto prompt_mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(
            prompt_buffer.data(),
            prompt_buffer.size() * sizeof(std::uint32_t)));
    auto result_mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(result_buffer.data(), result_buffer.size()));
    constexpr std::size_t kAckRingSize = 128;
    std::vector<AckWire> ack_ring(kAckRingSize);
    std::vector<std::shared_ptr<rdmapp::local_mr>> ack_mrs;
    ack_mrs.reserve(kAckRingSize);
    for (auto& ack_slot : ack_ring) {
        ack_mrs.push_back(std::make_shared<rdmapp::local_mr>(
            qp->pd_ptr()->reg_mr(&ack_slot, sizeof(AckWire))));
    }
    std::size_t ack_slot_id = 0;

    while (true) {
        RequestWire req = raw_recv_pod_wait<RequestWire>(qp, "request");
        if (req.magic != promptdb_rdma::kRequestMagic ||
            req.version != promptdb_rdma::kProtocolVersion) {
            throw std::runtime_error("bad request header");
        }
        if (req.op == static_cast<std::uint16_t>(RequestOp::shutdown)) {
            co_return;
        }
        if (req.op != static_cast<std::uint16_t>(RequestOp::query)) {
            const std::size_t slot = ack_slot_id++ % kAckRingSize;
            AckWire& ack = ack_ring[slot];
            ack = AckWire{};
            ack.status = static_cast<std::uint16_t>(Status::bad_request);
            ack.request_id = req.request_id;
            post_ack_without_wait(qp, *ack_mrs[slot]);
            continue;
        }

        const std::size_t slot = ack_slot_id++ % kAckRingSize;
        AckWire& ack = ack_ring[slot];
        ack = AckWire{};
        ack.request_id = req.request_id;
        try {
            if (req.token_count == 0 || req.token_count > hello.max_tokens) {
                throw std::runtime_error("request token_count out of range");
            }

            const std::uint32_t prompt_bytes =
                promptdb_rdma::prompt_bytes_for_tokens(req.token_count);
            auto prompt_sub =
                sub_remote_mr(prompt_remote, 0, prompt_bytes, "prompt");
            ack.prompt_transfer_us = raw_rdma_wait(
                qp,
                IBV_WR_RDMA_READ,
                prompt_sub,
                *prompt_mr,
                prompt_buffer.data(),
                prompt_bytes);

            std::vector<std::uint32_t> tokens(
                prompt_buffer.begin(),
                prompt_buffer.begin() +
                    static_cast<std::ptrdiff_t>(req.token_count));
            const auto match_start = std::chrono::steady_clock::now();
            FindResult result = db->find_most_similar_prompt(tokens);
            const auto match_end = std::chrono::steady_clock::now();
            ack.server_match_us = elapsed_us(match_start, match_end);
            fill_result_buffer(result_buffer, req.request_id, result);

            const auto* header =
                reinterpret_cast<const ResultHeaderWire*>(result_buffer.data());
            auto result_sub =
                sub_remote_mr(result_remote, 0, header->result_bytes, "result");
            ack.result_transfer_us = raw_rdma_wait(
                qp,
                IBV_WR_RDMA_WRITE,
                result_sub,
                *result_mr,
                result_buffer.data(),
                header->result_bytes);

            ack.status = header->status;
            ack.result_bytes = header->result_bytes;
            ack.best_idx = header->best_idx;
        } catch (const std::exception& e) {
            spdlog::warn(
                "request {} failed: {}",
                static_cast<unsigned long long>(req.request_id),
                e.what());
            ack.status = static_cast<std::uint16_t>(Status::server_error);
            ack.best_idx = -1;
        }

        post_ack_without_wait(qp, *ack_mrs[slot]);
    }
}

asio::awaitable<void> server_loop(
    std::shared_ptr<PromptDbAcceptor> acceptor,
    std::shared_ptr<const PromptDbState> db,
    std::shared_ptr<const ServerOptions> opts) {
    auto ex = co_await asio::this_coro::executor;
    while (true) {
        auto client = co_await acceptor->accept();
        if (!client || !client->qp) {
            continue;
        }
        if (g_live_connections.load() >= opts->max_connections) {
            spdlog::warn("rejecting connection: max_connections reached");
            continue;
        }

        g_live_connections.fetch_add(1);
        asio::co_spawn(
            ex,
            [client, db, opts]() -> asio::awaitable<void> {
                try {
                    co_await handle_client(client, db, opts);
                } catch (const std::exception& e) {
                    spdlog::info("client handler ended: {}", e.what());
                }
                g_live_connections.fetch_sub(1);
                co_return;
            }(),
            asio::detached);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        ServerOptions parsed = parse_args(argc, argv);
        set_spdlog_level_from_name(parsed.rdma_log_level);
        auto opts = std::make_shared<ServerOptions>(std::move(parsed));

        PromptDbConfig config;
        config.prompts_len = opts->prompts;
        config.num_hash_planes = opts->num_hash_planes;
        config.vocab_size = opts->vocab_size;
        config.bucket_size = opts->bucket_size;
        config.seed = opts->seed;

        spdlog::info(
            "building PromptDB: prompts={} token_len={} vocab={} bucket_size={} seed={}",
            opts->prompts,
            opts->token_len,
            opts->vocab_size,
            opts->bucket_size,
            opts->seed);
        auto db = std::make_shared<const PromptDbState>(
            promptdb_rdma::make_prefilled_prompt_db(config, opts->token_len));
        spdlog::info("PromptDB ready: num_prompts={}", db->num_prompts());

        auto device = std::make_shared<rdmapp::device>(
            opts->ib_device, opts->ib_port);
        auto pd = std::make_shared<rdmapp::pd>(device);
        auto io_ctx = std::make_shared<asio::io_context>(opts->io_threads);
        auto acceptor = std::make_shared<PromptDbAcceptor>(
            io_ctx,
            opts->port,
            pd,
            device);

        spdlog::info(
            "starting RDMA PromptDB server: port={} ib_device={} ib_port={} io_threads={}",
            opts->port,
            opts->ib_device,
            opts->ib_port,
            opts->io_threads);

        auto guard = asio::make_work_guard(*io_ctx);
        asio::co_spawn(*io_ctx, server_loop(acceptor, db, opts), asio::detached);

        std::vector<std::jthread> workers;
        for (int i = 1; i < opts->io_threads; ++i) {
            workers.emplace_back([io_ctx] { io_ctx->run(); });
        }
        io_ctx->run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rdma_promptdb_server failed: " << e.what() << "\n";
        return 1;
    }
}
