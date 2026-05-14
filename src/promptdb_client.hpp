#pragma once

#include "promptdb_lsh.hpp"

#include "qp_connector.h"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/use_future.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <infiniband/verbs.h>
#include <rdmapp/cq.h>
#include <rdmapp/device.h>
#include <rdmapp/mr.h>
#include <rdmapp/pd.h>
#include <rdmapp/qp.h>

namespace promptdb_rdma {

struct PromptDbClientResult {
    int best_idx = -1;
    double best_score = -1.0;
    std::vector<std::uint32_t> best_tokens;
};

namespace detail {

inline void raw_send_mr_wait(std::shared_ptr<rdmapp::qp> qp,
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

inline void raw_send_mr_nowait(std::shared_ptr<rdmapp::qp> qp,
                               const rdmapp::local_mr& local_mr,
                               bool signaled) {
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
    send_wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;

    qp->post_send(send_wr, bad_send_wr);
}

inline void raw_wait_send_completion(std::shared_ptr<rdmapp::qp> qp,
                                     const char* what) {
    struct ibv_wc wc = {};
    while (!qp->poll_send_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw send completion failed for ") + what + ": " +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
}

struct RawRecvState {
    std::shared_ptr<rdmapp::qp> qp;
    std::uint32_t expected_bytes = 0;
    const char* what = "recv";
};

inline RawRecvState raw_post_recv_mr(std::shared_ptr<rdmapp::qp> qp,
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

    qp->post_recv(recv_wr, bad_recv_wr);
    return RawRecvState{std::move(qp), expected_bytes, what};
}

inline void raw_wait_recv(const RawRecvState& state) {
    struct ibv_wc wc = {};
    while (!state.qp->poll_recv_cq(wc)) {
    }
    if (wc.status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            std::string("raw recv failed for ") + state.what + ": " +
            ibv_wc_status_str(wc.status) +
            " (status=" + std::to_string(static_cast<int>(wc.status)) + ")");
    }
    if (wc.byte_len != state.expected_bytes) {
        throw std::runtime_error(
            std::string("short raw recv for ") + state.what + ": got " +
            std::to_string(wc.byte_len) + " expected " +
            std::to_string(state.expected_bytes));
    }
}

template <typename T>
inline void raw_send_pod_wait(std::shared_ptr<rdmapp::qp> qp,
                              const T& value) {
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(const_cast<T*>(&value), sizeof(T)));
    raw_send_mr_wait(qp, *mr);
}

inline void raw_send_bytes_wait(std::shared_ptr<rdmapp::qp> qp,
                                std::vector<std::uint8_t>& bytes) {
    auto mr = std::make_shared<rdmapp::local_mr>(
        qp->pd_ptr()->reg_mr(bytes.data(), bytes.size()));
    raw_send_mr_wait(qp, *mr);
}

} // namespace detail

class PromptDbRdmaClient {
public:
    PromptDbRdmaClient(std::string host,
                       std::uint16_t port,
                       std::uint32_t max_tokens,
                       int ib_device = 1,
                       int ib_port = 1,
                       int io_workers = 1,
                       std::size_t cq_batch_size = 16)
        : host_(std::move(host)),
          port_(port),
          max_tokens_(max_tokens),
          device_(std::make_shared<rdmapp::device>(ib_device, ib_port)),
          pd_(std::make_shared<rdmapp::pd>(device_)),
          recv_cq_(std::make_shared<rdmapp::cq>(device_)),
          send_cq_(std::make_shared<rdmapp::cq>(device_)),
          io_ctx_(std::make_shared<asio::io_context>(io_workers)),
          prompt_buffer_(max_tokens_, 0),
          result_buffer_(result_bytes_for_tokens(max_tokens_), 0) {
        if (max_tokens_ == 0) {
            throw std::runtime_error("max_tokens must be positive");
        }

        work_guard_.emplace(asio::make_work_guard(*io_ctx_));
        io_threads_.reserve(static_cast<std::size_t>(io_workers));
        for (int i = 0; i < io_workers; ++i) {
            io_threads_.emplace_back([ctx = io_ctx_] { ctx->run(); });
        }

        prompt_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(prompt_buffer_.data(),
                        prompt_buffer_.size() * sizeof(std::uint32_t)));
        result_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(result_buffer_.data(), result_buffer_.size()));
        ack_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(&ack_buffer_, sizeof(ack_buffer_)));
        request_mr_ = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(&request_buffer_, sizeof(request_buffer_)));

        auto connector = std::make_shared<rdmapp::qp_connector>(
            host_, port_, pd_, recv_cq_, send_cq_);
        auto connect_future = asio::co_spawn(
            *io_ctx_,
            [connector]() -> asio::awaitable<std::shared_ptr<rdmapp::qp>> {
                co_return co_await connector->connect();
            }(),
            asio::use_future);
        qp_ = connect_future.get();
        if (!qp_) {
            throw std::runtime_error("failed to connect PromptDB RDMA QP");
        }
        handshake();

        work_guard_.reset();
        io_ctx_->stop();
        io_threads_.clear();
    }

    ~PromptDbRdmaClient() {
        close_noexcept();
        work_guard_.reset();
        if (io_ctx_) {
            io_ctx_->stop();
        }
    }

    std::optional<int> add_prompt(int idx,
                                  const std::vector<std::uint32_t>& tokens) {
        AckWire ack = issue_request(
            RequestOp::add_prompt,
            tokens,
            next_request_id(),
            idx);
        if (ack.status != static_cast<std::uint16_t>(Status::ok)) {
            throw std::runtime_error(
                "add_prompt failed with status=" +
                std::to_string(ack.status));
        }
        if (ack.best_idx >= 0) {
            return ack.best_idx;
        }
        return std::nullopt;
    }

    PromptDbClientResult find_most_similar_prompt(
        const std::vector<std::uint32_t>& tokens) {
        const std::uint64_t request_id = next_request_id();
        AckWire ack = issue_request(RequestOp::query, tokens, request_id, -1);
        if (ack.status == static_cast<std::uint16_t>(Status::not_found)) {
            return {};
        }
        if (ack.status != static_cast<std::uint16_t>(Status::ok)) {
            throw std::runtime_error(
                "query failed with status=" + std::to_string(ack.status));
        }
        return parse_result(request_id, ack);
    }

    void close() {
        close_noexcept();
    }

private:
    void handshake() {
        ClientHelloWire hello;
        hello.max_tokens = max_tokens_;
        hello.result_bytes = static_cast<std::uint32_t>(result_buffer_.size());

        ServerHelloWire resp{};
        auto resp_mr = std::make_shared<rdmapp::local_mr>(
            pd_->reg_mr(&resp, sizeof(resp)));
        auto resp_state = detail::raw_post_recv_mr(
            qp_, *resp_mr, sizeof(resp), "server hello");

        detail::raw_send_pod_wait(qp_, hello);

        auto prompt_serialized = prompt_mr_->serialize();
        detail::raw_send_bytes_wait(qp_, prompt_serialized);
        auto result_serialized = result_mr_->serialize();
        detail::raw_send_bytes_wait(qp_, result_serialized);

        detail::raw_wait_recv(resp_state);
        if (resp.magic != kServerHelloMagic ||
            resp.version != kProtocolVersion ||
            resp.status != static_cast<std::uint16_t>(Status::ok)) {
            throw std::runtime_error("server rejected PromptDB RDMA handshake");
        }
    }

    AckWire issue_request(RequestOp op,
                          const std::vector<std::uint32_t>& tokens,
                          std::uint64_t request_id,
                          int prompt_idx) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (closed_) {
            throw std::runtime_error("PromptDB RDMA client is closed");
        }
        if (tokens.empty() || tokens.size() > max_tokens_) {
            throw std::runtime_error("token count out of range for PromptDB RDMA request");
        }

        std::copy(tokens.begin(), tokens.end(), prompt_buffer_.begin());
        request_buffer_ = RequestWire{};
        request_buffer_.op = static_cast<std::uint16_t>(op);
        request_buffer_.request_id = request_id;
        request_buffer_.token_count = static_cast<std::uint32_t>(tokens.size());
        request_buffer_.prompt_idx = prompt_idx;

        auto ack_state = detail::raw_post_recv_mr(
            qp_, *ack_mr_, sizeof(ack_buffer_), "ack");
        std::atomic_thread_fence(std::memory_order_release);
        const bool signal_request =
            (++request_send_count_ % kSignalRequestEvery) == 0;
        detail::raw_send_mr_nowait(qp_, *request_mr_, signal_request);
        detail::raw_wait_recv(ack_state);
        if (signal_request) {
            detail::raw_wait_send_completion(qp_, "request");
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        AckWire ack = ack_buffer_;
        if (ack.magic != kAckMagic ||
            ack.version != kProtocolVersion ||
            ack.request_id != request_id) {
            throw std::runtime_error("bad PromptDB RDMA ack");
        }
        return ack;
    }

    PromptDbClientResult parse_result(std::uint64_t request_id,
                                      const AckWire& ack) const {
        if (ack.result_bytes < sizeof(ResultHeaderWire) ||
            ack.result_bytes > result_buffer_.size()) {
            throw std::runtime_error("bad result byte count in PromptDB RDMA ack");
        }

        ResultHeaderWire header;
        std::memcpy(&header, result_buffer_.data(), sizeof(header));
        if (header.magic != kResultMagic ||
            header.version != kProtocolVersion ||
            header.request_id != request_id ||
            header.result_bytes != ack.result_bytes ||
            header.result_bytes != result_bytes_for_tokens(header.token_count)) {
            throw std::runtime_error("bad PromptDB RDMA result header");
        }
        if (header.status == static_cast<std::uint16_t>(Status::not_found) ||
            header.best_idx < 0) {
            return {};
        }
        if (header.status != static_cast<std::uint16_t>(Status::ok)) {
            throw std::runtime_error(
                "bad PromptDB RDMA result status=" +
                std::to_string(header.status));
        }

        PromptDbClientResult result;
        result.best_idx = header.best_idx;
        result.best_score = header.best_score;
        result.best_tokens.resize(header.token_count);
        if (header.token_count != 0) {
            std::memcpy(
                result.best_tokens.data(),
                result_buffer_.data() + sizeof(ResultHeaderWire),
                static_cast<std::size_t>(header.token_count) *
                    sizeof(std::uint32_t));
        }
        return result;
    }

    std::uint64_t next_request_id() {
        return next_request_id_++;
    }

    void close_noexcept() noexcept {
        std::lock_guard<std::mutex> lk(mutex_);
        if (closed_ || !qp_) {
            return;
        }
        closed_ = true;
        try {
            request_buffer_ = RequestWire{};
            request_buffer_.op = static_cast<std::uint16_t>(RequestOp::shutdown);
            request_buffer_.request_id =
                std::numeric_limits<std::uint64_t>::max();
            detail::raw_send_mr_wait(qp_, *request_mr_);
        } catch (...) {
        }
    }

    std::string host_;
    std::uint16_t port_ = 0;
    std::uint32_t max_tokens_ = 0;
    std::shared_ptr<rdmapp::device> device_;
    std::shared_ptr<rdmapp::pd> pd_;
    std::shared_ptr<rdmapp::cq> recv_cq_;
    std::shared_ptr<rdmapp::cq> send_cq_;
    std::shared_ptr<asio::io_context> io_ctx_;
    std::shared_ptr<rdmapp::qp> qp_;
    std::vector<std::uint32_t> prompt_buffer_;
    std::vector<std::uint8_t> result_buffer_;
    std::shared_ptr<rdmapp::local_mr> prompt_mr_;
    std::shared_ptr<rdmapp::local_mr> result_mr_;
    AckWire ack_buffer_{};
    std::shared_ptr<rdmapp::local_mr> ack_mr_;
    RequestWire request_buffer_{};
    std::shared_ptr<rdmapp::local_mr> request_mr_;
    static constexpr std::uint32_t kSignalRequestEvery = 32;
    std::uint32_t request_send_count_ = 0;
    std::uint64_t next_request_id_ = 1;
    std::vector<std::jthread> io_threads_;
    std::optional<asio::executor_work_guard<asio::io_context::executor_type>>
        work_guard_;
    std::mutex mutex_;
    bool closed_ = false;
};

} // namespace promptdb_rdma
