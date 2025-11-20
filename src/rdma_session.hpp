#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/use_future.hpp>

#include <rdmapp/rdmapp.h>
#include <rdmapp/mr.h>
#include <rdmapp/qp.h>
#include <rdmapp/executor.h>

#include "qp_connector.h"

// ================================
// RdmaSession: 基于 remote_mr 的 KV 读写
// ================================
class RdmaSession {
public:
    RdmaSession(std::shared_ptr<rdmapp::qp> qp,
                std::shared_ptr<asio::io_context> io_ctx,
                rdmapp::remote_mr remote)
        : qp_(std::move(qp)),
          io_ctx_(std::move(io_ctx)),
          base_addr_(reinterpret_cast<std::uint64_t>(remote.addr())),
          rkey_(remote.rkey()),
          total_len_(static_cast<std::uint32_t>(remote.length())) {}

    // 同步 read：按 offset 读取
    std::vector<std::uint8_t> read(std::uint32_t offset,
                                   std::uint32_t length) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (offset + length > total_len_) {
            throw std::runtime_error("read out of range");
        }

        std::vector<std::uint8_t> buffer(length);
        auto span = std::as_writable_bytes(std::span(buffer));

        void *addr = reinterpret_cast<void*>(base_addr_ + offset);
        rdmapp::remote_mr remote(addr, length, rkey_);

        auto fut = asio::co_spawn(
            *io_ctx_,
            qp_->read(remote, span),
            asio::use_future
        );
        std::uint32_t n = fut.get();
        buffer.resize(n);
        return buffer;
    }

    // 同步 write：按 offset 写
    void write(std::uint32_t offset,
               const std::uint8_t *data,
               std::size_t len) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (offset + len > total_len_) {
            throw std::runtime_error("write out of range");
        }

        void *addr = reinterpret_cast<void*>(base_addr_ + offset);
        rdmapp::remote_mr remote(addr,
                                 static_cast<std::uint32_t>(len),
                                 rkey_);

        auto span = std::as_bytes(std::span(data, len));
        auto fut = asio::co_spawn(
            *io_ctx_,
            qp_->write(remote, span),
            asio::use_future
        );
        fut.get();
    }

private:
    std::shared_ptr<rdmapp::qp> qp_;
    std::shared_ptr<asio::io_context> io_ctx_;
    std::mutex mutex_;

    std::uint64_t base_addr_;
    std::uint32_t rkey_;
    std::uint32_t total_len_;
};

// ====================================
// RdmaEndpoint：负责 connect() + MR 握手
// ====================================
class RdmaEndpoint {
public:
    explicit RdmaEndpoint(int ib_port = 1) {
        device_ = std::make_shared<rdmapp::device>(0, ib_port);
        pd_     = std::make_shared<rdmapp::pd>(device_);
        cq_     = std::make_shared<rdmapp::cq>(device_);

        io_ctx_   = std::make_shared<asio::io_context>(4);
        executor_ = std::make_shared<rdmapp::executor>(io_ctx_);
        cq_poller_ =
            std::make_shared<rdmapp::cq_poller>(cq_, executor_);

        worker_ = std::jthread([this]() {
            auto guard = asio::make_work_guard(*io_ctx_);
            io_ctx_->run();
        });
    }

    std::shared_ptr<RdmaSession> connect(const std::string &host,
                                         std::uint16_t port) {
        auto connector =
            std::make_shared<rdmapp::qp_connector>(host, port, pd_, cq_);
        auto io_ctx = io_ctx_;

        auto fut = asio::co_spawn(
            *io_ctx_,
            [connector, io_ctx]()
                -> asio::awaitable<std::shared_ptr<RdmaSession>> {
                // 1. 建立 QP
                auto qp = co_await connector->connect();

                // 2. 接收 server 发来的 remote_mr
                char serialized[rdmapp::remote_mr::kSerializedSize];
                auto recv_span =
                    std::as_writable_bytes(std::span(serialized));
                co_await qp->recv(recv_span);
                auto remote =
                    rdmapp::remote_mr::deserialize(serialized);

                co_return std::make_shared<RdmaSession>(qp, io_ctx, remote);
            }(),
            asio::use_future
        );
        return fut.get();
    }

private:
    std::shared_ptr<rdmapp::device> device_;
    std::shared_ptr<rdmapp::pd> pd_;
    std::shared_ptr<rdmapp::cq> cq_;
    std::shared_ptr<asio::io_context> io_ctx_;
    std::shared_ptr<rdmapp::executor> executor_;
    std::shared_ptr<rdmapp::cq_poller> cq_poller_;
    std::jthread worker_;
};
