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

struct remote_mr_info {
    uint64_t addr;
    uint32_t rkey;
    uint64_t length;
    uint64_t prefix_len;   // 当前 MR 在连续地址空间中的起始 offset（前缀和）
};

// ================================
// RdmaSession: 支持多个 remote_mr 的 KV 读写
// ================================
class RdmaSession {
public:
    RdmaSession(std::shared_ptr<rdmapp::qp> qp,
                std::shared_ptr<asio::io_context> io_ctx,
                std::vector<rdmapp::remote_mr> remotes)
        : qp_(std::move(qp)),
          io_ctx_(std::move(io_ctx))
    {
        uint64_t prefix = 0;
        for (auto &r : remotes) {
            remote_mr_info info;
            info.addr = (uint64_t)r.addr();
            info.length = r.length();
            info.rkey = r.rkey();
            info.prefix_len = prefix;

            prefix += r.length();
            regions_.push_back(info);
        }
        total_len_ = prefix;

        std::cerr << "[RdmaSession] total_len=" << total_len_
                  << " num_regions=" << regions_.size() << std::endl;
    }

    // 根据 offset 找 MR
    const remote_mr_info& find_region(uint64_t offset, uint64_t len) {
        uint64_t end = offset + len;
        for (auto &r : regions_) {
            if (offset >= r.prefix_len &&
                end <= r.prefix_len + r.length) {
                return r;
            }
        }
        throw std::runtime_error("offset does not fit in any MR region");
    }

    // 同步 read
    std::vector<uint8_t> read(uint64_t offset, uint64_t length) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (offset + length > total_len_)
            throw std::runtime_error("read out of range");

        const auto& r = find_region(offset, length);
        uint64_t in_mr_offset = offset - r.prefix_len;

        std::vector<uint8_t> buffer(length);
        auto span = std::as_writable_bytes(std::span(buffer));

        void* addr = (void*)(r.addr + in_mr_offset);
        rdmapp::remote_mr remote(addr, length, r.rkey);

        auto fut = asio::co_spawn(
            *io_ctx_, qp_->read(remote, span), asio::use_future);
        fut.get();
        return buffer;
    }

    // 同步 write
    void write(uint64_t offset, const uint8_t* data, size_t len) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (offset + len > total_len_)
            throw std::runtime_error("write out of range");

        const auto& r = find_region(offset, len);
        uint64_t in_mr_offset = offset - r.prefix_len;

        void* addr = (void*)(r.addr + in_mr_offset);
        rdmapp::remote_mr remote(addr, (uint32_t)len, r.rkey);

        auto span = std::as_bytes(std::span(data, len));
        auto fut = asio::co_spawn(
            *io_ctx_, qp_->write(remote, span), asio::use_future);
        fut.get();
    }

private:
    std::shared_ptr<rdmapp::qp> qp_;
    std::shared_ptr<asio::io_context> io_ctx_;

    std::vector<remote_mr_info> regions_;
    uint64_t total_len_;

    std::mutex mutex_;
};

// ====================================
// RdmaEndpoint：接收多个 MR
// ====================================
class RdmaEndpoint {
public:
    explicit RdmaEndpoint(int ib_port = 1) {
        device_ = std::make_shared<rdmapp::device>(1, ib_port);
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

    std::shared_ptr<RdmaSession> connect(const std::string& host,
                                         uint16_t port)
    {
        auto connector =
            std::make_shared<rdmapp::qp_connector>(host, port, pd_, cq_);
        auto io_ctx = io_ctx_;

        auto fut = asio::co_spawn(
            *io_ctx_,
            [connector, io_ctx]() -> asio::awaitable<std::shared_ptr<RdmaSession>> {
                auto qp = co_await connector->connect();

                // 先接收 n_mr
                uint32_t n_mr = 0;
                {
                    auto span = std::as_writable_bytes(std::span(&n_mr, 1));
                    co_await qp->recv(span);
                }

                std::vector<rdmapp::remote_mr> remotes;
                remotes.reserve(n_mr);

                for (uint32_t i = 0; i < n_mr; i++) {
                    char serialized[rdmapp::remote_mr::kSerializedSize];
                    auto span = std::as_writable_bytes(std::span(serialized));
                    co_await qp->recv(span);
                    remotes.push_back(rdmapp::remote_mr::deserialize(serialized));
                }

                co_return std::make_shared<RdmaSession>(qp, io_ctx, remotes);
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
