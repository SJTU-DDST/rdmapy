#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

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
        // 同步 read，支持跨多个 MR
    std::vector<uint8_t> read(uint64_t offset, uint64_t length) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (offset + length > total_len_) {
            throw std::runtime_error("read out of range");
        }

        // 用 shared_ptr 保证在协程中使用时生命周期安全
        auto buffer = std::make_shared<std::vector<uint8_t>>(length);

        auto fut = asio::co_spawn(
            *io_ctx_,
            [this, offset, length, buffer]() -> asio::awaitable<void> {
                uint64_t cur_off = offset;
                uint64_t remaining = length;
                uint64_t dst_offset = 0;

                while (remaining > 0) {
                    const auto& r = find_region_by_offset(cur_off);
                    uint64_t in_mr_offset = cur_off - r.prefix_len;
                    uint64_t cap = r.length - in_mr_offset;        // 该 MR 中还能读多少
                    uint64_t chunk = std::min<uint64_t>(remaining, cap);

                    void* addr = (void*)(r.addr + in_mr_offset);
                    rdmapp::remote_mr remote(addr, (uint32_t)chunk, r.rkey);

                    auto span = std::as_writable_bytes(
                        std::span(buffer->data() + dst_offset, chunk));

                    co_await qp_->read(remote, span);

                    cur_off    += chunk;
                    dst_offset += chunk;
                    remaining  -= chunk;
                }
                co_return;
            }(),
            asio::use_future
        );

        fut.get();
        return *buffer;
    }


        // 同步 write，支持跨多个 MR
    void write(uint64_t offset, const uint8_t* data, size_t len) {
        std::lock_guard<std::mutex> lk(mutex_);

        std::cerr << "[RdmaSession::write] offset=" << offset
                  << " len=" << len
                  << " total_len_=" << total_len_ << std::endl;

        if (offset + len > total_len_) {
            std::cerr << "[RdmaSession::write] OUT OF RANGE! "
                      << "offset+len=" << (offset + len)
                      << " total_len_=" << total_len_ << std::endl;
            throw std::runtime_error("write out of range");
        }

        auto fut = asio::co_spawn(
            *io_ctx_,
            [this, offset, len, data]() -> asio::awaitable<void> {
                uint64_t cur_off  = offset;
                size_t   remaining = len;
                const uint8_t* ptr = data;

                while (remaining > 0) {
                    const auto& r = find_region_by_offset(cur_off);
                    uint64_t in_mr_offset = cur_off - r.prefix_len;
                    uint64_t cap = r.length - in_mr_offset;      // 该 MR 中还能写多少
                    uint64_t chunk = std::min<uint64_t>(
                        (uint64_t)remaining, cap);

                    std::cerr << "[RdmaSession::write] region prefix="
                              << r.prefix_len
                              << " region.len=" << r.length
                              << " in_mr_offset=" << in_mr_offset
                              << " chunk=" << chunk
                              << " r.addr=" << std::hex << r.addr << std::dec
                              << std::endl;

                    void* addr = (void*)(r.addr + in_mr_offset);
                    rdmapp::remote_mr remote(addr, (uint32_t)chunk, r.rkey);

                    auto span = std::as_bytes(std::span(ptr, chunk));
                    co_await qp_->write(remote, span);

                    cur_off   += chunk;
                    ptr       += chunk;
                    remaining -= chunk;
                }

                co_return;
            }(),
            asio::use_future
        );

        fut.get();
    }


private:
    std::shared_ptr<rdmapp::qp> qp_;
    std::shared_ptr<asio::io_context> io_ctx_;

    std::vector<remote_mr_info> regions_;
    uint64_t total_len_;

    std::mutex mutex_;

    const remote_mr_info& find_region_by_offset(uint64_t offset) {
        for (auto &r : regions_) {
            if (offset >= r.prefix_len && offset < r.prefix_len + r.length) {
                return r;
            }
        }
        throw std::runtime_error("offset does not fit in any MR region");
    }
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
