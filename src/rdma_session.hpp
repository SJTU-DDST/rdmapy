#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <cstddef>

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

struct local_mr_cache_key {
    std::uintptr_t addr;
    uint64_t length;

    bool operator==(const local_mr_cache_key& other) const noexcept {
        return addr == other.addr && length == other.length;
    }
};

struct local_mr_cache_key_hash {
    std::size_t operator()(const local_mr_cache_key& key) const noexcept {
        std::size_t h1 = std::hash<std::uintptr_t>{}(key.addr);
        std::size_t h2 = std::hash<uint64_t>{}(key.length);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
    }
};

struct cached_local_mr {
    // Keep the owning Python tensor/storage alive until the MR is deregistered.
    // Field order matters: mr is destroyed before owner.
    std::shared_ptr<void> owner;
    std::shared_ptr<rdmapp::local_mr> mr;
    std::list<local_mr_cache_key>::iterator lru_it;
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

            if (r.length() > std::numeric_limits<uint64_t>::max() - prefix) {
                throw std::runtime_error("remote MR total length overflows uint64");
            }
            prefix += r.length();
            regions_.push_back(info);
        }
        total_len_ = prefix;

        std::cerr << "[RdmaSession] total_len=" << total_len_
                  << " num_regions=" << regions_.size() << std::endl;
    }

    size_t mr_cache_size() {
        std::lock_guard<std::mutex> lk(mutex_);
        return mr_cache_.size();
    }

    size_t mr_cache_limit() {
        std::lock_guard<std::mutex> lk(mutex_);
        return max_cached_mrs_;
    }

    uint64_t mr_cache_hits() {
        std::lock_guard<std::mutex> lk(mutex_);
        return mr_cache_hits_;
    }

    uint64_t mr_cache_misses() {
        std::lock_guard<std::mutex> lk(mutex_);
        return mr_cache_misses_;
    }

    void set_mr_cache_limit(size_t max_entries) {
        std::lock_guard<std::mutex> lk(mutex_);
        max_cached_mrs_ = max_entries;
        trim_mr_cache();
    }

    void clear_mr_cache() {
        std::lock_guard<std::mutex> lk(mutex_);
        mr_cache_.clear();
        mr_lru_.clear();
        mr_cache_hits_ = 0;
        mr_cache_misses_ = 0;
    }

    // 根据 offset 找 MR
    const remote_mr_info& find_region(uint64_t offset, uint64_t len) {
        if (len > total_len_ || offset > total_len_ - len) {
            throw std::runtime_error("offset does not fit in any MR region");
        }
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

        if (length > total_len_ || offset > total_len_ - length) {
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
                    uint64_t max_wr_bytes = std::numeric_limits<uint32_t>::max();
                    uint64_t chunk = std::min<uint64_t>(
                        std::min<uint64_t>(remaining, cap),
                        max_wr_bytes);

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

    // Synchronous read into an already allocated local buffer.
    //
    // The caller owns the buffer lifetime. This is intended for CUDA tensor
    // data_ptr() as well as ordinary host buffers. If local_addr is GPU memory
    // and the platform supports GPUDirect RDMA, ibv_reg_mr will register the
    // GPU pages and the NIC can DMA directly into the tensor.
    void read_into(uint64_t offset,
                   std::uintptr_t local_addr,
                   uint64_t length,
                   uint64_t chunk_bytes = 64ull * 1024 * 1024,
                   std::shared_ptr<void> owner = nullptr,
                   uint64_t register_length = 0) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (length > total_len_ || offset > total_len_ - length) {
            throw std::runtime_error("read_into out of range");
        }
        if (length > 0 && local_addr == 0) {
            throw std::runtime_error("read_into got null local address");
        }
        if (chunk_bytes == 0) {
            throw std::runtime_error("read_into chunk_bytes must be > 0");
        }
        if (length == 0) {
            return;
        }
        if (register_length == 0) {
            register_length = length;
        }
        if (register_length < length) {
            throw std::runtime_error("read_into register_length is smaller than length");
        }

        auto local_mr = get_cached_local_mr(
            local_addr, register_length, std::move(owner));

        auto fut = asio::co_spawn(
            *io_ctx_,
            [this, offset, local_addr, length, chunk_bytes, local_mr]() -> asio::awaitable<void> {
                uint64_t cur_off = offset;
                uint64_t remaining = length;
                uint64_t dst_offset = 0;

                while (remaining > 0) {
                    const auto& r = find_region_by_offset(cur_off);
                    uint64_t in_mr_offset = cur_off - r.prefix_len;
                    uint64_t cap = r.length - in_mr_offset;
                    uint64_t max_wr_bytes = std::min<uint64_t>(
                        chunk_bytes,
                        std::numeric_limits<uint32_t>::max());
                    uint64_t chunk = std::min<uint64_t>(
                        std::min<uint64_t>(remaining, cap),
                        max_wr_bytes);

                    void* remote_addr = (void*)(r.addr + in_mr_offset);
                    rdmapp::remote_mr remote(remote_addr, (uint32_t)chunk, r.rkey);

                    auto* dst = reinterpret_cast<void*>(local_addr + dst_offset);
                    co_await qp_->read(remote, local_mr, dst, static_cast<size_t>(chunk));

                    cur_off    += chunk;
                    dst_offset += chunk;
                    remaining  -= chunk;
                }
                co_return;
            }(),
            asio::use_future
        );

        fut.get();
    }


        // 同步 write，支持跨多个 MR
    void write(uint64_t offset, const uint8_t* data, size_t len) {
        std::lock_guard<std::mutex> lk(mutex_);

        // std::cerr << "[RdmaSession::write] offset=" << offset
        //           << " len=" << len
        //           << " total_len_=" << total_len_ << std::endl;

        if (len > total_len_ || offset > total_len_ - len) {
            std::cerr << "[RdmaSession::write] OUT OF RANGE! "
                      << "offset=" << offset << " len=" << len
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
                    uint64_t max_wr_bytes = std::numeric_limits<uint32_t>::max();
                    uint64_t chunk = std::min<uint64_t>(
                        std::min<uint64_t>((uint64_t)remaining, cap),
                        max_wr_bytes);

                    // std::cerr << "[RdmaSession::write] region prefix="
                    //           << r.prefix_len
                    //           << " region.len=" << r.length
                    //           << " in_mr_offset=" << in_mr_offset
                    //           << " chunk=" << chunk
                    //           << " r.addr=" << std::hex << r.addr << std::dec
                    //           << std::endl;

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

    // Synchronous write from an already allocated local buffer.
    //
    // Passing a CUDA tensor data_ptr() here enables the local side of
    // GPUDirect RDMA when the NIC/driver stack supports registering GPU
    // memory. The caller must ensure any CUDA work producing this buffer has
    // completed before calling this method.
    void write_from(uint64_t offset,
                    std::uintptr_t local_addr,
                    uint64_t length,
                    uint64_t chunk_bytes = 64ull * 1024 * 1024,
                    std::shared_ptr<void> owner = nullptr,
                    uint64_t register_length = 0) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (length > total_len_ || offset > total_len_ - length) {
            std::cerr << "[RdmaSession::write_from] OUT OF RANGE! "
                      << "offset=" << offset << " len=" << length
                      << " total_len_=" << total_len_ << std::endl;
            throw std::runtime_error("write_from out of range");
        }
        if (length > 0 && local_addr == 0) {
            throw std::runtime_error("write_from got null local address");
        }
        if (chunk_bytes == 0) {
            throw std::runtime_error("write_from chunk_bytes must be > 0");
        }
        if (length == 0) {
            return;
        }
        if (register_length == 0) {
            register_length = length;
        }
        if (register_length < length) {
            throw std::runtime_error("write_from register_length is smaller than length");
        }

        auto local_mr = get_cached_local_mr(
            local_addr, register_length, std::move(owner));

        auto fut = asio::co_spawn(
            *io_ctx_,
            [this, offset, local_addr, length, chunk_bytes, local_mr]() -> asio::awaitable<void> {
                uint64_t cur_off = offset;
                uint64_t remaining = length;
                uint64_t src_offset = 0;

                while (remaining > 0) {
                    const auto& r = find_region_by_offset(cur_off);
                    uint64_t in_mr_offset = cur_off - r.prefix_len;
                    uint64_t cap = r.length - in_mr_offset;
                    uint64_t max_wr_bytes = std::min<uint64_t>(
                        chunk_bytes,
                        std::numeric_limits<uint32_t>::max());
                    uint64_t chunk = std::min<uint64_t>(
                        std::min<uint64_t>(remaining, cap),
                        max_wr_bytes);

                    void* remote_addr = (void*)(r.addr + in_mr_offset);
                    rdmapp::remote_mr remote(remote_addr, (uint32_t)chunk, r.rkey);

                    auto* src = reinterpret_cast<void*>(local_addr + src_offset);
                    co_await qp_->write(remote, local_mr, src, static_cast<size_t>(chunk));

                    cur_off    += chunk;
                    src_offset += chunk;
                    remaining  -= chunk;
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

    size_t max_cached_mrs_ = 16;
    uint64_t mr_cache_hits_ = 0;
    uint64_t mr_cache_misses_ = 0;
    std::list<local_mr_cache_key> mr_lru_;
    std::unordered_map<
        local_mr_cache_key,
        cached_local_mr,
        local_mr_cache_key_hash
    > mr_cache_;

    std::shared_ptr<rdmapp::local_mr> register_local_mr(
        std::uintptr_t addr,
        uint64_t length) {
        if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("local MR length exceeds size_t");
        }
        return std::make_shared<rdmapp::local_mr>(
            qp_->pd_ptr()->reg_mr(
                reinterpret_cast<void*>(addr),
                static_cast<size_t>(length)));
    }

    std::shared_ptr<rdmapp::local_mr> get_cached_local_mr(
        std::uintptr_t addr,
        uint64_t length,
        std::shared_ptr<void> owner) {
        if (max_cached_mrs_ == 0) {
            ++mr_cache_misses_;
            return register_local_mr(addr, length);
        }

        local_mr_cache_key key{addr, length};
        auto it = mr_cache_.find(key);
        if (it != mr_cache_.end()) {
            ++mr_cache_hits_;
            mr_lru_.splice(mr_lru_.begin(), mr_lru_, it->second.lru_it);
            it->second.lru_it = mr_lru_.begin();
            return it->second.mr;
        }

        for (auto cover = mr_cache_.begin(); cover != mr_cache_.end(); ++cover) {
            if (contains_local_range(cover->first, addr, length)) {
                ++mr_cache_hits_;
                mr_lru_.splice(mr_lru_.begin(), mr_lru_, cover->second.lru_it);
                cover->second.lru_it = mr_lru_.begin();
                return cover->second.mr;
            }
        }

        ++mr_cache_misses_;
        auto mr = register_local_mr(addr, length);
        mr_lru_.push_front(key);
        mr_cache_.emplace(
            key,
            cached_local_mr{std::move(owner), mr, mr_lru_.begin()});
        trim_mr_cache();
        return mr;
    }

    void trim_mr_cache() {
        while (mr_cache_.size() > max_cached_mrs_ && !mr_lru_.empty()) {
            auto old_key = mr_lru_.back();
            mr_cache_.erase(old_key);
            mr_lru_.pop_back();
        }
    }

    bool contains_local_range(const local_mr_cache_key& key,
                              std::uintptr_t addr,
                              uint64_t length) const {
        if (addr < key.addr) {
            return false;
        }
        uint64_t delta = static_cast<uint64_t>(addr - key.addr);
        return delta <= key.length && length <= key.length - delta;
    }

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
    explicit RdmaEndpoint(int ib_port = 1, int device_num = 1) {
        device_ = std::make_shared<rdmapp::device>(device_num, ib_port);
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
