#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <condition_variable>
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
#include <spdlog/spdlog.h>

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

        spdlog::debug("[RdmaSession] total_len={} num_regions={}",
                      total_len_,
                      regions_.size());
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
                   uint64_t register_length = 0,
                   uint32_t max_inflight = 16,
                   bool debug = false) {
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

        uint64_t hits_before = 0;
        uint64_t misses_before = 0;
        std::shared_ptr<rdmapp::local_mr> local_mr;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            hits_before = mr_cache_hits_;
            misses_before = mr_cache_misses_;
            local_mr = get_cached_local_mr(
                local_addr, register_length, std::move(owner));
        }

        auto t0 = std::chrono::steady_clock::now();
        size_t chunks = 0;
        const remote_mr_info* single_region =
            length <= chunk_bytes ? find_region_for_range(offset, length) : nullptr;
        if (single_region != nullptr &&
            length <= std::numeric_limits<uint32_t>::max()) {
            chunks = post_single_rdma(
                IBV_WR_RDMA_READ,
                *single_region,
                offset,
                local_addr,
                static_cast<uint32_t>(length),
                local_mr);
        } else {
            chunks = post_windowed_rdma(
                IBV_WR_RDMA_READ,
                offset,
                local_addr,
                length,
                chunk_bytes,
                local_mr,
                max_inflight
            );
        }
        debug_report(
            debug,
            "read_into",
            offset,
            length,
            chunk_bytes,
            max_inflight,
            chunks,
            t0,
            hits_before,
            misses_before);
    }

    // Same operation as read_into(), but returns the measured submit-to-complete
    // latency in microseconds. Setup work such as MR cache lookup and region
    // lookup is intentionally done before the timer starts.
    double read_into_transfer_latency_us(
                   uint64_t offset,
                   std::uintptr_t local_addr,
                   uint64_t length,
                   uint64_t chunk_bytes = 64ull * 1024 * 1024,
                   std::shared_ptr<void> owner = nullptr,
                   uint64_t register_length = 0,
                   uint32_t max_inflight = 16,
                   bool debug = false) {
        if (length > total_len_ || offset > total_len_ - length) {
            throw std::runtime_error("read_into_transfer_latency_us out of range");
        }
        if (length > 0 && local_addr == 0) {
            throw std::runtime_error("read_into_transfer_latency_us got null local address");
        }
        if (chunk_bytes == 0) {
            throw std::runtime_error("read_into_transfer_latency_us chunk_bytes must be > 0");
        }
        if (length == 0) {
            return 0.0;
        }
        if (register_length == 0) {
            register_length = length;
        }
        if (register_length < length) {
            throw std::runtime_error(
                "read_into_transfer_latency_us register_length is smaller than length");
        }

        uint64_t hits_before = 0;
        uint64_t misses_before = 0;
        std::shared_ptr<rdmapp::local_mr> local_mr;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            hits_before = mr_cache_hits_;
            misses_before = mr_cache_misses_;
            local_mr = get_cached_local_mr(
                local_addr, register_length, std::move(owner));
        }

        auto debug_t0 = std::chrono::steady_clock::now();
        double transfer_us = 0.0;
        size_t chunks = 0;
        const remote_mr_info* single_region =
            length <= chunk_bytes ? find_region_for_range(offset, length) : nullptr;
        if (single_region != nullptr &&
            length <= std::numeric_limits<uint32_t>::max()) {
            chunks = post_single_rdma(
                IBV_WR_RDMA_READ,
                *single_region,
                offset,
                local_addr,
                static_cast<uint32_t>(length),
                local_mr,
                &transfer_us);
        } else {
            auto t0 = std::chrono::steady_clock::now();
            chunks = post_windowed_rdma(
                IBV_WR_RDMA_READ,
                offset,
                local_addr,
                length,
                chunk_bytes,
                local_mr,
                max_inflight
            );
            auto t1 = std::chrono::steady_clock::now();
            transfer_us = std::chrono::duration<double, std::micro>(
                t1 - t0).count();
        }
        debug_report(
            debug,
            "read_into_transfer_latency_us",
            offset,
            length,
            chunk_bytes,
            max_inflight,
            chunks,
            debug_t0,
            hits_before,
            misses_before);
        return transfer_us;
    }

    double read_into_transfer_latency_us_raw_poll(
                   uint64_t offset,
                   std::uintptr_t local_addr,
                   uint64_t length,
                   uint64_t chunk_bytes = 64ull * 1024 * 1024,
                   std::shared_ptr<void> owner = nullptr,
                   uint64_t register_length = 0,
                   uint32_t max_inflight = 16,
                   bool debug = false) {
        (void)max_inflight;
        (void)debug;
        if (length > total_len_ || offset > total_len_ - length) {
            throw std::runtime_error("read_into_transfer_latency_us_raw_poll out of range");
        }
        if (length > 0 && local_addr == 0) {
            throw std::runtime_error("read_into_transfer_latency_us_raw_poll got null local address");
        }
        if (chunk_bytes == 0) {
            throw std::runtime_error("read_into_transfer_latency_us_raw_poll chunk_bytes must be > 0");
        }
        if (length == 0) {
            return 0.0;
        }
        if (register_length == 0) {
            register_length = length;
        }
        if (register_length < length) {
            throw std::runtime_error(
                "read_into_transfer_latency_us_raw_poll register_length is smaller than length");
        }

        std::shared_ptr<rdmapp::local_mr> local_mr;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            local_mr = get_cached_local_mr(
                local_addr, register_length, std::move(owner));
        }

        const remote_mr_info* single_region =
            length <= chunk_bytes ? find_region_for_range(offset, length) : nullptr;
        if (single_region == nullptr ||
            length > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                "raw-poll latency path supports one in-region RDMA READ only");
        }

        return post_single_rdma_raw_poll(
            IBV_WR_RDMA_READ,
            *single_region,
            offset,
            local_addr,
            static_cast<uint32_t>(length),
            local_mr);
    }

    double read_into_transfer_latency_us_raw_poll_windowed(
                   uint64_t offset,
                   std::uintptr_t local_addr,
                   uint64_t length,
                   uint64_t chunk_bytes,
                   std::shared_ptr<void> owner = nullptr,
                   uint64_t register_length = 0,
                   uint32_t max_inflight = 16,
                   bool debug = false) {
        (void)debug;
        if (length > total_len_ || offset > total_len_ - length) {
            throw std::runtime_error(
                "read_into_transfer_latency_us_raw_poll_windowed out of range");
        }
        if (length > 0 && local_addr == 0) {
            throw std::runtime_error(
                "read_into_transfer_latency_us_raw_poll_windowed got null local address");
        }
        if (chunk_bytes == 0) {
            throw std::runtime_error(
                "read_into_transfer_latency_us_raw_poll_windowed chunk_bytes must be > 0");
        }
        if (length == 0) {
            return 0.0;
        }
        if (register_length == 0) {
            register_length = length;
        }
        if (register_length < length) {
            throw std::runtime_error(
                "read_into_transfer_latency_us_raw_poll_windowed register_length is smaller than length");
        }

        std::shared_ptr<rdmapp::local_mr> local_mr;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            local_mr = get_cached_local_mr(
                local_addr, register_length, std::move(owner));
        }

        auto t0 = std::chrono::steady_clock::now();
        post_windowed_rdma_raw_poll(
            IBV_WR_RDMA_READ,
            offset,
            local_addr,
            length,
            chunk_bytes,
            local_mr,
            max_inflight);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double read_repeated_into_transfer_latency_us_raw_poll(
                   uint64_t offset,
                   std::uintptr_t local_addr,
                   uint64_t length,
                   uint64_t chunk_bytes,
                   uint32_t repeat_count,
                   std::shared_ptr<void> owner = nullptr,
                   uint64_t register_length = 0,
                   uint32_t max_inflight = 16,
                   bool debug = false) {
        (void)debug;
        if (repeat_count == 0) {
            throw std::runtime_error(
                "read_repeated_into_transfer_latency_us_raw_poll repeat_count must be > 0");
        }
        if (length > total_len_ || offset > total_len_ - length) {
            throw std::runtime_error(
                "read_repeated_into_transfer_latency_us_raw_poll out of range");
        }
        if (length > 0 && local_addr == 0) {
            throw std::runtime_error(
                "read_repeated_into_transfer_latency_us_raw_poll got null local address");
        }
        if (chunk_bytes == 0) {
            throw std::runtime_error(
                "read_repeated_into_transfer_latency_us_raw_poll chunk_bytes must be > 0");
        }
        if (length == 0) {
            return 0.0;
        }
        if (repeat_count >
            std::numeric_limits<uint64_t>::max() / length) {
            throw std::runtime_error(
                "read_repeated_into_transfer_latency_us_raw_poll total local length overflows");
        }
        const uint64_t total_local_length =
            length * static_cast<uint64_t>(repeat_count);
        if (register_length == 0) {
            register_length = total_local_length;
        }
        if (register_length < total_local_length) {
            throw std::runtime_error(
                "read_repeated_into_transfer_latency_us_raw_poll register_length is smaller than total local length");
        }

        std::shared_ptr<rdmapp::local_mr> local_mr;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            local_mr = get_cached_local_mr(
                local_addr, register_length, std::move(owner));
        }

        auto t0 = std::chrono::steady_clock::now();
        post_repeated_rdma_raw_poll(
            IBV_WR_RDMA_READ,
            offset,
            local_addr,
            length,
            chunk_bytes,
            repeat_count,
            local_mr,
            max_inflight);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
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
                    uint64_t register_length = 0,
                    uint32_t max_inflight = 16,
                    bool debug = false) {
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

        uint64_t hits_before = 0;
        uint64_t misses_before = 0;
        std::shared_ptr<rdmapp::local_mr> local_mr;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            hits_before = mr_cache_hits_;
            misses_before = mr_cache_misses_;
            local_mr = get_cached_local_mr(
                local_addr, register_length, std::move(owner));
        }

        auto t0 = std::chrono::steady_clock::now();
        size_t chunks = 0;
        const remote_mr_info* single_region =
            length <= chunk_bytes ? find_region_for_range(offset, length) : nullptr;
        if (single_region != nullptr &&
            length <= std::numeric_limits<uint32_t>::max()) {
            chunks = post_single_rdma(
                IBV_WR_RDMA_WRITE,
                *single_region,
                offset,
                local_addr,
                static_cast<uint32_t>(length),
                local_mr);
        } else {
            chunks = post_windowed_rdma(
                IBV_WR_RDMA_WRITE,
                offset,
                local_addr,
                length,
                chunk_bytes,
                local_mr,
                max_inflight
            );
        }
        debug_report(
            debug,
            "write_from",
            offset,
            length,
            chunk_bytes,
            max_inflight,
            chunks,
            t0,
            hits_before,
            misses_before);
    }


private:
    std::shared_ptr<rdmapp::qp> qp_;
    std::shared_ptr<asio::io_context> io_ctx_;

    std::vector<remote_mr_info> regions_;
    uint64_t total_len_;

    std::mutex mutex_;
    std::mutex send_mutex_;
    std::mutex post_mutex_;
    std::condition_variable send_cv_;
    size_t outstanding_send_wrs_ = 0;

    static constexpr uint32_t kMaxSendWr = 128;
    static constexpr uint32_t kSendWrSlack = 1;
    static constexpr uint32_t kMaxReadInflight = 16;

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

    struct rdma_chunk {
        const remote_mr_info* region;
        uint64_t in_mr_offset;
        uint64_t local_offset;
        uint32_t bytes;
    };

    struct batch_state {
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        std::exception_ptr exception;
    };

    const remote_mr_info* find_region_for_range(uint64_t offset,
                                                uint64_t length) {
        uint64_t end = offset + length;
        for (auto &r : regions_) {
            if (offset >= r.prefix_len &&
                end <= r.prefix_len + r.length) {
                return &r;
            }
        }
        return nullptr;
    }

    size_t post_single_rdma(
        enum ibv_wr_opcode opcode,
        const remote_mr_info& region,
        uint64_t offset,
        std::uintptr_t local_addr,
        uint32_t length,
        std::shared_ptr<rdmapp::local_mr> local_mr,
        double* submit_to_complete_us = nullptr) {
        batch_state state;
        reserve_send_wrs(1);

        rdmapp::executor::callback_ptr callback =
            rdmapp::executor::make_callback(
                [&state](struct ibv_wc const &wc) {
                    {
                        std::lock_guard<std::mutex> lk(state.mutex);
                        if (wc.status != IBV_WC_SUCCESS) {
                            state.exception = std::make_exception_ptr(
                                std::runtime_error(
                                    std::string("single RDMA failed: ") +
                                    ibv_wc_status_str(wc.status)));
                        }
                        state.done = true;
                    }
                    state.cv.notify_one();
                });
        bool callback_owned_by_caller = true;

        auto submit_t0 = std::chrono::steady_clock::time_point{};
        try {
            struct ibv_sge sge = {};
            sge.addr = static_cast<uint64_t>(local_addr);
            sge.length = length;
            sge.lkey = local_mr->lkey();

            struct ibv_send_wr send_wr = {};
            struct ibv_send_wr *bad_send_wr = nullptr;
            send_wr.opcode = opcode;
            send_wr.next = nullptr;
            send_wr.num_sge = 1;
            send_wr.sg_list = &sge;
            send_wr.wr_id = reinterpret_cast<uint64_t>(callback);
            send_wr.send_flags = IBV_SEND_SIGNALED;
            send_wr.wr.rdma.remote_addr =
                static_cast<uint64_t>(region.addr + (offset - region.prefix_len));
            send_wr.wr.rdma.rkey = region.rkey;

            {
                std::lock_guard<std::mutex> post_lk(post_mutex_);
                submit_t0 = std::chrono::steady_clock::now();
                qp_->post_send(send_wr, bad_send_wr);
            }
            callback_owned_by_caller = false;

            std::unique_lock<std::mutex> lk(state.mutex);
            state.cv.wait(lk, [&state] { return state.done; });
            if (submit_to_complete_us != nullptr) {
                auto t1 = std::chrono::steady_clock::now();
                *submit_to_complete_us =
                    std::chrono::duration<double, std::micro>(
                        t1 - submit_t0).count();
            }
            if (state.exception) {
                std::rethrow_exception(state.exception);
            }
        } catch (...) {
            if (callback_owned_by_caller) {
                rdmapp::executor::destroy_callback(callback);
            }
            release_send_wrs(1);
            throw;
        }

        release_send_wrs(1);
        return 1;
    }

    double post_single_rdma_raw_poll(
        enum ibv_wr_opcode opcode,
        const remote_mr_info& region,
        uint64_t offset,
        std::uintptr_t local_addr,
        uint32_t length,
        std::shared_ptr<rdmapp::local_mr> local_mr) {
        struct ibv_sge sge = {};
        sge.addr = static_cast<uint64_t>(local_addr);
        sge.length = length;
        sge.lkey = local_mr->lkey();

        struct ibv_send_wr send_wr = {};
        struct ibv_send_wr *bad_send_wr = nullptr;
        send_wr.opcode = opcode;
        send_wr.next = nullptr;
        send_wr.num_sge = 1;
        send_wr.sg_list = &sge;
        send_wr.wr_id = 1;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        send_wr.wr.rdma.remote_addr =
            static_cast<uint64_t>(region.addr + (offset - region.prefix_len));
        send_wr.wr.rdma.rkey = region.rkey;

        struct ibv_wc wc = {};
        auto t0 = std::chrono::steady_clock::now();
        qp_->post_send(send_wr, bad_send_wr);
        while (!qp_->poll_send_cq(wc)) {
        }
        auto t1 = std::chrono::steady_clock::now();

        if (wc.status != IBV_WC_SUCCESS) {
            throw std::runtime_error(
                std::string("raw-poll RDMA failed: ") +
                ibv_wc_status_str(wc.status));
        }

        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    std::vector<rdma_chunk> make_chunks(
        uint64_t offset,
        uint64_t length,
        uint64_t chunk_bytes) {
        std::vector<rdma_chunk> chunks;
        chunks.reserve(static_cast<std::size_t>(
            length / chunk_bytes + (length % chunk_bytes == 0 ? 0 : 1)));
        uint64_t cur_off = offset;
        uint64_t remaining = length;
        uint64_t local_offset = 0;
        uint64_t max_wr_bytes = std::min<uint64_t>(
            chunk_bytes,
            std::numeric_limits<uint32_t>::max());

        while (remaining > 0) {
            const auto& r = find_region_by_offset(cur_off);
            uint64_t in_mr_offset = cur_off - r.prefix_len;
            uint64_t cap = r.length - in_mr_offset;
            uint64_t chunk = std::min<uint64_t>(
                std::min<uint64_t>(remaining, cap),
                max_wr_bytes);

            chunks.push_back(rdma_chunk{
                &r,
                in_mr_offset,
                local_offset,
                static_cast<uint32_t>(chunk),
            });

            cur_off += chunk;
            local_offset += chunk;
            remaining -= chunk;
        }

        return chunks;
    }

    uint32_t effective_window(enum ibv_wr_opcode opcode,
                              uint32_t max_inflight) const {
        uint32_t window = std::clamp<uint32_t>(
            max_inflight,
            1,
            kMaxSendWr - kSendWrSlack);
        if (opcode == IBV_WR_RDMA_READ) {
            window = std::min<uint32_t>(window, kMaxReadInflight);
        }
        return window;
    }

    void reserve_send_wrs(size_t wrs) {
        if (wrs == 0) {
            return;
        }
        const size_t usable = kMaxSendWr - kSendWrSlack;
        if (wrs > usable) {
            throw std::runtime_error("RDMA window exceeds QP send queue capacity");
        }

        std::unique_lock<std::mutex> lk(send_mutex_);
        send_cv_.wait(lk, [this, wrs, usable] {
            return outstanding_send_wrs_ + wrs <= usable;
        });
        outstanding_send_wrs_ += wrs;
    }

    void release_send_wrs(size_t wrs) {
        {
            std::lock_guard<std::mutex> lk(send_mutex_);
            if (wrs > outstanding_send_wrs_) {
                outstanding_send_wrs_ = 0;
            } else {
                outstanding_send_wrs_ -= wrs;
            }
        }
        send_cv_.notify_all();
    }

    size_t post_windowed_rdma(
        enum ibv_wr_opcode opcode,
        uint64_t offset,
        std::uintptr_t local_addr,
        uint64_t length,
        uint64_t chunk_bytes,
        std::shared_ptr<rdmapp::local_mr> local_mr,
        uint32_t max_inflight) {
        auto chunks = make_chunks(offset, length, chunk_bytes);
        if (chunks.empty()) {
            return 0;
        }

        uint32_t window = effective_window(opcode, max_inflight);

        for (size_t begin = 0; begin < chunks.size(); begin += window) {
            size_t end = std::min<size_t>(begin + window, chunks.size());
            size_t window_len = end - begin;
            auto state = std::make_shared<batch_state>();
            reserve_send_wrs(window_len);

            try {
                for (size_t i = begin; i < end; ++i) {
                    const auto& chunk = chunks[i];
                    const bool signaled = (i + 1 == end);

                    rdmapp::executor::callback_ptr callback = nullptr;
                    if (signaled) {
                        callback = rdmapp::executor::make_callback(
                            [state](struct ibv_wc const &wc) {
                                {
                                    std::lock_guard<std::mutex> lk(state->mutex);
                                    if (wc.status != IBV_WC_SUCCESS) {
                                        state->exception = std::make_exception_ptr(
                                            std::runtime_error(
                                                std::string("windowed RDMA failed: ") +
                                                ibv_wc_status_str(wc.status)));
                                    }
                                    state->done = true;
                                }
                                state->cv.notify_one();
                            });
                    }

                    struct ibv_sge sge = {};
                    sge.addr = static_cast<uint64_t>(local_addr + chunk.local_offset);
                    sge.length = chunk.bytes;
                    sge.lkey = local_mr->lkey();

                    struct ibv_send_wr send_wr = {};
                    struct ibv_send_wr *bad_send_wr = nullptr;
                    send_wr.opcode = opcode;
                    send_wr.next = nullptr;
                    send_wr.num_sge = 1;
                    send_wr.sg_list = &sge;
                    send_wr.wr_id = signaled ? reinterpret_cast<uint64_t>(callback) : 0;
                    send_wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
                    send_wr.wr.rdma.remote_addr =
                        static_cast<uint64_t>(chunk.region->addr + chunk.in_mr_offset);
                    send_wr.wr.rdma.rkey = chunk.region->rkey;

                    try {
                        std::lock_guard<std::mutex> post_lk(post_mutex_);
                        qp_->post_send(send_wr, bad_send_wr);
                    } catch (...) {
                        if (callback != nullptr) {
                            rdmapp::executor::destroy_callback(callback);
                        }
                        throw;
                    }
                }

                std::unique_lock<std::mutex> lk(state->mutex);
                state->cv.wait(lk, [&state] { return state->done; });
                if (state->exception) {
                    std::rethrow_exception(state->exception);
                }
            } catch (...) {
                release_send_wrs(window_len);
                throw;
            }
            release_send_wrs(window_len);
        }
        return chunks.size();
    }

    size_t post_windowed_rdma_raw_poll(
        enum ibv_wr_opcode opcode,
        uint64_t offset,
        std::uintptr_t local_addr,
        uint64_t length,
        uint64_t chunk_bytes,
        std::shared_ptr<rdmapp::local_mr> local_mr,
        uint32_t max_inflight) {
        auto chunks = make_chunks(offset, length, chunk_bytes);
        if (chunks.empty()) {
            return 0;
        }

        uint32_t window = effective_window(opcode, max_inflight);
        for (size_t begin = 0; begin < chunks.size(); begin += window) {
            size_t end = std::min<size_t>(begin + window, chunks.size());
            size_t window_len = end - begin;
            reserve_send_wrs(window_len);

            try {
                std::array<struct ibv_sge, kMaxSendWr> sges{};
                std::array<struct ibv_send_wr, kMaxSendWr> send_wrs{};
                for (size_t local_i = 0; local_i < window_len; ++local_i) {
                    const auto& chunk = chunks[begin + local_i];
                    const bool signaled = local_i + 1 == window_len;

                    auto& sge = sges[local_i];
                    sge = {};
                    sge.addr =
                        static_cast<uint64_t>(local_addr + chunk.local_offset);
                    sge.length = chunk.bytes;
                    sge.lkey = local_mr->lkey();

                    auto& send_wr = send_wrs[local_i];
                    send_wr = {};
                    send_wr.opcode = opcode;
                    send_wr.next = signaled ? nullptr : &send_wrs[local_i + 1];
                    send_wr.num_sge = 1;
                    send_wr.sg_list = &sge;
                    send_wr.wr_id = signaled ? 1 : 0;
                    send_wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
                    send_wr.wr.rdma.remote_addr =
                        static_cast<uint64_t>(
                            chunk.region->addr + chunk.in_mr_offset);
                    send_wr.wr.rdma.rkey = chunk.region->rkey;
                }

                struct ibv_send_wr* bad_send_wr = nullptr;
                qp_->post_send(send_wrs.front(), bad_send_wr);

                struct ibv_wc wc = {};
                while (!qp_->poll_send_cq(wc)) {
                }
                if (wc.status != IBV_WC_SUCCESS) {
                    throw std::runtime_error(
                        std::string("raw-poll windowed RDMA failed: ") +
                        ibv_wc_status_str(wc.status));
                }
            } catch (...) {
                release_send_wrs(window_len);
                throw;
            }
            release_send_wrs(window_len);
        }
        return chunks.size();
    }

    size_t post_repeated_rdma_raw_poll(
        enum ibv_wr_opcode opcode,
        uint64_t offset,
        std::uintptr_t local_addr,
        uint64_t length,
        uint64_t chunk_bytes,
        uint32_t repeat_count,
        std::shared_ptr<rdmapp::local_mr> local_mr,
        uint32_t max_inflight) {
        auto single_chunks = make_chunks(offset, length, chunk_bytes);
        if (single_chunks.empty()) {
            return 0;
        }

        std::vector<rdma_chunk> chunks;
        chunks.reserve(single_chunks.size() *
                       static_cast<std::size_t>(repeat_count));
        for (uint32_t repeat = 0; repeat < repeat_count; ++repeat) {
            const uint64_t local_base =
                static_cast<uint64_t>(repeat) * length;
            for (auto chunk : single_chunks) {
                chunk.local_offset += local_base;
                chunks.push_back(chunk);
            }
        }

        uint32_t window = effective_window(opcode, max_inflight);
        for (size_t begin = 0; begin < chunks.size(); begin += window) {
            size_t end = std::min<size_t>(begin + window, chunks.size());
            size_t window_len = end - begin;
            reserve_send_wrs(window_len);

            try {
                std::array<struct ibv_sge, kMaxSendWr> sges{};
                std::array<struct ibv_send_wr, kMaxSendWr> send_wrs{};
                for (size_t local_i = 0; local_i < window_len; ++local_i) {
                    const auto& chunk = chunks[begin + local_i];
                    const bool signaled = local_i + 1 == window_len;

                    auto& sge = sges[local_i];
                    sge = {};
                    sge.addr =
                        static_cast<uint64_t>(local_addr + chunk.local_offset);
                    sge.length = chunk.bytes;
                    sge.lkey = local_mr->lkey();

                    auto& send_wr = send_wrs[local_i];
                    send_wr = {};
                    send_wr.opcode = opcode;
                    send_wr.next = signaled ? nullptr : &send_wrs[local_i + 1];
                    send_wr.num_sge = 1;
                    send_wr.sg_list = &sge;
                    send_wr.wr_id = signaled ? 1 : 0;
                    send_wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
                    send_wr.wr.rdma.remote_addr =
                        static_cast<uint64_t>(
                            chunk.region->addr + chunk.in_mr_offset);
                    send_wr.wr.rdma.rkey = chunk.region->rkey;
                }

                struct ibv_send_wr* bad_send_wr = nullptr;
                qp_->post_send(send_wrs.front(), bad_send_wr);

                struct ibv_wc wc = {};
                while (!qp_->poll_send_cq(wc)) {
                }
                if (wc.status != IBV_WC_SUCCESS) {
                    throw std::runtime_error(
                        std::string("raw-poll repeated RDMA failed: ") +
                        ibv_wc_status_str(wc.status));
                }
            } catch (...) {
                release_send_wrs(window_len);
                throw;
            }
            release_send_wrs(window_len);
        }
        return chunks.size();
    }

    void debug_report(
        bool debug,
        const char* op,
        uint64_t offset,
        uint64_t length,
        uint64_t chunk_bytes,
        uint32_t max_inflight,
        size_t chunks,
        std::chrono::steady_clock::time_point t0,
        uint64_t hits_before,
        uint64_t misses_before) {
        if (!debug) {
            return;
        }
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        double gib = static_cast<double>(length) / (1024.0 * 1024.0 * 1024.0);
        double gibps = seconds > 0.0 ? gib / seconds : 0.0;
        uint64_t hits_after = 0;
        uint64_t misses_after = 0;
        size_t cache_size = 0;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            hits_after = mr_cache_hits_;
            misses_after = mr_cache_misses_;
            cache_size = mr_cache_.size();
        }
        std::cerr << "[RdmaSession][GDR] " << op
                  << " offset=" << offset
                  << " bytes=" << length
                  << " chunk_bytes=" << chunk_bytes
                  << " max_inflight=" << max_inflight
                  << " chunks=" << chunks
                  << " seconds=" << seconds
                  << " GiB/s=" << gibps
                  << " mr_cache_size=" << cache_size
                  << " mr_hits+=" << (hits_after - hits_before)
                  << " mr_misses+=" << (misses_after - misses_before)
                  << std::endl;
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
    explicit RdmaEndpoint(int ib_port = 1,
                          int device_num = 1,
                          uint32_t io_workers = 1,
                          size_t cq_batch_size = 16) {
        io_workers = std::max<uint32_t>(io_workers, 1);
        cq_batch_size = std::max<size_t>(cq_batch_size, 1);

        device_ = std::make_shared<rdmapp::device>(device_num, ib_port);
        pd_     = std::make_shared<rdmapp::pd>(device_);
        cq_     = std::make_shared<rdmapp::cq>(device_);

        io_ctx_   = std::make_shared<asio::io_context>(io_workers);
        executor_ = std::make_shared<rdmapp::executor>(io_ctx_);
        cq_poller_ =
            std::make_shared<rdmapp::cq_poller>(cq_, executor_, cq_batch_size);

        workers_.reserve(io_workers);
        for (uint32_t worker_id = 0; worker_id < io_workers; ++worker_id) {
            workers_.emplace_back([this]() {
                auto guard = asio::make_work_guard(*io_ctx_);
                io_ctx_->run();
            });
        }
    }

    ~RdmaEndpoint() {
        if (io_ctx_) {
            io_ctx_->stop();
        }
    }

    void stop_cq_poller() {
        cq_poller_.reset();
    }

    void stop_background_workers() {
        cq_poller_.reset();
        if (io_ctx_) {
            io_ctx_->stop();
        }
        workers_.clear();
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
    std::vector<std::jthread> workers_;
};
