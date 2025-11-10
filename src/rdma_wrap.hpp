#pragma once
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

#include <rdmapp/rdmapp.h>
#include <rdmapp/srq.h>
#include "acceptor.h"
#include "connector.h"
#include "socket/event_loop.h"

namespace pywrap {

class Session {
public:
  Session(int dev_index=0, int port=1) {
    device_ = std::make_shared<rdmapp::device>(dev_index, port);
    pd_     = std::make_shared<rdmapp::pd>(device_);
    cq_     = std::make_shared<rdmapp::cq>(device_);

    poller_ = std::make_shared<rdmapp::cq_poller>(cq_);

    // ✅ 再建 loop，并启动
    loop_   = rdmapp::socket::event_loop::new_loop();
    looper_ = std::thread([this]{ loop_->loop(); });
  }

  ~Session() {
    if (loop_) loop_->close();
    if (looper_.joinable()) looper_.join();
  }

  std::shared_ptr<rdmapp::pd> pd() const { return pd_; }
  std::shared_ptr<rdmapp::cq> cq() const { return cq_; }
  std::shared_ptr<rdmapp::socket::event_loop> loop() const { return loop_; }

private:
  std::shared_ptr<rdmapp::device> device_;
  std::shared_ptr<rdmapp::pd> pd_;
  std::shared_ptr<rdmapp::cq> cq_;
  std::shared_ptr<rdmapp::cq_poller> poller_;
  std::shared_ptr<rdmapp::socket::event_loop> loop_;
  std::thread looper_;
};

class Endpoint {
public:
  Endpoint() = default;  // ✅ 新增：允许默认构造
  Endpoint(std::shared_ptr<rdmapp::qp> qp) : qp_(std::move(qp)) {}
  virtual ~Endpoint() = default;

  // ---- send_bytes ----
  void send_bytes(const std::string& bytes) {
    auto buf = std::make_shared<std::vector<char>>(bytes.begin(), bytes.end());
    auto t = [this, buf]() -> rdmapp::task<void> {
      fprintf(stderr, "[send_bytes] post_send %zu\n", buf->size());
      co_await qp_->send(buf->data(), buf->size());
      fprintf(stderr, "[send_bytes] send_done\n");
      co_return;
    }();
    t.get_future().get();
  }

  // ---- recv_bytes ----
  std::string recv_bytes(size_t need) {
    auto out = std::make_shared<std::vector<char>>(need);
    size_t got = 0;
    auto t = [this, out, need, &got]() -> rdmapp::task<void> {
      fprintf(stderr, "[recv_bytes] post_recv %zu\n", need);
      while (got < need) {
        auto [n, imm] = co_await qp_->recv(out->data() + got, need - got);
        got += n;
        fprintf(stderr, "[recv_bytes] chunk n=%zu total=%zu imm=%s\n",
                n, got, imm ? "yes" : "no");
      }
      co_return;
    }();
    t.get_future().get();
    return std::string(out->data(), out->data() + got);
  }

protected:
  std::shared_ptr<rdmapp::qp> qp_;
};


class Server : public Endpoint {
public:
  explicit Server(std::shared_ptr<Session> sess, uint16_t port)
  : sess_(sess)  // ✅ 确保 Session 一直活着
  {
    // // SRQ 可空
    // std::shared_ptr<rdmapp::srq> srq;
    // // 采用 “单 CQ + 空 SRQ” 的重载
    // acceptor_ = std::make_unique<rdmapp::acceptor>(
    //   sess_->loop(), port, sess_->pd(), sess_->cq(), srq);
    // acceptor_ = std::make_unique<rdmapp::acceptor>(sess_->loop(), port, sess_->pd(), sess_->cq());
    // 如果将来想用 SRQ，这里可改为 new rdmapp::srq(sess_->pd())
    std::shared_ptr<rdmapp::srq> srq = nullptr;

    if (srq) {
      acceptor_ = std::make_unique<rdmapp::acceptor>(
          sess_->loop(), port, sess_->pd(), sess_->cq(), srq);
    } else {
      // ✅ 默认走不带 SRQ 的安全构造
      acceptor_ = std::make_unique<rdmapp::acceptor>(
          sess_->loop(), port, sess_->pd(), sess_->cq());
    }
  }

  // 阻塞接受一次，返回后持有一个 QP
  void accept_once() {
    auto t = [&]() -> rdmapp::task<void> {
      qp_ = co_await acceptor_->accept();
      co_return;
    }();
    t.get_future().get();
  }

  // === 下发一个 MR 给对端（read/write/atomics 用） ===
  // 返回序列化后的 remote_mr 元数据（发给 client）
  std::string expose_buffer_as_mr(void* addr, size_t len) {
    auto lmr = std::make_shared<rdmapp::local_mr>(sess_->pd()->reg_mr(addr, len));
    auto s   = lmr->serialize();
    // 记录，方便等会儿 recv 绑定 MR（WRITE with imm 后我们能知道并打印）
    last_local_mr_ = lmr;
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
  }

  // 等待对端 write_with_imm 到 last_local_mr_，返回是否收到 imm
  bool wait_write_imm() {
    auto t = [&]() -> rdmapp::task<void> {
      auto p = co_await qp_->recv(last_local_mr_);
      has_imm_ = p.second.has_value();
      co_return;
    }();
    t.get_future().get();
    return has_imm_;
  }

  // 等待一次 FAA/CAS 完成通知（同样复用 last_local_mr_ 做接收同步）
  bool wait_atomic_done() { return wait_write_imm(); }

private:
  std::shared_ptr<Session> sess_;
  std::unique_ptr<rdmapp::acceptor> acceptor_;
  std::shared_ptr<rdmapp::local_mr> last_local_mr_;
  bool has_imm_{false};
};

class Client : public Endpoint {
public:
  Client(std::shared_ptr<Session> sess, const std::string& host, uint16_t port)
  : sess_(sess)
  {
    // std::shared_ptr<rdmapp::srq> srq;
    // connector_ = std::make_unique<rdmapp::connector>(
    //   sess_->loop(), host, port, sess_->pd(), sess_->cq(), srq);
    // auto t = [&]() -> rdmapp::task<void> {
    //   qp_ = co_await connector_->connect();
    //   co_return;
    // }();
    std::shared_ptr<rdmapp::srq> srq = nullptr;

    if (srq) {
      connector_ = std::make_unique<rdmapp::connector>(
          sess_->loop(), host, port, sess_->pd(), sess_->cq(), srq);
    } else {
      // ✅ 安全构造
      connector_ = std::make_unique<rdmapp::connector>(
          sess_->loop(), host, port, sess_->pd(), sess_->cq());
    }

    auto t = [&]() -> rdmapp::task<void> {
      qp_ = co_await connector_->connect();
      co_return;
    }();
    t.get_future().get();
  }

  // 接收一条 MR 元数据，反序列化
  void recv_remote_mr_meta() {
    char meta[rdmapp::remote_mr::kSerializedSize];
    auto t = [&]() -> rdmapp::task<void> {
      co_await qp_->recv(meta, sizeof(meta));
      co_return;
    }();
    t.get_future().get();
    remote_mr_ = rdmapp::remote_mr::deserialize(meta);
  }

  // RDMA Read 远端 MR 到本地缓冲（返回读取字节数）
  size_t read_from_remote(void* dst, size_t bytes) {
    size_t n = 0;
    auto t = [&]() -> rdmapp::task<void> {
      n = co_await qp_->read(remote_mr_, dst, bytes);
      co_return;
    }();
    t.get_future().get();
    return n;
  }

  // RDMA Write with imm 把本地缓冲写回远端（imm 可用于对端确认）
  void write_to_remote_with_imm(const void* src, size_t bytes, uint32_t imm) {
    auto t = [&]() -> rdmapp::task<void> {
      co_await qp_->write_with_imm(remote_mr_, const_cast<void*>(static_cast<const void*>(src)), bytes, imm);
      co_return;
    }();
    t.get_future().get();
  }

  // 原子操作：FAA / CAS
  // 前置：服务端会先下发 counter 的 MR（和上面的 data MR 一样，再 recv 一次拿到它）
  void recv_counter_mr_meta() {
    char meta[rdmapp::remote_mr::kSerializedSize];
    auto t = [&]() -> rdmapp::task<void> {
      co_await qp_->recv(meta, sizeof(meta));
      co_return;
    }();
    t.get_future().get();
    counter_mr_ = rdmapp::remote_mr::deserialize(meta);
  }

  uint64_t fetch_and_add(uint64_t addend) {
    uint64_t old = 0;
    auto t = [&]() -> rdmapp::task<void> {
      co_await qp_->fetch_and_add(counter_mr_, &old, sizeof(old), addend);
      co_return;
    }();
    t.get_future().get();
    return old;
  }

  uint64_t compare_and_swap(uint64_t expected, uint64_t desired) {
    uint64_t old = 0;
    auto t = [&]() -> rdmapp::task<void> {
      co_await qp_->compare_and_swap(counter_mr_, &old, sizeof(old), expected, desired);
      co_return;
    }();
    t.get_future().get();
    return old;
  }

private:
  std::shared_ptr<Session> sess_;
  std::unique_ptr<rdmapp::connector> connector_;
  rdmapp::remote_mr remote_mr_{};
  rdmapp::remote_mr counter_mr_{};
};

} // namespace pywrap
