#include "qp_acceptor.h"
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/this_coro.hpp>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include <rdmapp/rdmapp.h>
#include <rdmapp/executor.h>
#include <rdmapp/mr.h>

// 为了让 QP 和缓冲区不被析构，简单放到全局容器里
static std::vector<std::shared_ptr<rdmapp::qp>> g_qps;
static std::vector<std::shared_ptr<std::vector<std::uint8_t>>> g_buffers;
static std::vector<std::shared_ptr<rdmapp::local_mr>> g_mrs;

constexpr std::size_t KV_BUFFER_SIZE = 8ull << 30;

asio::awaitable<void> handle_qp(std::shared_ptr<rdmapp::qp> qp) {
  spdlog::info("handling qp (KV server)");

  // 1. 为这个 QP 分配 KV buffer，并注册 MR
  auto kv_buf =
      std::make_shared<std::vector<std::uint8_t>>(KV_BUFFER_SIZE);
  auto local_mr = std::make_shared<rdmapp::local_mr>(
      qp->pd_ptr()->reg_mr(kv_buf->data(), kv_buf->size()));

  // 2. 序列化这个 MR，send 给客户端
  auto serialized = local_mr->serialize();
  auto bytes = std::as_bytes(std::span(serialized));
  co_await qp->send(bytes);

  spdlog::info("sent MR to client: addr={} length={} rkey={}",
               local_mr->addr(), local_mr->length(), local_mr->rkey());

  // 3. 把资源挂到全局，保证它们的生命周期 >= Python 读写
  g_qps.push_back(qp);
  g_buffers.push_back(kv_buf);
  g_mrs.push_back(local_mr);
  
  co_return;
}

asio::awaitable<void> server(std::shared_ptr<rdmapp::qp_acceptor> acceptor) {
  while (true) {
    auto qp = co_await acceptor->accept();
    asio::co_spawn(co_await asio::this_coro::executor,
                   handle_qp(qp),
                   asio::detached);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " [port]" << std::endl;
    return -1;
  }

  spdlog::set_level(spdlog::level::debug);

  auto device = std::make_shared<rdmapp::device>(0, 1);
  auto pd     = std::make_shared<rdmapp::pd>(device);
  auto cq     = std::make_shared<rdmapp::cq>(device);
  auto io_ctx = std::make_shared<asio::io_context>(4);
  auto executor  = std::make_shared<rdmapp::executor>(io_ctx);
  auto cq_poller = std::make_shared<rdmapp::cq_poller>(cq, executor);

  uint16_t port = static_cast<uint16_t>(std::stoi(argv[1]));
  auto acceptor = std::make_shared<rdmapp::qp_acceptor>(io_ctx, port, pd, cq);

  auto work_guard = asio::make_work_guard(*io_ctx);
  asio::co_spawn(*io_ctx, server(acceptor), asio::detached);
  io_ctx->run();

  return 0;
}
