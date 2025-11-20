#include "qp_connector.h"
#include <asio/co_spawn.hpp>
#include <asio/use_future.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <span>
#include <spdlog/spdlog.h>
#include <string>

#include <rdmapp/rdmapp.h>
#include <rdmapp/executor.h>
#include <rdmapp/mr.h>

using namespace std::literals::chrono_literals;

constexpr std::string_view msg  = "hello";
constexpr std::string_view resp = "world";

asio::awaitable<void> client(std::shared_ptr<rdmapp::qp_connector> connector) {
  auto qp = co_await connector->connect();
  std::string buffer;
  buffer.resize(msg.size());
  auto recv_buffer = std::as_writable_bytes(std::span(buffer));

  // Send/Recv
  auto [n, _] = co_await qp->recv(recv_buffer);
  spdlog::info("received {} bytes from server: {}", n, buffer);
  buffer = std::string(resp);
  auto send_buffer = std::as_bytes(std::span(buffer));
  co_await qp->send(send_buffer);
  spdlog::info("sent to server: {}", buffer);

  // Read/Write
  char remote_mr_serialized[rdmapp::remote_mr::kSerializedSize];
  auto remote_mr_serialized_data =
      std::as_writable_bytes(std::span(remote_mr_serialized));
  co_await qp->recv(remote_mr_serialized_data);
  auto remote_mr = rdmapp::remote_mr::deserialize(remote_mr_serialized);
  spdlog::info("received mr addr={} length={} rkey={} from server",
               remote_mr.addr(), remote_mr.length(), remote_mr.rkey());

  buffer.resize(msg.size());
  recv_buffer = std::as_writable_bytes(std::span(buffer));
  n = co_await qp->read(remote_mr, recv_buffer);
  spdlog::info("read {} bytes from server: {}", n, buffer);
  buffer = std::string(resp);
  send_buffer = std::as_bytes(std::span(buffer));
  co_await qp->write_with_imm(remote_mr, send_buffer, 1);

  // Atomic Fetch-and-Add / Compare-and-Swap
  char counter_mr_serialized[rdmapp::remote_mr::kSerializedSize];
  recv_buffer = std::as_writable_bytes(std::span(counter_mr_serialized));
  co_await qp->recv(recv_buffer);
  auto counter_mr = rdmapp::remote_mr::deserialize(counter_mr_serialized);
  spdlog::info("received mr addr={} length={} rkey={} from server",
               counter_mr.addr(), counter_mr.length(), counter_mr.rkey());

  uint64_t counter = 0;
  auto cnt_buffer = std::as_writable_bytes(std::span(&counter, 1));
  co_await qp->fetch_and_add(counter_mr, cnt_buffer, 1);
  spdlog::info("fetched and added from server: {}", counter);
  co_await qp->write_with_imm(counter_mr, cnt_buffer, 1);
  spdlog::info("written with imm: buffer={} imm={}", counter, 1);
  co_await qp->compare_and_swap(counter_mr, cnt_buffer, 43, 4422);
  spdlog::info("compared and swapped from server: {}", counter);
  co_await qp->write_with_imm(counter_mr, cnt_buffer, 1);

  co_return;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " [server_ip] [port]" << std::endl;
    return -1;
  }

  spdlog::set_level(spdlog::level::debug);

  auto device = std::make_shared<rdmapp::device>(0, 1);
  auto pd     = std::make_shared<rdmapp::pd>(device);
  auto cq     = std::make_shared<rdmapp::cq>(device);
  auto io_ctx = std::make_shared<asio::io_context>(4);
  auto executor  = std::make_shared<rdmapp::executor>(io_ctx);
  auto cq_poller = std::make_shared<rdmapp::cq_poller>(cq, executor);

  auto connector = std::make_shared<rdmapp::qp_connector>(
      argv[1], static_cast<uint16_t>(std::stoi(argv[2])), pd, cq
  );

  std::jthread _([io_ctx]() { io_ctx->run(); });

  auto fut = asio::co_spawn(*io_ctx, client(connector), asio::use_future);
  fut.get();

  spdlog::info("client exit after communicated with {}:{}", argv[1], argv[2]);
  return 0;
}
