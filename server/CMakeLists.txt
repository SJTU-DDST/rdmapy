cmake_minimum_required(VERSION 3.14)
project(rdma_kv_server CXX)

set(CMAKE_CXX_STANDARD 20)

set(RDMAPP_DIR "/home/xingda/rdmapp")

include_directories(
    ${RDMAPP_DIR}/include
    ${RDMAPP_DIR}/examples/include
    ${RDMAPP_DIR}/build/_deps/asio-src/asio/include
    ${RDMAPP_DIR}/build/_deps/spdlog-src/include
    ${RDMAPP_DIR}/build/_deps/fmt-src/include
)

add_executable(rdma_server
    server.cc
    ${RDMAPP_DIR}/examples/qp_acceptor.cc
    ${RDMAPP_DIR}/examples/qp_connector.cc
    ${RDMAPP_DIR}/examples/qp_transmission.cc
)

target_link_directories(rdma_server PRIVATE
    ${RDMAPP_DIR}/build
)

target_link_libraries(rdma_server
    rdmapp
    ibverbs
    pthread
)
