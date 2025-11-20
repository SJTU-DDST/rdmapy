cmake_minimum_required(VERSION 3.14)
project(rdma_py CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== rdmapp 路径（你给过的结构）=====
set(RDMAPP_DIR "/home/xingda/rdmapp")

# ===== include 路径 =====
include_directories(
    ${RDMAPP_DIR}/include
    ${RDMAPP_DIR}/examples/include
    ${RDMAPP_DIR}/build/_deps/asio-src/asio/include
    ${RDMAPP_DIR}/build/_deps/spdlog-src/include
    ${RDMAPP_DIR}/build/_deps/fmt-src/include

    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ===== 找 pybind11 =====
find_package(pybind11 REQUIRED)

# ===== 构建 python 模块 =====
pybind11_add_module(rdma_py
    src/binding.cpp
    src/rdma_session.hpp            # header-only不会编译，但可以写进去无所谓

    # ❗ 必须加入你们 rdmapp/examples 下的实现
    ${RDMAPP_DIR}/examples/qp_acceptor.cc
    ${RDMAPP_DIR}/examples/qp_connector.cc
    ${RDMAPP_DIR}/examples/qp_transmission.cc
)

# ===== 链接 rdmapp 和系统库 =====
target_link_libraries(rdma_py
    PRIVATE
    rdmapp
    ibverbs
    pthread
)

# ===== 让 linker 能找到 librdmapp.a =====
target_link_directories(rdma_py PRIVATE
    ${RDMAPP_DIR}/build
)
