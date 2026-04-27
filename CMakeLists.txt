cmake_minimum_required(VERSION 3.22)
project(rdma_py CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RDMAPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rdmapp")
set(RDMAPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_EXAMPLES_PYBIND OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_NORTTI OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_PIC ON CACHE BOOL "" FORCE)
add_subdirectory(${RDMAPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/rdmapp-build)

# ===== 找 pybind11 =====
find_package(pybind11 REQUIRED)

# ===== 构建 python 模块 =====
pybind11_add_module(rdma_py
    src/binding.cpp
    src/rdma_session.hpp            # header-only不会编译，但可以写进去无所谓

    # QP TCP handshake helpers from local rdmapp.
    ${RDMAPP_DIR}/examples/qp_acceptor.cc
    ${RDMAPP_DIR}/examples/qp_connector.cc
    ${RDMAPP_DIR}/examples/qp_transmission.cc
)

target_compile_features(rdma_py PRIVATE cxx_std_20)

target_include_directories(rdma_py
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RDMAPP_DIR}/examples/include
)

# ===== 链接 rdmapp 和系统库 =====
target_link_libraries(rdma_py
    PRIVATE
    rdmapp
)
