cmake_minimum_required(VERSION 3.22)
project(rdma_kv_server CXX)

set(CMAKE_CXX_STANDARD 20)

set(RDMAPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rdmapp")
set(RDMAPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_EXAMPLES_PYBIND OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_PIC ON CACHE BOOL "" FORCE)
add_subdirectory(${RDMAPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/rdmapp-build)

add_executable(rdma_server
    server.cc
    ${RDMAPP_DIR}/examples/qp_acceptor.cc
    ${RDMAPP_DIR}/examples/qp_connector.cc
    ${RDMAPP_DIR}/examples/qp_transmission.cc
)

target_compile_features(rdma_server PRIVATE cxx_std_20)

target_include_directories(rdma_server
    PRIVATE
    ${RDMAPP_DIR}/examples/include
)

target_link_libraries(rdma_server
    PRIVATE
    rdmapp
)
