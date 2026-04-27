cmake_minimum_required(VERSION 3.22)
project(rdma_kv_server CXX)

set(CMAKE_CXX_STANDARD 20)

get_filename_component(RDMAPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rdmapp" ABSOLUTE)
message(STATUS "Using bundled rdmapp: ${RDMAPP_DIR}")
if(NOT EXISTS "${RDMAPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Bundled rdmapp is missing. Expected: ${RDMAPP_DIR}")
endif()
foreach(RDMAPP_EXAMPLE_SOURCE
        qp_acceptor.cc
        qp_connector.cc
        qp_transmission.cc)
    if(NOT EXISTS "${RDMAPP_DIR}/examples/${RDMAPP_EXAMPLE_SOURCE}")
        message(FATAL_ERROR
            "Bundled rdmapp example source is missing: "
            "${RDMAPP_DIR}/examples/${RDMAPP_EXAMPLE_SOURCE}")
    endif()
endforeach()

set(RDMAPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_EXAMPLES_PYBIND OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_PIC ON CACHE BOOL "" FORCE)
add_subdirectory(${RDMAPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/rdmapp-build)

add_executable(rdma_server
    server.cc
)

target_compile_features(rdma_server PRIVATE cxx_std_20)

target_include_directories(rdma_server
    PRIVATE
    ${RDMAPP_DIR}/examples/include
)

target_link_libraries(rdma_server
    PRIVATE
    rdmapp_qp_helpers
)
