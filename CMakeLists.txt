cmake_minimum_required(VERSION 3.22)
project(rdma_py CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(RDMAPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rdmapp" ABSOLUTE)
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
set(RDMAPP_BUILD_NORTTI OFF CACHE BOOL "" FORCE)
set(RDMAPP_BUILD_PIC ON CACHE BOOL "" FORCE)
add_subdirectory(${RDMAPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/rdmapp-build)

option(RDMAPY_BUILD_PYTHON "Build Python extension module" ON)

if(RDMAPY_BUILD_PYTHON)
# ===== 找 Python / pybind11 =====
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKEDIR}")
    list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
    message(STATUS "Using pybind11 from Python environment: ${PYBIND11_CMAKEDIR}")
endif()

find_package(pybind11 2.13 CONFIG REQUIRED)

# ===== 构建 python 模块 =====
pybind11_add_module(rdma_py
    src/binding.cpp
    src/rdma_session.hpp            # header-only不会编译，但可以写进去无所谓
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
    rdmapp_qp_helpers
)
endif()

# ===== 构建纯 C++ 延迟测试 =====
option(RDMAPY_BUILD_CXX_BENCHES "Build standalone C++ RDMA benchmark clients" ON)

if(RDMAPY_BUILD_CXX_BENCHES)
    find_package(CUDAToolkit QUIET)

    add_executable(rdma_prompt_read_latency
        examples/rdma_prompt_read_latency.cc
        src/rdma_session.hpp
    )

    target_compile_features(rdma_prompt_read_latency PRIVATE cxx_std_20)

    target_include_directories(rdma_prompt_read_latency
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RDMAPP_DIR}/examples/include
    )

    target_link_libraries(rdma_prompt_read_latency
        PRIVATE
        rdmapp_qp_helpers
    )

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDAToolkit found: enabling CUDA buffer mode for rdma_prompt_read_latency")
        target_compile_definitions(rdma_prompt_read_latency PRIVATE RDMAPY_HAS_CUDA=1)
        target_link_libraries(rdma_prompt_read_latency PRIVATE CUDA::cudart)
    else()
        message(STATUS "CUDAToolkit not found: rdma_prompt_read_latency will support --memory host only")
    endif()

    add_executable(rdma_read_matrix_latency
        examples/rdma_read_matrix_latency.cc
        src/rdma_read_bench_common.hpp
        src/rdma_session.hpp
    )

    target_compile_features(rdma_read_matrix_latency PRIVATE cxx_std_20)

    target_include_directories(rdma_read_matrix_latency
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RDMAPP_DIR}/examples/include
    )

    target_link_libraries(rdma_read_matrix_latency
        PRIVATE
        rdmapp_qp_helpers
    )

    add_executable(rdma_read_matrix_throughput
        examples/rdma_read_matrix_throughput.cc
        src/rdma_read_bench_common.hpp
        src/rdma_session.hpp
    )

    target_compile_features(rdma_read_matrix_throughput PRIVATE cxx_std_20)

    target_include_directories(rdma_read_matrix_throughput
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RDMAPP_DIR}/examples/include
    )

    target_link_libraries(rdma_read_matrix_throughput
        PRIVATE
        rdmapp_qp_helpers
    )

    add_executable(rdma_send_matrix_throughput
        examples/rdma_send_matrix_throughput.cc
        src/rdma_read_bench_common.hpp
    )

    target_compile_features(rdma_send_matrix_throughput PRIVATE cxx_std_20)

    target_include_directories(rdma_send_matrix_throughput
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RDMAPP_DIR}/examples/include
    )

    target_link_libraries(rdma_send_matrix_throughput
        PRIVATE
        rdmapp_qp_helpers
    )

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDAToolkit found: enabling CUDA buffer mode for RDMA read matrix benches")
        target_compile_definitions(rdma_read_matrix_latency PRIVATE RDMAPY_HAS_CUDA=1)
        target_compile_definitions(rdma_read_matrix_throughput PRIVATE RDMAPY_HAS_CUDA=1)
        target_link_libraries(rdma_read_matrix_latency PRIVATE CUDA::cudart)
        target_link_libraries(rdma_read_matrix_throughput PRIVATE CUDA::cudart)
    else()
        message(STATUS "CUDAToolkit not found: RDMA read matrix benches will support --memory host only")
    endif()

    add_executable(rdma_server
        server/server.cc
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

    add_executable(rdma_promptdb_server
        examples/rdma_promptdb_server.cc
        src/promptdb_lsh.hpp
    )

    target_compile_features(rdma_promptdb_server PRIVATE cxx_std_20)

    target_include_directories(rdma_promptdb_server
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RDMAPP_DIR}/examples/include
    )

    target_link_libraries(rdma_promptdb_server
        PRIVATE
        rdmapp_qp_helpers
    )

    add_executable(rdma_promptdb_find_latency
        examples/rdma_promptdb_find_latency.cc
        src/promptdb_lsh.hpp
    )

    target_compile_features(rdma_promptdb_find_latency PRIVATE cxx_std_20)

    target_include_directories(rdma_promptdb_find_latency
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${RDMAPP_DIR}/examples/include
    )

    target_link_libraries(rdma_promptdb_find_latency
        PRIVATE
        rdmapp_qp_helpers
    )
endif()
