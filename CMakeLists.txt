cmake_minimum_required(VERSION 3.18)
project(rdma_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 让你自己改：指向你的 rdmapp 源码根
set(RDMAPP_ROOT "${CMAKE_SOURCE_DIR}/../rdmapp")

# 找 pybind11（系统有 apt 包的话：sudo apt install pybind11-dev）
find_package(pybind11 REQUIRED)

# 头文件：主库 + examples
include_directories(
  "${RDMAPP_ROOT}/include"
  "${RDMAPP_ROOT}/examples/include"
)

# 目标：生成 Python 模块 rdma_py*.so
pybind11_add_module(rdma_py
  src/rdma_bind.cpp
)

# 链接 rdmapp 两个库 + verbs + pthread
# 优先用静态库，省去 LD_LIBRARY_PATH
if (EXISTS "${RDMAPP_ROOT}/build/librdmapp.a")
  target_link_libraries(rdma_py PRIVATE
    "${RDMAPP_ROOT}/build/librdmapp.a"
    "${RDMAPP_ROOT}/build/librdmapp_examples.a"
    ibverbs pthread
  )
else()
  # 若只有 .so（不推荐），你还要设置 RPATH 或运行时 LD_LIBRARY_PATH
  link_directories("${RDMAPP_ROOT}/build")
  target_link_libraries(rdma_py PRIVATE rdmapp rdmapp_examples ibverbs pthread)
endif()
