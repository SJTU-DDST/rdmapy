# rdmapy

`rdmapy` 是一个基于 `pybind11` 的 Python RDMA 绑定，底层使用项目内置的
`rdmapp`。编译后会生成 Python 扩展模块 `rdma_py`，可以在 Python 里连接
RDMA KV server，并对远端注册内存做 read/write。

当前主要能力：

- CPU bytes 读写：
  - `session.read(offset, length) -> bytes`
  - `session.write(offset, bytes_payload)`
- GPUDirect RDMA CUDA tensor 读写：
  - `session.read_into_tensor(...)`
  - `session.write_from_tensor(...)`
- 本地 MR cache：
  - `session.mr_cache_stats()`
  - `session.set_mr_cache_limit(max_entries)`
  - `session.clear_mr_cache()`

项目里有两个 CMake 构建入口：

- `CMakeLists.txt`：构建 Python 扩展模块 `rdma_py`
- `server/CMakeLists.txt`：构建独立的 RDMA KV server 可执行文件

下面的命令默认都在 `rdmapy/` 目录下执行。

## 环境依赖

该项目面向 Linux RDMA 环境。

编译依赖：

- CMake 3.22+
- 支持 C++20 的编译器，例如 GCC 10+
- Python 3 development headers
- 当前 Python 环境中的 `pybind11 >= 2.13`
- `rdma-core` / `libibverbs` 开发库

运行依赖：

- 可用的 InfiniBand/RoCE 网卡和 RDMA 驱动
- 正确的 RDMA device 和 IB port
- 如果使用 GPUDirect RDMA tensor API，还需要 CUDA、支持 CUDA 的 PyTorch，
  以及 `nvidia-peermem` 或匹配的 peer-memory 模块

Ubuntu/Debian 上可以参考：

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ python3-dev rdma-core libibverbs-dev
python3 -m pip install "pybind11>=2.13"
```

## 编译 Python 模块

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

编译产物会在 `build/` 目录下，例如：

```text
build/rdma_py*.so
```

如果不安装到系统环境，可以直接把 `build/` 加到 `PYTHONPATH`：

```bash
export PYTHONPATH="$PWD/build:$PYTHONPATH"
python3 -c "import rdma_py; print(rdma_py)"
```

仓库里的 `test_connect.py` 和 `test_rw.py` 也会把 `./build` 加进
`sys.path`，所以在 `rdmapy/` 下编译完成后可以直接运行。

## 编译 RDMA Server

顶层的 `cmake --build build` 只构建 Python 模块，不会构建 server。
server 需要单独编译：

```bash
cmake -S server -B server/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build server/build -j
```

生成的可执行文件：

```text
server/build/rdma_server
```

启动 server：

```bash
./server/build/rdma_server --port 2333 --ib-device 0 --ib-port 1
```

默认参数：

```text
--port 2333 --ib-device 0 --ib-port 1 --max-connections 64
```

查看完整参数：

```bash
./server/build/rdma_server --help
```

## 快速测试

先在暴露远端内存的机器上启动 server：

```bash
./server/build/rdma_server --port 2333 --ib-device 0 --ib-port 1
```

然后在 client 侧测试连接：

```bash
export PYTHONPATH="$PWD/build:$PYTHONPATH"
python3 test_connect.py
```

测试 CPU bytes 写入：

```bash
python3 test_rw.py
```

如果 server 不在本机，需要先改 `test_connect.py` / `test_rw.py` 里的 host。

## Python 使用示例

```python
import rdma_py

endpoint = rdma_py.Endpoint(ib_port=1, device_num=0)
session = endpoint.connect("127.0.0.1", 2333)

session.write(0, b"hello")
data = session.read(0, 5)
print(data)
```

`Endpoint(ib_port, device_num)` 里的 `device_num` 是本地 RDMA 设备序号。
不同机器上的序号可能不同，可以用下面的命令检查：

```bash
ibv_devices
ibv_devinfo
```

## GPUDirect RDMA

CUDA tensor API 要求 tensor 是 CUDA 上的 contiguous tensor：

```python
import rdma_py
import torch

endpoint = rdma_py.Endpoint(ib_port=1, device_num=0)
session = endpoint.connect("server-ip", 2333)

src = torch.arange(1024 * 1024, dtype=torch.int32, device="cuda:0").contiguous()
dst = torch.empty_like(src)

torch.cuda.synchronize()
session.write_from_tensor(0, src, chunk_bytes=64 * 1024 * 1024, max_inflight=16)
session.read_into_tensor(0, dst, chunk_bytes=64 * 1024 * 1024, max_inflight=16)
torch.cuda.synchronize()

assert torch.equal(src, dst)
```

GDR smoke test：

```bash
python3 examples/gdr_tensor_smoke.py \
  --host <server-ip> \
  --port 2333 \
  --device cuda:0 \
  --ib-device 0 \
  --ib-port 1 \
  --debug
```

GDR 带宽测试：

```bash
python3 examples/gdr_tensor_bench.py \
  --host <server-ip> \
  --port 2333 \
  --device cuda:0 \
  --ib-device 0 \
  --ib-port 1 \
  --bytes 512m \
  --chunk-bytes 4m,16m,64m \
  --max-inflight 1,4,8,16
```

更详细的 GDR 说明见 [`GDR_USAGE.md`](GDR_USAGE.md)。

## 常见问题

`cmake` 找不到 `pybind11` 时，确认 `pybind11` 安装在 CMake 使用的同一个
Python 环境里：

```bash
python3 -m pybind11 --cmakedir
```

如果 `import rdma_py` 失败，通常是 `PYTHONPATH` 没有包含 `rdmapy/build`：

```bash
export PYTHONPATH="/path/to/rdmapy/build:$PYTHONPATH"
```

CMake configure 阶段可能会通过 `FetchContent` 拉取 `asio`、`fmt`、`spdlog`
等 `rdmapp` 依赖；离线环境下需要提前准备这些依赖或 CMake 缓存。
