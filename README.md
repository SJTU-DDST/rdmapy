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

## C++ Prompt READ 延迟测试

顶层 CMake 现在也会构建一个不经过 Python/PyTorch 调度循环的 C++ 延迟测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j --target rdma_prompt_read_latency
```

如果这台机器只需要 C++ 测试、不需要 Python extension，可以跳过
Python/pybind11：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DRDMAPY_BUILD_PYTHON=OFF
cmake --build build -j --target rdma_prompt_read_latency
```

CUDA Toolkit 可被 CMake 找到时，测试程序支持 `--memory cuda`，直接用
`cudaMalloc` 出来的本地 buffer 做 GPUDirect RDMA READ；否则仍可用
`--memory host` 做 host buffer 延迟测试。

示例：

```bash
./build/rdma_prompt_read_latency \
  --host <server-ip> \
  --port 2333 \
  --ib-device 0 \
  --ib-port 1 \
  --memory cuda \
  --cuda-device 0 \
  --prompt-bytes 128 \
  --min-threads 1 \
  --max-threads 16 \
  --repeats 5000 \
  --warmup 500 \
  --trials 5 \
  --cpu-list 0-15
```

注意：bash 续行符 `\` 必须是这一行最后一个字符，后面不能有空格。
如果写成 `\ `，shell 会把后面的空格当作普通参数，并把下一行的
`--port` 当成新命令执行。

输出固定为 `threads,avg_us`。每个 `threads=N` 的每个 trial 都会重新
创建 N 个 endpoint/CQ/QP，然后执行 `warmup + repeats` 轮 READ，默认
每轮用 `--round-sync` 让 N 个 worker 同时各发起一次 READ，最终把
所有 trial 的 READ 延迟取平均。默认 `--completion-mode raw-poll`，连接
和预热后会关闭后台 CQ poller，正式计时阶段由发起 READ 的 worker 线程
直接 `post_send` 并 busy-poll 自己的 CQ；计时范围是 `post_send` 到
poll 到 completion 的耗时。可以用 `--completion-mode callback` 对比旧的
callback/executor/condition_variable 路径开销。

raw-poll 对 CPU 调度非常敏感。`--cpu-list` 会把第 N 个 worker 固定到列表
中的第 N 个 CPU，例如 `--cpu-list 0-7,16-23`。实际测试时应优先选择和
RDMA 网卡、GPU 在同一个 NUMA node 上的物理核，避免使用 SMT sibling。
也可以用 `--auto-cpu-list --cpu-candidates ...` 先在候选 CPU 上逐个测
单线程延迟，再自动选出最快的 `--max-threads` 个 CPU 做正式测试。
raw-poll 正式测试前也会停止 endpoint 的后台 CQ poller 和 `io_context`
线程，避免额外线程影响调度。

如果要把 worker 分摊到多个 RDMA port/device，可以使用 endpoint 列表。
这些列表按 worker id 轮询匹配；长度为 1 的列表会广播到所有 worker：

```bash
./build/rdma_prompt_read_latency \
  --server-endpoints 192.168.98.70:2334,192.168.98.70:2335 \
  --ib-devices 0,0 \
  --ib-ports 1,2 \
  --cpu-list 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
```

更详细的 GDR 说明见 [`GDR_USAGE.md`](GDR_USAGE.md)。

## C++ RDMA PromptDB / LSH 延迟测试

`PromptDatabaseServer.py` 里的 LSH bucket、`RandomState(seed).rand(...)`
hash 矩阵、`zlib.crc32(str(signature)) % 20000`、Jaccard overlap
匹配逻辑已经有对应的 C++ 实现。RDMA 数据路径为：

1. client 暴露 prompt/result 两块 MR 给 server。
2. 每次查询 client 用预注册 request buffer 发一个小的 control request。
3. server 用 RDMA READ 从 client prompt MR 拉取当前 prompt。
4. server 在本地 C++ PromptDB 中执行 `find_most_similar_prompt`。
5. server 用 RDMA WRITE 把 result header + best token 写回 client result MR，
   再用预注册 ACK buffer 发一个 SEND 通知完成。

构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DRDMAPY_BUILD_PYTHON=OFF
cmake --build build -j --target rdma_promptdb_server rdma_promptdb_find_latency
```

启动 server：

```bash
./build/rdma_promptdb_server \
  --port 2333 \
  --ib-device 0 \
  --ib-port 1 \
  --io-threads 32 \
  --prompts 4096 \
  --token-len 32 \
  --vocab-size 32000 \
  --bucket-size 8 \
  --seed 42
```

运行 1 到 16 线程测试：

```bash
./build/rdma_promptdb_find_latency \
  --host <server-ip> \
  --port 2333 \
  --ib-device 0 \
  --ib-port 1 \
  --prompts 4096 \
  --token-len 32 \
  --vocab-size 32000 \
  --seed 42 \
  --min-threads 1 \
  --max-threads 16 \
  --repeats 1000 \
  --warmup 100 \
  --trials 3 \
  --no-round-sync
```

server 和 client 的 `--prompts/--token-len/--vocab-size/--seed` 要保持一致，
这样 benchmark 查询的是 server 预填充的同一批 prompt。输出默认会拆成：

- `total_avg_us`：client 观察到的端到端 query 时间。
- `prompt_transfer_avg_us`：server 用 RDMA READ 从计算节点 prompt MR 拉取
  当前 prompt 的时间。
- `server_match_avg_us`：server 执行 C++ LSH/Jaccard 匹配的时间。
- `match_bucket_hash_avg_us`：server 计算 LSH signature 和 CRC32 bucket id 的时间。
- `match_candidate_select_avg_us`：server 查找 bucket 并判断是否需要兜底扫描的时间。
- `match_overlap_scan_avg_us`：server 对候选 prompt 执行 Jaccard overlap 的时间。
- `match_result_copy_avg_us`：server 把当前 best prompt tokens 拷贝到返回结果里的时间。
- `server_match_other_avg_us`：`server_match_avg_us` 减去上述四个子阶段后的残差。
- `result_transfer_avg_us`：server 用 RDMA WRITE 把 best prompt/result 写回
  计算节点 result MR 的时间。
- `client_overhead_avg_us`：request SEND、ACK SEND/RECV 通知、CQ polling
  和其余没有归入上面三个 server 计时段的软件开销。

加 `--verbose` 会额外输出 total 的 p50/p95/p99 和各阶段 p50。

默认测试模式只在 worker 启动时同步一次，然后各 worker 独立循环，和
`rdma_prompt_read_latency.py` 的 steady 模式一致。需要复现“每一轮所有
worker 同时发起请求”的 burst 压测时再加 `--sync-each-repeat` 或
`--round-sync`。如果要降低 CPU 调度抖动，可以像 C++ READ benchmark 一样
加 `--cpu-list 0-15`，把第 N 个 worker 绑到列表里的第 N 个 CPU。
PromptDB server 的数据面和控制面都使用 raw-poll；server 用 `io_context`
只负责接收新连接，每个 client 连接会交给一个独立 handler 线程处理，所以
`--io-threads` 设为 2 到 4 通常就够了。16 个 client worker 时建议
`--max-connections 128`，避免反复测试时连接数上限太紧。

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
