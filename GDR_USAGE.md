# GPUDirect RDMA Tensor API

This branch keeps the original CPU bytes API:

```python
session.read(offset, length) -> bytes
session.write(offset, bytes_payload)
```

and adds CUDA tensor APIs:

```python
session.read_into_tensor(
    offset, tensor, nbytes=None, tensor_offset_bytes=0,
    chunk_bytes=64 * 1024 * 1024, max_inflight=16, debug=False)
session.write_from_tensor(
    offset, tensor, nbytes=None, tensor_offset_bytes=0,
    chunk_bytes=64 * 1024 * 1024, max_inflight=16, debug=False)
```

`Endpoint` also accepts the RDMA device index so you can choose the NIC closest
to the target GPU:

```python
endpoint = rdma_py.Endpoint(ib_port=1, device_num=1)
```

The tensor must be contiguous and CUDA-resident. The binding takes
`tensor.data_ptr()` and registers that address with ibverbs. If the host has a
working GPUDirect RDMA stack, the NIC DMAs directly into or out of GPU memory.

Before writing from a CUDA tensor, synchronize the CUDA work that produced it:

```python
torch.cuda.synchronize()
session.write_from_tensor(remote_offset, cuda_tensor)
```

After `read_into_tensor` returns, the RDMA completion has been observed by the
CPU thread. Avoid running CUDA kernels that read the destination tensor before
the call returns.

## Local MR cache

The tensor APIs keep a small local MR cache inside each `Session`. The first
operation on a CUDA allocation still calls `ibv_reg_mr`, but repeated
operations whose tensor byte range is covered by an existing cached MR reuse
that registration. Each RDMA chunk then posts with a subrange of that cached
MR, so large tensor operations do not re-register every chunk.

When `nbytes` is smaller than the tensor storage, the cache registers the
remaining contiguous tensor storage from `tensor_offset_bytes`, while RDMA only
transfers `nbytes`. This lets a larger reusable staging tensor serve many
smaller reads/writes.

Useful runtime helpers:

```python
session.mr_cache_stats()          # {"size": ..., "limit": ..., "hits": ..., "misses": ...}
session.set_mr_cache_limit(16)    # default is 16 entries; use 0 to disable
session.clear_mr_cache()          # deregister cached local MRs and reset stats
```

For safety, cached entries keep the Python tensor object alive until the entry
is evicted or `clear_mr_cache()` is called. This prevents a registered CUDA
allocation from being freed while the MR is cached, but it also means cached
entries retain GPU memory. For best hit rates and predictable memory use, reuse
CUDA staging tensors for repeated GDR reads/writes.

Minimal round-trip test:

```bash
python examples/gdr_tensor_smoke.py --host <server-ip> --port 2333 --device cuda:0 --ib-device 1 --max-inflight 16 --debug
```

Bandwidth sweep:

```bash
python examples/gdr_tensor_bench.py --host <server-ip> --port 2333 --device cuda:0 --ib-device 1 --bytes 512m --chunk-bytes 4m,16m,64m --max-inflight 1,4,8,16
```

`max_inflight` posts multiple RDMA WRs before waiting for a completion. For
RDMA read it is clamped to the QP's configured `max_rd_atomic` value (16 in
this tree). For RDMA write it can be higher, but it is still bounded by the QP
send queue capacity. `debug=True` prints per-call bytes, chunks, elapsed time,
estimated GB/s, and MR-cache deltas.

Sequoia's `RdmaBackend` also exposes the staging and debug knobs through
environment variables:

```bash
SEQUOIA_GDR_STAGE_BYTES=67108864
SEQUOIA_GDR_RDMA_CHUNK_BYTES=16777216
SEQUOIA_GDR_MAX_INFLIGHT=16
SEQUOIA_GDR_DEBUG=1
```

`SEQUOIA_GDR_STAGE_BYTES=4194304` restores the old conservative 4MB staging
size if a specific GDR driver stack still rejects larger CUDA registrations.
`SEQUOIA_GDR_RDMA_CHUNK_BYTES` controls the WR size inside each staging
transfer; with `SEQUOIA_GDR_MAX_INFLIGHT=16`, a 64MB stage and 16MB WRs posts
up to four RDMA operations before waiting.

Runtime prerequisites:

- Mellanox/NVIDIA OFED and a NIC that supports GPUDirect RDMA.
- NVIDIA driver/CUDA with the GPUDirect peer-memory module loaded
  (`nvidia-peermem` on modern drivers, or `nv_peer_memory` on older stacks).
- IOMMU configuration compatible with GPUDirect RDMA.
- Prefer a GPU and NIC under the same PCIe root complex.
