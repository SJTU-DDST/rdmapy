import argparse
import time

import rdma_py
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for GPUDirect RDMA tensor read/write APIs."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2333)
    parser.add_argument("--ib-port", type=int, default=1)
    parser.add_argument("--ib-device", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--numel", type=int, default=1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPUDirect RDMA needs a CUDA tensor")

    endpoint = rdma_py.Endpoint(args.ib_port, args.ib_device)
    session = endpoint.connect(args.host, args.port)

    src = torch.arange(args.numel, dtype=torch.int32, device=args.device).contiguous()
    dst = torch.empty_like(src)

    # Ensure the GPU kernel that produced src is complete before the NIC reads it.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    written = session.write_from_tensor(
        args.offset,
        src,
        chunk_bytes=args.chunk_bytes,
        max_inflight=args.max_inflight,
        debug=args.debug,
    )
    write_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    read = session.read_into_tensor(
        args.offset,
        dst,
        chunk_bytes=args.chunk_bytes,
        max_inflight=args.max_inflight,
        debug=args.debug,
    )
    read_seconds = time.perf_counter() - t0

    # Ensure the comparison kernel sees the completed RDMA write into dst.
    torch.cuda.synchronize()
    if not torch.equal(src, dst):
        diff = (src != dst).nonzero()
        first = diff[0].item() if diff.numel() else -1
        raise RuntimeError(f"GDR roundtrip mismatch at element {first}")

    write_gbps = written / write_seconds / 1e9 if write_seconds > 0 else 0.0
    read_gbps = read / read_seconds / 1e9 if read_seconds > 0 else 0.0
    print(
        "OK: wrote={} bytes read={} bytes device={} chunk_bytes={} max_inflight={}".format(
            written, read, args.device, args.chunk_bytes, args.max_inflight
        )
    )
    print(f"write={write_seconds:.6f}s {write_gbps:.2f} GB/s")
    print(f"read ={read_seconds:.6f}s {read_gbps:.2f} GB/s")
    print(f"MR cache: {session.mr_cache_stats()}")


if __name__ == "__main__":
    main()
