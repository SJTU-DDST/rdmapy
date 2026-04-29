import argparse
import time

import rdma_py
import torch


def parse_size(text: str) -> int:
    value = text.strip().lower()
    scale = 1
    if value.endswith("kb") or value.endswith("k"):
        scale = 1024
        value = value.rstrip("b").rstrip("k")
    elif value.endswith("mb") or value.endswith("m"):
        scale = 1024 * 1024
        value = value.rstrip("b").rstrip("m")
    elif value.endswith("gb") or value.endswith("g"):
        scale = 1024 * 1024 * 1024
        value = value.rstrip("b").rstrip("g")
    return int(float(value) * scale)


def parse_size_list(text: str) -> list[int]:
    return [parse_size(part) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "int8": torch.int8,
        "int32": torch.int32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def fill_tensor(numel: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.int8:
        return (torch.arange(numel, dtype=torch.int32, device=device) % 127).to(torch.int8)
    if dtype == torch.int32:
        return torch.arange(numel, dtype=torch.int32, device=device)
    return torch.arange(numel, dtype=torch.float32, device=device).to(dtype)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPUDirect RDMA tensor APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2333)
    parser.add_argument("--ib-port", type=int, default=1)
    parser.add_argument("--ib-device", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offset", type=parse_size, default=0)
    parser.add_argument("--bytes", dest="nbytes", type=parse_size, default=parse_size("256m"))
    parser.add_argument("--dtype", default="int32")
    parser.add_argument("--chunk-bytes", type=parse_size_list, default=parse_size_list("4m,16m,64m"))
    parser.add_argument("--max-inflight", type=parse_int_list, default=parse_int_list("1,4,8,16"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPUDirect RDMA needs a CUDA tensor")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    dtype = dtype_from_name(args.dtype)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = args.nbytes // elem_size
    nbytes = numel * elem_size
    if numel <= 0:
        raise ValueError("--bytes is smaller than one dtype element")

    endpoint = rdma_py.Endpoint(args.ib_port, args.ib_device)
    session = endpoint.connect(args.host, args.port)

    src = fill_tensor(numel, dtype, args.device).contiguous()
    dst = torch.empty_like(src)
    torch.cuda.synchronize()

    print(
        "bytes={} dtype={} numel={} device={} repeats={}".format(
            nbytes, dtype, numel, args.device, args.repeats
        )
    )
    print("chunk_bytes,max_inflight,write_GBps,read_GBps,ok")

    for chunk_bytes in args.chunk_bytes:
        for max_inflight in args.max_inflight:
            session.write_from_tensor(
                args.offset,
                src,
                nbytes=nbytes,
                chunk_bytes=chunk_bytes,
                max_inflight=max_inflight,
                debug=args.debug,
            )
            session.read_into_tensor(
                args.offset,
                dst,
                nbytes=nbytes,
                chunk_bytes=chunk_bytes,
                max_inflight=max_inflight,
                debug=args.debug,
            )
            torch.cuda.synchronize()

            write_seconds = 0.0
            read_seconds = 0.0
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                session.write_from_tensor(
                    args.offset,
                    src,
                    nbytes=nbytes,
                    chunk_bytes=chunk_bytes,
                    max_inflight=max_inflight,
                    debug=args.debug,
                )
                write_seconds += time.perf_counter() - t0

                t0 = time.perf_counter()
                session.read_into_tensor(
                    args.offset,
                    dst,
                    nbytes=nbytes,
                    chunk_bytes=chunk_bytes,
                    max_inflight=max_inflight,
                    debug=args.debug,
                )
                read_seconds += time.perf_counter() - t0

            torch.cuda.synchronize()
            ok = torch.equal(src, dst)
            write_gbps = (nbytes * args.repeats) / write_seconds / 1e9
            read_gbps = (nbytes * args.repeats) / read_seconds / 1e9
            print(f"{chunk_bytes},{max_inflight},{write_gbps:.2f},{read_gbps:.2f},{ok}")
            if not ok:
                diff = (src != dst).nonzero()
                first = diff[0].item() if diff.numel() else -1
                raise RuntimeError(f"GDR roundtrip mismatch at element {first}")

    print(f"MR cache: {session.mr_cache_stats()}")


if __name__ == "__main__":
    main()
