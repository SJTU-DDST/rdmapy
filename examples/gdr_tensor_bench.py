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
    parser.add_argument("--stage-bytes", type=parse_size, default=parse_size("64m"))
    parser.add_argument("--chunk-bytes", type=parse_size_list, default=parse_size_list("4m,16m,64m"))
    parser.add_argument("--max-inflight", type=parse_int_list, default=parse_int_list("1,4,8,16"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Register one full-size tensor. This may fail on GDR stacks that reject large CUDA MRs.",
    )
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
    stage_bytes = min(args.stage_bytes, nbytes)
    stage_numel = max(1, stage_bytes // elem_size)
    stage_bytes = stage_numel * elem_size

    endpoint = rdma_py.Endpoint(args.ib_port, args.ib_device)
    session = endpoint.connect(args.host, args.port)

    if args.direct:
        src = fill_tensor(numel, dtype, args.device).contiguous()
        dst = torch.empty_like(src)
    else:
        src = fill_tensor(stage_numel, dtype, args.device).contiguous()
        dst = torch.empty_like(src)
    torch.cuda.synchronize()

    print(
        "bytes={} dtype={} numel={} device={} repeats={} mode={} stage_bytes={}".format(
            nbytes,
            dtype,
            numel,
            args.device,
            args.repeats,
            "direct" if args.direct else "staged",
            nbytes if args.direct else stage_bytes,
        )
    )
    print("chunk_bytes,max_inflight,write_GBps,read_GBps,ok")

    def write_total(chunk_bytes: int, max_inflight: int):
        if args.direct:
            session.write_from_tensor(
                args.offset,
                src,
                nbytes=nbytes,
                chunk_bytes=chunk_bytes,
                max_inflight=max_inflight,
                debug=args.debug,
            )
            return

        written = 0
        while written < nbytes:
            this_bytes = min(stage_bytes, nbytes - written)
            session.write_from_tensor(
                args.offset + written,
                src,
                nbytes=this_bytes,
                chunk_bytes=chunk_bytes,
                max_inflight=max_inflight,
                debug=args.debug,
            )
            written += this_bytes

    def read_total(chunk_bytes: int, max_inflight: int):
        if args.direct:
            session.read_into_tensor(
                args.offset,
                dst,
                nbytes=nbytes,
                chunk_bytes=chunk_bytes,
                max_inflight=max_inflight,
                debug=args.debug,
            )
            return nbytes

        read_bytes = 0
        last_bytes = 0
        while read_bytes < nbytes:
            this_bytes = min(stage_bytes, nbytes - read_bytes)
            session.read_into_tensor(
                args.offset + read_bytes,
                dst,
                nbytes=this_bytes,
                chunk_bytes=chunk_bytes,
                max_inflight=max_inflight,
                debug=args.debug,
            )
            read_bytes += this_bytes
            last_bytes = this_bytes
        return last_bytes

    def check_last_chunk(last_bytes: int) -> bool:
        elems = last_bytes // elem_size
        return torch.equal(src[:elems], dst[:elems])

    for chunk_bytes in args.chunk_bytes:
        for max_inflight in args.max_inflight:
            write_total(chunk_bytes, max_inflight)
            last_bytes = read_total(chunk_bytes, max_inflight)
            torch.cuda.synchronize()

            write_seconds = 0.0
            read_seconds = 0.0
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                write_total(chunk_bytes, max_inflight)
                write_seconds += time.perf_counter() - t0

                t0 = time.perf_counter()
                last_bytes = read_total(chunk_bytes, max_inflight)
                read_seconds += time.perf_counter() - t0

            torch.cuda.synchronize()
            ok = check_last_chunk(last_bytes)
            write_gbps = (nbytes * args.repeats) / write_seconds / 1e9
            read_gbps = (nbytes * args.repeats) / read_seconds / 1e9
            print(f"{chunk_bytes},{max_inflight},{write_gbps:.2f},{read_gbps:.2f},{ok}")
            if not ok:
                elems = last_bytes // elem_size
                diff = (src[:elems] != dst[:elems]).nonzero()
                first = diff[0].item() if diff.numel() else -1
                raise RuntimeError(f"GDR roundtrip mismatch at element {first}")

    print(f"MR cache: {session.mr_cache_stats()}")


if __name__ == "__main__":
    main()
