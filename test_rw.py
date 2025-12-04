import sys
import os
import time
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
from rdma_py import Endpoint


def human(x):
    return f"{x/1024/1024:.2f} MB"


def main():
    ep = Endpoint()
    sess = ep.connect("192.168.98.70", 2333)

    offset = 0

    # 测试写入大小（指数增长）
    sizes = [
        4 << 20,   # 4 MB
    ]

    print("Starting RDMA WRITE test...\n")

    for size in sizes:
        try:
            print(f"Testing WRITE of {human(size)} ... ", end="", flush=True)

            data = b"A" * size

            t0 = time.time()
            sess.write(offset, data)
            t1 = time.time()

            print(f"OK  (time={t1 - t0:.4f}s)")

        except Exception as e:
            print("FAIL!")
            print("Exception:", e)
            traceback.print_exc()
            print(f"\n❗ Max safe single WRITE ≈ {human(size//2)}\n")
            return

    print("\nAll tests OK??? (Unlikely)")
    print("But your HCA probably masked DMA SG limits with hugepages or ODP.\n")


if __name__ == "__main__":
    main()