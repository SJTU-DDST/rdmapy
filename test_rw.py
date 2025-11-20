# ~/rdma_py/test_rw.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
from rdma_py import Endpoint


def main():
    ep = Endpoint()
    sess = ep.connect("127.0.0.1", 2333)

    offset = 0
    data = b"ABCDEFGHIJ"

    print(f"Write {len(data)} bytes at offset {offset}")
    sess.write(offset, data)

    out = sess.read(offset, len(data))
    print("Read back:", out, "==", data, "?", out == data)


if __name__ == "__main__":
    main()
