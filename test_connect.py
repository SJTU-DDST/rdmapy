# ~/rdma_py/test_connect.py
import sys
import os

# 让 Python 能找到 build 里的 .so
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

from rdma_py import Endpoint

def main():
    ep = Endpoint()
    sess = ep.connect("127.0.0.1", 2333)
    print("Connected OK:", sess)

if __name__ == "__main__":
    main()
