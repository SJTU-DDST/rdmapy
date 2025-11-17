# client_mr_test.py
import rdma_py
import struct

sess = rdma_py.Session()
cli = rdma_py.Client(sess, "127.0.0.1", 2333)

# 0) 告诉 server 我已就绪
cli.send_bytes(b"READY")
print("[client] sent READY")

# 1) 先收远端 buffer 长度（4 字节）
size_bytes = cli.recv_bytes(4)
(buf_len,) = struct.unpack("!I", size_bytes)
print("[client] remote buffer len =", buf_len)

# 2) 再收 MR meta（内部用 kSerializedSize，所以不用管长度）
print("[client] ready to recv MR meta")
cli.recv_remote_mr_meta()
print("[client] recv_remote_mr_meta done")

# 3) RDMA read
buf = bytearray(buf_len)
n = cli.read_from_remote(buf, buf_len)
print(f"[client] RDMA read {n} bytes:",
      buf[:n].decode("utf-8", errors="ignore"))

# 4) 修改内容并 RDMA write_with_imm 回去
payload = b"world_cache!!!!!"[:buf_len]  # 截断到不超过 buf_len
cli.write_to_remote_with_imm(payload, len(payload), 1)
print("[client] RDMA write_with_imm done")
