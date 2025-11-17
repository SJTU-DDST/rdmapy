# server_mr_test.py
import rdma_py
import struct

sess = rdma_py.Session()
srv = rdma_py.Server(sess, 2333)

print("[server] waiting accept...")
srv.accept_once()
print("[server] accepted.")

# 0) 先收 READY
print("[server] waiting READY...")
ready = srv.recv_bytes(5)
print("[server] got READY:", ready)

# 1) 暴露一块缓冲区（注意：直接用 bytearray，不要 bytes(...)）
data = bytearray(b"hello_kvcache")
buf_len = len(data)
print("[server] local buffer len =", buf_len)

# 先把长度发给 client（4 字节 network order）
srv.send_bytes(struct.pack("!I", buf_len))

# 然后暴露 MR，并发 meta
meta = srv.expose_buffer_as_mr(data)   # ✅ 这里不要 bytes()
print(f"[server] exposed MR, meta_len={len(meta)}")
srv.send_bytes(meta)

# 3) 等待 client 用 write_with_imm 写回来
print("[server] waiting write_with_imm...")
srv.wait_write_imm()
print("[server] write_with_imm received")

print("[server] buffer after write:",
      data.decode("utf-8", errors="ignore"))
