import rdma_py, time

sess = rdma_py.Session()
srv = rdma_py.Server(sess, 2333)

print("[server] waiting accept...")
srv.accept_once()
print("[server] accepted.")

# 0) 先收客户端的 READY，确保对端已建立并开始流程
print("[server] waiting READY...")
ready = srv.recv_bytes(5)
print("[server] got READY:", ready)

# 1) 暴露一块缓冲区
data = bytearray(b"hello_kvcache")
meta = srv.expose_buffer_as_mr(bytes(data))
print(f"[server] exposed MR, len={len(meta)} hex={meta.hex()}")

# 2) 把 MR 元数据发给 client（此时 client 已经会去挂 recv）
srv.send_bytes(meta)
print("[server] sent MR meta")

# 3) 等待 client 用 write_with_imm 写回来
print("[server] waiting write_with_imm...")
srv.wait_write_imm()
print("[server] write_with_imm received")

print("[server] buffer after write:", data.decode("utf-8", errors="ignore"))
