import rdma_py, time

sess = rdma_py.Session()
cli = rdma_py.Client(sess, "127.0.0.1", 2333)

# 0) 先告诉 server 我已就绪
cli.send_bytes(b"READY")
print("[client] sent READY")

# 1) 立刻挂上对 MR meta 的接收（阻塞等待）
print("[client] ready to recv MR meta")
cli.recv_remote_mr_meta()     # C++ 内部用正确长度
print("[client] recv_remote_mr_meta done")

# 2) RDMA read 远端内容
buf = bytearray(32)
n = cli.read_from_remote(buf, len(buf))
print(f"[client] RDMA read {n} bytes:", buf[:n].decode("utf-8", errors="ignore"))

# 3) 修改内容并写回（with imm）
payload = b"world_cache!"
cli.write_to_remote_with_imm(payload, len(payload), 1)
print("[client] RDMA write_with_imm done")
