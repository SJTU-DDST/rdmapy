# /home/xingda/rdma_py/examples/client.py
import rdma_py

sess = rdma_py.Session()
cli = rdma_py.Client(sess, "127.0.0.1", 2333)

print("ready to recv")
msg = cli.recv_bytes(5)          # 收服务端的 hello
print("[client] recv:", msg)

print("ready to send")
cli.send_bytes(b"world")         # 回 world
print("[client] sent: world")

print("[client] done.")
