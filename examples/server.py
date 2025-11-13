# /home/xingda/rdma_py/examples/server.py
import rdma_py, time

sess = rdma_py.Session()
srv = rdma_py.Server(sess, 2333)

print("[server] waiting accept...")
srv.accept_once()
print("[server] accepted.")

print("[server] ready to send")
srv.send_bytes(b"hello")          # 发 hello
print("[server] send hello")
resp = srv.recv_bytes(5)          # 等客户端回信
print("[server] recv:", resp)

print("[server] done.")
