#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rdma_wrap.hpp"

namespace py = pybind11;
using namespace pywrap;

PYBIND11_MODULE(rdma_py, m) {
  m.doc() = "Python bindings for rdmapp examples (acceptor/connector/qp)";

  py::class_<Session, std::shared_ptr<Session>>(m, "Session")
    .def(py::init<int,int>(), py::arg("dev_index")=0, py::arg("port")=1);

  py::class_<Server>(m, "Server")
    .def(py::init<std::shared_ptr<Session>, uint16_t>(),
         py::arg("session"), py::arg("port"))
    .def("accept_once", &Server::accept_once)
    .def("send_bytes",  &Server::send_bytes)
    .def("recv_bytes",  &Server::recv_bytes)
    .def("expose_buffer_as_mr", [](Server& s, py::bytes pybuf) {
        // 把 Python bytes 拷到一块持久缓冲
        std::string holder = pybuf;
        std::string meta = s.expose_buffer_as_mr(holder.data(), holder.size());
        return py::bytes(meta);  // ✅ 用 py::bytes 包装返回值，防止 UTF-8 解码
    })
    .def("wait_write_imm", &Server::wait_write_imm)
    .def("wait_atomic_done", &Server::wait_atomic_done);

  py::class_<Client>(m, "Client")
    .def(py::init<std::shared_ptr<Session>, std::string, uint16_t>(),
         py::arg("session"), py::arg("host"), py::arg("port"))
    .def("send_bytes",  &Client::send_bytes)
    .def("recv_bytes",  &Client::recv_bytes)
    .def("recv_remote_mr_meta", &Client::recv_remote_mr_meta)
    .def("read_from_remote", [](Client& c, size_t n){
        std::string out; out.resize(n);
        size_t got = c.read_from_remote(out.data(), n);
        out.resize(got);
        return py::bytes(out);
    })
    .def("write_to_remote_with_imm", [](Client& c, py::bytes pybuf, uint32_t imm){
        std::string tmp = pybuf;
        c.write_to_remote_with_imm(tmp.data(), tmp.size(), imm);
    })
    .def("recv_counter_mr_meta", &Client::recv_counter_mr_meta)
    .def("fetch_and_add", &Client::fetch_and_add, py::arg("addend"))
    .def("compare_and_swap", &Client::compare_and_swap,
         py::arg("expected"), py::arg("desired"));
}
