#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rdma_session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rdma_py, m) {
    m.doc() = "RDMA KV-cache bindings using rdmapp";

    py::class_<RdmaSession, std::shared_ptr<RdmaSession>>(m, "Session")
        .def("read",
             [](RdmaSession &sess,
                std::uint32_t offset,
                std::uint32_t length) {
                 auto data = sess.read(offset, length);
                 // 返回 Python bytes
                 return py::bytes(
                     reinterpret_cast<const char *>(data.data()),
                     data.size());
             },
             py::arg("offset"),
             py::arg("length"))
        .def("write",
             [](RdmaSession &sess,
                std::uint32_t offset,
                py::bytes py_data) {
                 std::string buf = py_data;
                 sess.write(offset,
                            reinterpret_cast<
                                const std::uint8_t *>(buf.data()),
                            buf.size());
             },
             py::arg("offset"),
             py::arg("data"));

    py::class_<RdmaEndpoint>(m, "Endpoint")
        .def(py::init<int>(), py::arg("ib_port") = 1)
        .def("connect",
             &RdmaEndpoint::connect,
             py::arg("host"),
             py::arg("port"));
}
