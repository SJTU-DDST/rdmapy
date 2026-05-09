#include <cstdint>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

#include "rdma_session.hpp"

namespace py = pybind11;

namespace {

constexpr std::uint64_t kDefaultChunkBytes = 64ull * 1024 * 1024;

std::uint64_t checked_u64_add(std::uint64_t a, std::uint64_t b,
                              const char *what) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        throw std::overflow_error(std::string(what) + " overflows uint64");
    }
    return a + b;
}

struct TensorSpan {
    std::uintptr_t addr;
    std::uint64_t nbytes;
    std::uint64_t register_nbytes;
};

TensorSpan tensor_span(py::object tensor,
                       py::object py_nbytes,
                       std::uint64_t tensor_offset_bytes,
                       bool require_cuda) {
    bool is_contiguous = tensor.attr("is_contiguous")().cast<bool>();
    if (!is_contiguous) {
        throw std::runtime_error("tensor must be contiguous");
    }

    bool is_cuda = tensor.attr("is_cuda").cast<bool>();
    if (require_cuda && !is_cuda) {
        throw std::runtime_error("tensor must be a CUDA tensor for GPUDirect RDMA");
    }

    auto data_ptr = tensor.attr("data_ptr")().cast<std::uintptr_t>();
    auto numel = tensor.attr("numel")().cast<std::uint64_t>();
    auto elem_size = tensor.attr("element_size")().cast<std::uint64_t>();

    if (elem_size != 0 &&
        numel > std::numeric_limits<std::uint64_t>::max() / elem_size) {
        throw std::overflow_error("tensor byte size overflows uint64");
    }
    std::uint64_t total_nbytes = numel * elem_size;

    if (tensor_offset_bytes > total_nbytes) {
        throw std::runtime_error("tensor_offset_bytes is past the tensor end");
    }

    std::uint64_t nbytes = 0;
    if (py_nbytes.is_none()) {
        nbytes = total_nbytes - tensor_offset_bytes;
    } else {
        nbytes = py_nbytes.cast<std::uint64_t>();
        if (checked_u64_add(tensor_offset_bytes, nbytes,
                            "tensor byte range") > total_nbytes) {
            throw std::runtime_error("requested byte range exceeds tensor storage");
        }
    }

    if (nbytes > 0 && data_ptr == 0) {
        throw std::runtime_error("tensor data_ptr is null");
    }

    return TensorSpan{
        checked_u64_add(data_ptr, tensor_offset_bytes, "tensor data pointer"),
        nbytes,
        total_nbytes - tensor_offset_bytes,
    };
}

std::shared_ptr<void> keep_python_object_alive(py::object obj) {
    auto *held = new py::object(std::move(obj));
    return std::shared_ptr<void>(
        held,
        [](void *ptr) {
            py::gil_scoped_acquire gil;
            delete static_cast<py::object *>(ptr);
        });
}

void set_spdlog_level_from_name(const std::string& level) {
    if (level == "trace") {
        spdlog::set_level(spdlog::level::trace);
    } else if (level == "debug") {
        spdlog::set_level(spdlog::level::debug);
    } else if (level == "info") {
        spdlog::set_level(spdlog::level::info);
    } else if (level == "warn" || level == "warning") {
        spdlog::set_level(spdlog::level::warn);
    } else if (level == "error") {
        spdlog::set_level(spdlog::level::err);
    } else if (level == "critical") {
        spdlog::set_level(spdlog::level::critical);
    } else if (level == "off") {
        spdlog::set_level(spdlog::level::off);
    } else {
        throw std::runtime_error("unsupported rdma log level: " + level);
    }
}

} // namespace

PYBIND11_MODULE(rdma_py, m) {
    m.doc() = "RDMA KV-cache bindings using rdmapp";
    set_spdlog_level_from_name("warn");

    m.def("set_log_level",
          [](const std::string& level) {
              set_spdlog_level_from_name(level);
          },
          py::arg("level"));

    py::class_<RdmaSession, std::shared_ptr<RdmaSession>>(m, "Session")
        .def("read",
             [](RdmaSession &sess,
                std::uint64_t offset,
                std::uint64_t length) {
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
                std::uint64_t offset,
                py::bytes py_data) {
                 std::string buf = py_data;
                 sess.write(offset,
                            reinterpret_cast<
                                const std::uint8_t *>(buf.data()),
                            buf.size());
             },
             py::arg("offset"),
             py::arg("data"))
        .def("read_into_tensor",
             [](RdmaSession &sess,
                std::uint64_t offset,
                py::object tensor,
                py::object py_nbytes,
                std::uint64_t tensor_offset_bytes,
                std::uint64_t chunk_bytes,
                std::uint32_t max_inflight,
                bool debug) {
                 auto span = tensor_span(
                     tensor, py_nbytes, tensor_offset_bytes,
                     true /* require_cuda */);
                 auto owner = keep_python_object_alive(tensor);
                 {
                     py::gil_scoped_release release;
                     sess.read_into(
                         offset,
                         span.addr,
                         span.nbytes,
                         chunk_bytes,
                         std::move(owner),
                         span.register_nbytes,
                         max_inflight,
                         debug);
                 }
                 return span.nbytes;
             },
             py::arg("offset"),
             py::arg("tensor"),
             py::arg("nbytes") = py::none(),
             py::arg("tensor_offset_bytes") = 0,
             py::arg("chunk_bytes") = kDefaultChunkBytes,
             py::arg("max_inflight") = 16,
             py::arg("debug") = false)
        .def("read_into_tensor_latencies_us",
             [](RdmaSession &sess,
                std::uint64_t offset,
                py::object tensor,
                py::object py_nbytes,
                std::uint32_t repeats,
                std::uint32_t warmup,
                std::uint64_t tensor_offset_bytes,
                std::uint64_t chunk_bytes,
                std::uint32_t max_inflight,
                std::uint64_t pace_us,
                bool debug) {
                 auto span = tensor_span(
                     tensor, py_nbytes, tensor_offset_bytes,
                     true /* require_cuda */);
                 auto owner = keep_python_object_alive(tensor);
                 std::vector<double> latencies_us;
                 latencies_us.reserve(repeats);

                 {
                     py::gil_scoped_release release;
                     const std::uint32_t total_iters = warmup + repeats;
                     const auto interval =
                         std::chrono::microseconds(pace_us);
                     auto next_issue = std::chrono::steady_clock::now();
                     for (std::uint32_t iteration = 0;
                          iteration < total_iters;
                          ++iteration) {
                         if (pace_us > 0 && iteration > 0) {
                             next_issue += interval;
                             auto now = std::chrono::steady_clock::now();
                             if (next_issue > now) {
                                 std::this_thread::sleep_until(next_issue);
                             } else {
                                 next_issue = now;
                             }
                         }
                         double latency_us = sess.read_into_transfer_latency_us(
                             offset,
                             span.addr,
                             span.nbytes,
                             chunk_bytes,
                             owner,
                             span.register_nbytes,
                             max_inflight,
                             debug);
                         if (iteration >= warmup) {
                             latencies_us.push_back(latency_us);
                         }
                     }
                 }

                 return latencies_us;
             },
             py::arg("offset"),
             py::arg("tensor"),
             py::arg("nbytes") = py::none(),
             py::arg("repeats") = 100,
             py::arg("warmup") = 10,
             py::arg("tensor_offset_bytes") = 0,
             py::arg("chunk_bytes") = kDefaultChunkBytes,
             py::arg("max_inflight") = 1,
             py::arg("pace_us") = 0,
             py::arg("debug") = false)
        .def("write_from_tensor",
             [](RdmaSession &sess,
                std::uint64_t offset,
                py::object tensor,
                py::object py_nbytes,
                std::uint64_t tensor_offset_bytes,
                std::uint64_t chunk_bytes,
                std::uint32_t max_inflight,
                bool debug) {
                 auto span = tensor_span(
                     tensor, py_nbytes, tensor_offset_bytes,
                     true /* require_cuda */);
                 auto owner = keep_python_object_alive(tensor);
                 {
                     py::gil_scoped_release release;
                     sess.write_from(
                         offset,
                         span.addr,
                         span.nbytes,
                         chunk_bytes,
                         std::move(owner),
                         span.register_nbytes,
                         max_inflight,
                         debug);
                 }
                 return span.nbytes;
             },
             py::arg("offset"),
             py::arg("tensor"),
             py::arg("nbytes") = py::none(),
             py::arg("tensor_offset_bytes") = 0,
             py::arg("chunk_bytes") = kDefaultChunkBytes,
             py::arg("max_inflight") = 16,
             py::arg("debug") = false)
        .def("mr_cache_size",
             &RdmaSession::mr_cache_size)
        .def("mr_cache_limit",
             &RdmaSession::mr_cache_limit)
        .def("set_mr_cache_limit",
             &RdmaSession::set_mr_cache_limit,
             py::arg("max_entries"))
        .def("clear_mr_cache",
             &RdmaSession::clear_mr_cache)
        .def("mr_cache_stats",
             [](RdmaSession &sess) {
                 py::dict stats;
                 stats["size"] = sess.mr_cache_size();
                 stats["limit"] = sess.mr_cache_limit();
                 stats["hits"] = sess.mr_cache_hits();
                 stats["misses"] = sess.mr_cache_misses();
                 return stats;
             });

    py::class_<RdmaEndpoint>(m, "Endpoint")
        .def(py::init<int, int, std::uint32_t, std::size_t>(),
             py::arg("ib_port") = 1,
             py::arg("device_num") = 1,
             py::arg("io_workers") = 1,
             py::arg("cq_batch_size") = 16)
        .def("connect",
             &RdmaEndpoint::connect,
             py::arg("host"),
             py::arg("port"));
}
