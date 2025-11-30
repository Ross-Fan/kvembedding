#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <torch/torch.h>
#include "kv_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(kv_core_backend, m) {
    py::class_<KVEmbeddingCore>(m, "KVEmbeddingCore")
        .def(py::init<int, double, double>())
        .def("fetch_vector", &KVEmbeddingCore::fetch_vector)
        .def("accumulate_gradients", &KVEmbeddingCore::accumulate_gradients)
        .def("apply_adam_updates", &KVEmbeddingCore::apply_adam_updates)
        .def("clear_gradients", &KVEmbeddingCore::clear_gradients)
        .def("get_keys", &KVEmbeddingCore::get_keys)
        .def("get_embedding_count", &KVEmbeddingCore::get_embedding_count);
}