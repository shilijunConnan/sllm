#include <torch/extension.h>

#include "paged_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SLLM paged attention CUDA extension";
    m.def(
        "forward",
        &paged_attention_forward,
        "Paged attention forward for decode",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("block_tables"),
        pybind11::arg("seq_lens"),
        pybind11::arg("layer"));
}
