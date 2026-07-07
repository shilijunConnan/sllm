#include <torch/extension.h>
#include "paged_attention.h"

// Python:
// output = paged_attention.forward(...)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Toy vLLM Paged Attention CUDA Extension";

    m.def(
        "forward",
        &paged_attention_forward,
        "Paged Attention Forward",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("block_table"),
        pybind11::arg("seq_len"),
        pybind11::arg("layer")
    );
}