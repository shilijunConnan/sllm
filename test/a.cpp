#include <torch/extension.h>

// 实现一个简单的张量相加
torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// 绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &my_add, "My Add forward (C++)",py::arg("a"), py::arg("b"));
}