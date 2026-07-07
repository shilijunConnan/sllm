#include <torch/extension.h>

void vector_add_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c);

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b)
{
    auto c = torch::zeros_like(a);

    vector_add_cuda(a, b, c);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vector_add", &vector_add, "Vector Add (CUDA)");
}