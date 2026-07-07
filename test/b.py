from torch.utils.cpp_extension import load
import torch 

vector_add = load(
    name="vector_add",
    sources=[
        "vector_add.cpp",
        "vector_add_kernel.cu",
    ],
    verbose=True,
)
shape = (1024)
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = vector_add.vector_add(a,b)
print(c)
