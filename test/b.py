import torch
from torch.utils.cpp_extension import load

# 运行时自动编译
my_lib = load(
    name="my_add_extension",
    sources=["a.cpp"],
    verbose=True
)

# 使用编译后的扩展
x = torch.ones(3)
y = torch.ones(3) * 2
z = my_lib.forward(x, y)

print(z)  # 输出: tensor([3., 3., 3.])

