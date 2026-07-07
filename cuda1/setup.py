from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="paged_attention_cuda1",
    ext_modules=[
        CUDAExtension(
            name="paged_attention_cuda1",
            sources=["binding.cpp", "paged_attention.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
