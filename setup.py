from setuptools import setup
from torch.utils import cpp_extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='torchturbojpeg',
    version ='0.0.1',
    author       = "GhostCai",
    author_email = "zqtsai@gmail.com",
    ext_modules=[
        cpp_extension.CUDAExtension(name='torchturbojpeg',
                      sources=['torchturbojpeg.cpp'],
                      extra_compile_args={'cxx': ['-std=c++17'], 'nvcc': ['-O3', '-use_fast_math']},
                      libraries=['nvjpeg'],
        )
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)