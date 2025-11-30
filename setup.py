import os
import subprocess
import torch
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
# Import cpp_extension from torch.utils
from torch.utils import cpp_extension

__version__ = "0.1.0"

# Get the directory of this setup.py file
here = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Pybind11Extension(
        "hkv_embedding._C",
        [
            "src/kv_core.cpp",
            "src/kv_core_binding.cpp"
        ],
        include_dirs=[
            "src",
            cpp_extension.include_paths()[0],  # PyTorch includes
        ],
        cxx_std=14,
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="hkv_embedding",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hkv_embedding",
    description="High-performance key-value embedding for PyTorch",
    long_description="",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        "torch>=1.9.0",
    ],
    setup_requires=[
        "pybind11>=2.5.0",
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)