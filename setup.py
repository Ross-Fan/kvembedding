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
torch_include_dirs = cpp_extension.include_paths()
print("::torch_include_dirs:", torch_include_dirs)

# Try to find Abseil headers
# we use it as from internal path 
absl_include_dirs = [os.path.join(here, 'third_party', 'abseil-cpp')]
# try:
#     # Try to get Abseil include path
#     result = subprocess.run(['pkg-config', '--cflags', 'absl_flat_hash_map'], 
#                           capture_output=True, text=True)
#     if result.returncode == 0:
#         absl_include_dirs = [result.stdout.strip().replace('-I', '')]
# except FileNotFoundError:
#     # pkg-config not available, try common paths
#     common_absl_paths = [
#         '/usr/local/include',
#         '/usr/include',
#         os.path.expanduser('~/abseil-cpp'),  # If manually installed in home
#     ]
#     for path in common_absl_paths:
#         absl_flat_hash_path = os.path.join(path, 'absl', 'container')
#         if os.path.exists(os.path.join(absl_flat_hash_path, 'flat_hash_map.h')):
#             absl_include_dirs = [path]
#             break
print("::absl_include_dirs:", absl_include_dirs)

ext_modules = [
    Pybind11Extension(
        "kv_table.kv_core_backend",
        [
            "src/kv_core.cpp",
            "src/kv_core_binding.cpp"
        ],
        include_dirs=[
            "src",
            
        ] + torch_include_dirs + absl_include_dirs,
        cxx_std=17,
        # extra_compile_args=["-O3", "-fopenmp"],
        # extra_link_args=["-fopenmp"],
        extra_compile_args=["-O3"],  # Removed -fopenmp
        extra_link_args=[],          # Removed -fopenmp
    ),
]

setup(
    name="kv_table",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/kv_embedding",
    description="High-performance key-value embedding for PyTorch",
    long_description="",
    packages=find_packages(where="python"),  # Specify python directory
    package_dir={"": "python"},              # Map root package to python directory
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