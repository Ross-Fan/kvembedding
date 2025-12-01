import os
import sys 
import subprocess
import torch
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
# Import cpp_extension from torch.utils
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup, find_packages

__version__ = "0.1.0"

# Get the directory of this setup.py file
here = os.path.dirname(os.path.abspath(__file__))
# torch_include_dirs = cpp_extension.include_paths()
# print("::torch_include_dirs:", torch_include_dirs)

# Try to find Abseil headers
# we use it as from internal path 
# absl_include_dirs = [os.path.join(here, 'third_party', 'abseil-cpp')]
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
# print("::absl_include_dirs:", absl_include_dirs)
class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        # 获取项目根目录
        project_root = Path(__file__).parent.absolute()
        
        # 创建构建目录
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # 获取扩展的目标目录
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        
        # 配置 CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        
        # 构建参数
        build_args = ["--config", "Release", "--", "-j4"]
        
        # 运行 CMake 配置和构建
        os.chdir(str(build_temp))
        self.spawn(["cmake", str(project_root)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--target", "kv_core_backend"] + build_args)
        os.chdir(str(project_root))

# ext_modules = [
#     Pybind11Extension(
#         "kv_table.kv_core_backend",
#         [
#             "src/kv_core.cpp",
#             "src/kv_core_binding.cpp"
#         ],
#         include_dirs=[
#             "src",
            
#         ] + torch_include_dirs + absl_include_dirs,
#         cxx_std=17,
#         # extra_compile_args=["-O3", "-fopenmp"],
#         # extra_link_args=["-fopenmp"],
#         extra_compile_args=["-O3"],  # Removed -fopenmp
#         extra_link_args=[],          # Removed -fopenmp
#     ),
# ]
# 简化的扩展模块定义（实际构建由 CMake 完成）
ext_modules = [
    Pybind11Extension(
        "kv_table.kv_core_backend",
        ["src/kv_core_binding.cpp"],  # 这只是一个占位符
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
    cmdclass={"build_ext": CMakeBuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)