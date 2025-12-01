#!/bin/bash

# 清理之前的构建
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info python/kv_table/*.so

# 方法1：直接使用 CMake 构建
echo "Building with CMake..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

# 方法2：使用 setup.py 构建 wheel
echo "Building wheel..."
python setup.py bdist_wheel

echo "Build completed!"
ls -la dist/