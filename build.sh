#!/bin/bash

# 清理之前的构建
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info python/kv_table/*.so

# 方法1：直接使用 CMake 构建
echo "Building with CMake..."
mkdir -p build
cd build

# 如果使用 conda 或 virtualenv，设置 Python 路径
if command -v python &> /dev/null; then
    PYTHON_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        export CMAKE_PREFIX_PATH="$PYTHON_PATH:$CMAKE_PREFIX_PATH"
        echo "Setting CMAKE_PREFIX_PATH to: $CMAKE_PREFIX_PATH"
    fi
fi


cmake ..
make -j$(nproc)
cd ..

# 方法2：使用 setup.py 构建 wheel
echo "Building wheel..."
python setup.py bdist_wheel

echo "Build completed!"
ls -la dist/