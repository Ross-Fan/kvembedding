cmake_minimum_required(VERSION 3.18)
project(hkv_embedding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Add source files
add_library(kv_core SHARED
    src/kv_core.cpp
    src/kv_core_binding.cpp
)

# Set properties
set_target_properties(kv_core PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Link libraries
target_link_libraries(kv_core ${TORCH_LIBRARIES} pybind11::module)
target_include_directories(kv_core PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Compiler options
target_compile_options(kv_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
# target_compile_options(kv_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>)
# target_link_options(kv_core PRIVATE -fopenmp)

# Set output directory
set_target_properties(kv_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/kv_table)