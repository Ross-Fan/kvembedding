cmake_minimum_required(VERSION 3.18)
project(kv_embedding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

if(DEFINED ENV{PYTHONPATH})
    # 尝试从 Python 路径推断 PyTorch 路径
    execute_process(
        COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PATH)
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
    endif()
endif()

# Add Abseil as subdirectory
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
set(ABSL_USE_EXTERNAL_GOOGLETEST OFF)
add_subdirectory(third_party/abseil-cpp)

# # Add source files
# add_library(kv_core SHARED
#     src/kv_core.cpp
#     src/kv_core_binding.cpp
# )
# Add source files
pybind11_add_module(kv_core_backend MODULE
    src/kv_core.cpp
    src/kv_core_binding.cpp
)

# Link libraries - 确保链接所有必要的 Abseil 组件
target_link_libraries(kv_core_backend 
    PRIVATE 
    ${TORCH_LIBRARIES}
    absl::flat_hash_map
    absl::raw_hash_set
    absl::hash
    absl::city
    absl::low_level_hash
    absl::container_common
    absl::strings
    absl::str_format
    absl::base
    absl::memory
    absl::meta
    absl::numeric
    absl::span
    absl::utility
    absl::type_traits
    absl::bits
    absl::core_headers
    absl::log_severity
)

# Set properties
# set_target_properties(kv_core PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        #  SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Link libraries
# target_link_libraries(kv_core ${TORCH_LIBRARIES} pybind11::module absl::flat_hash_map)
# target_include_directories(kv_core PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(kv_core_backend PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Compiler options
target_compile_options(kv_core_backend PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
# target_compile_options(kv_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>)
# target_link_options(kv_core PRIVATE -fopenmp)

# Set output directory
set_target_properties(kv_core_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/kv_table)