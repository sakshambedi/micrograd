cmake_minimum_required(VERSION 3.14)
project(micrograd)

set(CMAKE_CXX_STANDARD 17)


include_directories(/opt/homebrew/include/eigen3)

# Fetch and setup pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python
# find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create a static library for testing
add_library(cpu_kernel_lib STATIC kernels/cpu_kernel.cpp)
target_include_directories(cpu_kernel_lib PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_SOURCE_DIR}/include
)

# Create the pybind11 module library
pybind11_add_module(cpu_kernel kernels/cpu_kernel.cpp)
target_include_directories(cpu_kernel PRIVATE
    ${PYTHON_INCLUDE_DIRS}
)
# target_include_directories(test_cpu_kernel PRIVATE ${Python3_INCLUDE_DIRS})

# GoogleTest setup
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests/kernel_test)
