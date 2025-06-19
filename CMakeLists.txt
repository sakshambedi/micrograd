cmake_minimum_required(VERSION 3.14)
project(micrograd)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)
include_directories(${eigen_SOURCE_DIR})

# Fetch and setup pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create a static library for testing
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
    GIT_TAG        9.0.1
)
FetchContent_MakeAvailable(xsimd)

add_library(cpu_kernel_lib STATIC cpu_kernel.cpp vecbuffer.cpp)
target_include_directories(cpu_kernel_lib PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_SOURCE_DIR}/include
    ${xsimd_SOURCE_DIR}/include
)


pybind11_add_module(cpu_kernel cpu_kernel.cpp vecbuffer.cpp)
target_include_directories(cpu_kernel PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${xsimd_SOURCE_DIR}/include
)



FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

# enable_testing()
# add_subdirectory(tests/kernels)
