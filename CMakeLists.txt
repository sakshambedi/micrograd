cmake_minimum_required(VERSION 3.14)
project(micrograd)

set(CMAKE_CXX_STANDARD 17)

# Set up vcpkg toolchain if not already set
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE
            "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING
            "Vcpkg toolchain file")
    elseif(WIN32 AND EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    elseif(UNIX AND NOT APPLE AND EXISTS "/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    elseif(APPLE AND EXISTS "/Users/${USER}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "/Users/${USER}/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

include(FetchContent)

# Fetch Eigen
# DO NOT REVOVE THE NEXT TWO LINES, IF REMOVED IT STARTS EIGEN TEST SUITES
set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests")
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

# Detect and configure xsimd include directories
if(DEFINED ENV{XSIMD_INCLUDE_DIR})
    set(XSIMD_INCLUDE_DIR $ENV{XSIMD_INCLUDE_DIR})
elseif(APPLE)
    set(XSIMD_INCLUDE_DIR "/Users/${USER}/vcpkg/installed/arm64-osx/include")
    if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
        set(XSIMD_INCLUDE_DIR "/opt/homebrew/include")
    endif()
elseif(WIN32)
    set(XSIMD_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
else()
    set(XSIMD_INCLUDE_DIR "/usr/local/vcpkg/installed/x64-linux/include")
    if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
        set(XSIMD_INCLUDE_DIR "/usr/include")
    endif()
endif()

if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
    message(WARNING "xsimd headers not found at ${XSIMD_INCLUDE_DIR}. You may need to install xsimd with vcpkg or specify XSIMD_INCLUDE_DIR manually.")
endif()

# Create a static library for testing
add_library(cpu_kernel_lib STATIC
    kernels/cpu_kernel.cpp
    kernels/vecbuffer.cpp)
target_include_directories(cpu_kernel_lib PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_SOURCE_DIR}/include
    ${XSIMD_INCLUDE_DIR}
)

# Create Python module
pybind11_add_module(cpu_kernel kernels/cpu_kernel.cpp kernels/vecbuffer.cpp)
target_include_directories(cpu_kernel PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${XSIMD_INCLUDE_DIR}
)

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()
add_subdirectory(tests/kernels)
