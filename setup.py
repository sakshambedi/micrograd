import os
import platform
import sys

import pybind11
from setuptools import Extension, find_packages, setup

extra_compile_args = [
    "-std=c++17" if not sys.platform == "win32" else "/std:c++17",
    "-O3" if not sys.platform == "win32" else "/O2",
    "-ffast-math" if not sys.platform == "win32" else "/fp:fast",
    "-DEIGEN_MAX_ALIGN_BYTES=64",
]

# Platform-specific compiler flags
if sys.platform == "darwin":
    extra_compile_args += [
        "-stdlib=libc++",
        "-mmacosx-version-min=10.9",
    ]
    # Add Apple Silicon specific flags if on M1/M2
    if platform.machine() == "arm64":
        extra_compile_args += ["-mcpu=apple-m1", "-Wnan-infinity-disabled"]
elif sys.platform == "win32":
    extra_compile_args += ["/EHsc"]
else:
    extra_compile_args += ["-fPIC"]
    # Add architecture optimization
    if platform.machine() == "x86_64":
        extra_compile_args += ["-march=x86-64"]
    elif platform.machine().startswith("arm"):
        extra_compile_args += ["-march=armv8-a"]


# Determine include directories based on platform
def get_include_dirs():
    dirs = [pybind11.get_include()]

    # Eigen include directory
    if "EIGEN3_INCLUDE_DIR" in os.environ:
        dirs.append(os.environ["EIGEN3_INCLUDE_DIR"])
    elif sys.platform == "darwin":
        dirs.append("/opt/homebrew/include/eigen3")
    elif sys.platform == "win32":
        dirs.append("C:/vcpkg/installed/x64-windows/include/eigen3")
    else:  # Linux
        for path in ["/usr/include/eigen3", "/usr/local/include/eigen3"]:
            if os.path.exists(path):
                dirs.append(path)
                break

    # XSIMD include directory
    if "XSIMD_INCLUDE_DIR" in os.environ:
        dirs.append(os.environ["XSIMD_INCLUDE_DIR"])
    elif sys.platform == "darwin":
        if platform.machine() == "arm64":
            xsimd_dir = os.path.expanduser("~/vcpkg/installed/arm64-osx/include")
        else:
            xsimd_dir = os.path.expanduser("~/vcpkg/installed/x64-osx/include")
        dirs.append(xsimd_dir)
    elif sys.platform == "win32":
        dirs.append("C:/vcpkg/installed/x64-windows/include")
    else:  # Linux
        for path in [
            "/usr/local/vcpkg/installed/x64-linux/include",
            os.path.expanduser("~/vcpkg/installed/x64-linux/include"),
            "/usr/include",
        ]:
            if os.path.exists(path):
                dirs.append(path)
                break

    return dirs


ext_modules = [
    Extension(
        "grad.kernels.cpu_kernel",
        ["./kernels/cpu_kernel.cpp", "./kernels/vecbuffer.cpp"],
        include_dirs=get_include_dirs(),
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="micrograd-kernels",
    version="0.12.2",
    packages=find_packages(),
    package_dir={"": "."},
    ext_modules=ext_modules,
)
