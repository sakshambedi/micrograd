import sys

import pybind11
from setuptools import Extension, find_packages, setup

extra_compile_args = ["-std=c++17"]
if sys.platform == "darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
elif sys.platform == "win32":
    extra_compile_args = ["/EHsc"]
else:  # Linux
    extra_compile_args += ["-fPIC"]
ext_modules = [
    Extension(
        "grad.kernels.cpu_kernel",
        ["./kernels/cpu_kernel.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/opt/homebrew/include/eigen3",
            "/usr/include/eigen3",
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="micrograd-kernels",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    ext_modules=ext_modules,
)
