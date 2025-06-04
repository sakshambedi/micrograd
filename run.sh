#!/bin/bash
set -e
mkdir -p build
cd build
cmake -DPython3_EXECUTABLE=$(which python3) ..
make
ctest --output-on-failure
