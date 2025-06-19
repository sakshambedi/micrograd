#!/bin/bash
set -e
mkdir -p build
cd build
cmake -Wno-dev -DPython3_EXECUTABLE=$(which python3) -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make
ctest --output-on-failure
