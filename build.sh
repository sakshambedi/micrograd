#!/usr/bin/env bash

# MicroGrad Build Utility
# =======================
# A single entry-point script for building, testing, cleaning, and
# watching the C++ kernels and Python bindings.
#
#  - Debug build     : ./build.sh build
#  - Release build   : ./build.sh release
#  - Run tests       : ./build.sh test
#  - Clean artifacts : ./build.sh clean
#  - Watch & rebuild : ./build.sh watch
#  - Help            : ./build.sh help
#
# This script supersedes the old dev.sh & run_cpp_tests.sh.

set -euo pipefail

# ---------- styling ----------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${YELLOW}[INFO] $*${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $*${NC}"; }
error()   { echo -e "${RED}[ERROR] $*${NC}" >&2; }

# ---------- sanity checks ----------
if [[ ! -f "CMakeLists.txt" ]]; then
  error "CMakeLists.txt not found. Please run from the project root."
  exit 1
fi

# ---------- helpers ----------
cmake_configure() {
  local build_type="$1" # Debug / Release / RelWithDebInfo / MinSizeRel
  mkdir -p build
  pushd build >/dev/null
  info "Configuring CMake (type=${build_type})"
  cmake -DCMAKE_BUILD_TYPE="${build_type}" \
        -DPython3_EXECUTABLE="$(which python3)" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ..
  popd >/dev/null
}

cmake_build() {
  local parallel_jobs
  # GNU & BSD uname variants differ; handle both
  parallel_jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  info "Building (${parallel_jobs} jobs)"
  cmake --build build -- -j"${parallel_jobs}"
}

cmake_test() {
  pushd build >/dev/null
  info "Running CTest suite"
  ctest --output-on-failure
  popd >/dev/null
}

cmake_clean() {
  info "Removing build directory"
  rm -rf build
  success "Clean completed"
}

python_smoke_test() {
  info "Importing compiled python module (smoke test)"
  python3 - <<'PY'
import importlib, sys
try:
    cpu_kernel = importlib.import_module('cpu_kernel')
    buf = cpu_kernel.Buffer([1,2,3], 'float32')
    print('✓ Python module import successful →', buf)
except Exception as e:
    print('⚠️  Python smoke test failed:', e, file=sys.stderr)
    sys.exit(1)
PY
  success "Python smoke test passed"
}

watch_sources() {
  info "Watching sources; press Ctrl+C to exit"
  if command -v fswatch >/dev/null 2>&1; then
    fswatch -o kernels/ tests/kernels/ | while read -r _; do
      info "Change detected → rebuilding"
      cmake_configure Debug
      cmake_build
    done
  elif command -v inotifywait >/dev/null 2>&1; then
    while inotifywait -r -e modify kernels/ tests/kernels/; do
      info "Change detected → rebuilding"
      cmake_configure Debug
      cmake_build
    done
  else
    error "No watcher (fswatch or inotifywait) available."
    exit 1
  fi
}

# ---------- dispatch ----------
cmd="${1:-help}"
# Backward-compatibility for old flag style (e.g. --debug)
case "${cmd}" in
  --debug)  cmd="build"   ;;
  --clean)  cmd="clean"   ;;
  --tests)  cmd="test"    ;;
  --test)   cmd="test"    ;;
  --release) cmd="release" ;;
  --help|-h) cmd="help"    ;;
esac

case "${cmd}" in
  build)
    cmake_configure Debug
    cmake_build
    success "Debug build completed"
    ;;
  release)
    cmake_configure Release
    cmake_build
    success "Release build completed"
    ;;
  test)
    cmake_configure Debug
    cmake_build
    cmake_test
    success "All tests passed"
    ;;
  clean)
    cmake_clean
    ;;
  python)
    cmake_configure Debug
    cmake_build
    python_smoke_test
    ;;
  watch)
    cmake_configure Debug
    cmake_build
    watch_sources
    ;;
  help|--help|-h)
    cat <<EOF
MicroGrad Build Utility
----------------------
Commands:
  build      – Build (Debug)
  release    – Build (Release)
  test       – Build + run C++ tests (Debug)
  clean      – Remove build artifacts
  watch      – Rebuild automatically on source changes (Debug)
  python     – Build + quick Python smoke test
  help       – Show this help text
EOF
    ;;
  *)
    error "Unknown command: ${cmd}"
    echo "Use ./build.sh help for usage instructions."
    exit 1
    ;;
esac
