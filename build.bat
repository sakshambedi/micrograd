@echo off
setlocal enabledelayedexpansion

REM MicroGrad C++ Build Script for Windows
REM This script builds the C++ kernel library and Python bindings

set "BUILD_TYPE=Release"
set "BUILD_DIR=build"
set "CLEAN_BUILD=false"
set "RUN_TESTS=false"
set "INSTALL_DEPS=false"
set "VERBOSE=false"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--debug" (
    set "BUILD_TYPE=Debug"
    shift
    goto :parse_args
)
if "%~1"=="--clean" (
    set "CLEAN_BUILD=true"
    shift
    goto :parse_args
)
if "%~1"=="--tests" (
    set "RUN_TESTS=true"
    shift
    goto :parse_args
)
if "%~1"=="--install-deps" (
    set "INSTALL_DEPS=true"
    shift
    goto :parse_args
)
if "%~1"=="--verbose" (
    set "VERBOSE=true"
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --debug         Build in Debug mode ^(default: Release^)
    echo   --clean         Clean build directory before building
    echo   --tests         Run tests after building
    echo   --install-deps  Install dependencies ^(requires vcpkg^)
    echo   --verbose       Enable verbose output
    echo   --help          Show this help message
    echo.
    echo Examples:
    echo   %0                    # Standard release build
    echo   %0 --debug --tests    # Debug build with tests
    echo   %0 --clean --verbose  # Clean build with verbose output
    exit /b 0
)
if "%~1"=="-h" (
    goto :parse_args
)
echo [ERROR] Unknown option: %~1
echo Use --help for usage information
exit /b 1

:end_parse

REM Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found. Please run this script from the project root.
    exit /b 1
)

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is required but not found. Please install Python.
    exit /b 1
)

for /f "tokens=*" %%i in ('where python') do set "PYTHON_EXECUTABLE=%%i"
echo [INFO] Using Python: !PYTHON_EXECUTABLE!

REM Install dependencies if requested
if "!INSTALL_DEPS!"=="true" (
    echo [INFO] Installing dependencies...

    REM Check if vcpkg is available
    vcpkg --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] vcpkg is required for dependency installation. Please install vcpkg first.
        echo [INFO] Visit: https://github.com/microsoft/vcpkg#quick-start-windows
        exit /b 1
    )

    REM Install required packages
    vcpkg install eigen3 xsimd pybind11
    echo [SUCCESS] Dependencies installed successfully
)

REM Clean build directory if requested
if "!CLEAN_BUILD!"=="true" (
    echo [INFO] Cleaning build directory...
    if exist "!BUILD_DIR!" rmdir /s /q "!BUILD_DIR!"
)

REM Create build directory
echo [INFO] Creating build directory: !BUILD_DIR!
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd "!BUILD_DIR!"

REM Configure CMake
echo [INFO] Configuring CMake with build type: !BUILD_TYPE!
set "CMAKE_ARGS=-DCMAKE_BUILD_TYPE=!BUILD_TYPE! -DPython3_EXECUTABLE=!PYTHON_EXECUTABLE! -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

if "!VERBOSE!"=="true" (
    set "CMAKE_ARGS=!CMAKE_ARGS! --verbose"
)

cmake !CMAKE_ARGS! ..
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    exit /b 1
)

REM Build the project
echo [INFO] Building project...
set "MAKE_ARGS="

if "!VERBOSE!"=="true" (
    set "MAKE_ARGS=VERBOSE=1"
)

cmake --build . --config !BUILD_TYPE! !MAKE_ARGS!
if errorlevel 1 (
    echo [ERROR] Build failed
    exit /b 1
)

echo [SUCCESS] Build completed successfully!

REM Run tests if requested
if "!RUN_TESTS!"=="true" (
    echo [INFO] Running tests...
    ctest --output-on-failure
    if errorlevel 1 (
        echo [ERROR] Tests failed
        exit /b 1
    )
    echo [SUCCESS] All tests passed!
)

REM Show build artifacts
echo [INFO] Build artifacts:
dir *.dll 2>nul || echo [WARNING] No shared libraries found
dir test_*.exe 2>nul || echo [WARNING] No test executables found

echo [SUCCESS] Build process completed!
echo [INFO] You can now import the cpu_kernel module in Python:
echo   import cpu_kernel
echo   buffer = cpu_kernel.Buffer([1, 2, 3], 'float32')
