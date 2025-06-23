# Build CPU Kernel tests
include(GoogleTest)
add_executable(test_cpu_kernel test_cpu_kernel.cpp)
target_include_directories(test_cpu_kernel PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_cpu_kernel gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance and consistent float-to-integer casting behavior
target_compile_options(test_cpu_kernel PRIVATE
    -fno-fast-math -ffp-contract=on -fno-rounding-math -ftrapping-math)
gtest_discover_tests(test_cpu_kernel)

# Build Binary Operations tests
add_executable(test_binary_operations test_binary_operations.cpp)
target_include_directories(test_binary_operations PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_binary_operations gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance and consistent float-to-integer casting behavior
target_compile_options(test_binary_operations PRIVATE
    -fno-fast-math -ffp-contract=on -fno-rounding-math -ftrapping-math)
gtest_discover_tests(test_binary_operations)

# Build VecBuffer tests
add_executable(test_vecbuffer test_vecbuffer.cpp)
target_include_directories(test_vecbuffer PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_vecbuffer gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance and consistent float-to-integer casting behavior
target_compile_options(test_vecbuffer PRIVATE
    -fno-fast-math -ffp-contract=on -fno-rounding-math -ftrapping-math)
gtest_discover_tests(test_vecbuffer)

# Build DTypeEnum tests
add_executable(test_dtype_enum test_dtype_enum.cpp)
target_include_directories(test_dtype_enum PRIVATE
                        ${pybind11_SOURCE_DIR}/include
                        ${Python3_INCLUDE_DIRS}
                        ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_dtype_enum gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance and consistent float-to-integer casting behavior
target_compile_options(test_dtype_enum PRIVATE
    -fno-fast-math -ffp-contract=on -fno-rounding-math -ftrapping-math)
gtest_discover_tests(test_dtype_enum)


# Build Buffer member tests
add_executable(test_buffer_members test_buffer_members.cpp)
target_include_directories(test_buffer_members PRIVATE
                        ${pybind11_SOURCE_DIR}/include
                        ${Python3_INCLUDE_DIRS}
                        ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_buffer_members gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
target_compile_options(test_buffer_members PRIVATE -fno-fast-math)
gtest_discover_tests(test_buffer_members)

# Build Alignment tests
add_executable(test_alignment test_alignment.cpp)
target_include_directories(test_alignment PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_alignment gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance in tests
target_compile_options(test_alignment PRIVATE -fno-fast-math)
gtest_discover_tests(test_alignment)

