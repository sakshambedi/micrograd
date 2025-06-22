# Build CPU Kernel tests
include(GoogleTest)
add_executable(test_cpu_kernel test_cpu_kernel.cpp)
target_include_directories(test_cpu_kernel PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_cpu_kernel gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance in tests
target_compile_options(test_cpu_kernel PRIVATE -fno-fast-math)
gtest_discover_tests(test_cpu_kernel)

# Build Binary Operations tests
add_executable(test_binary_ops test_binary_ops.cpp)
target_include_directories(test_binary_ops PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_binary_ops gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance in tests
target_compile_options(test_binary_ops PRIVATE -fno-fast-math)
gtest_discover_tests(test_binary_ops)

# Build Elementwise Add tests
add_executable(test_elementwise_add test_elementwise_add.cpp)
target_include_directories(test_elementwise_add PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_elementwise_add gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance in tests
target_compile_options(test_elementwise_add PRIVATE -fno-fast-math)
gtest_discover_tests(test_elementwise_add)

# Build VecBuffer tests
add_executable(test_vecbuffer test_vecbuffer.cpp)
target_include_directories(test_vecbuffer PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_vecbuffer gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance in tests
target_compile_options(test_vecbuffer PRIVATE -fno-fast-math)
gtest_discover_tests(test_vecbuffer)

# Build DTypeEnum tests
add_executable(test_dtype_enum test_dtype_enum.cpp)
target_include_directories(test_dtype_enum PRIVATE
                        ${pybind11_SOURCE_DIR}/include
                        ${Python3_INCLUDE_DIRS}
                        ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_dtype_enum gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
# Ensure IEEE 754 compliance in tests
target_compile_options(test_dtype_enum PRIVATE -fno-fast-math)
gtest_discover_tests(test_dtype_enum)
