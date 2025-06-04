# Build CPU Kernel tests
add_executable(test_cpu_kernel test_cpu_kernel.cpp)
target_include_directories(test_cpu_kernel PRIVATE ${pybind11_SOURCE_DIR}/include ${Python3_INCLUDE_DIRS})
target_link_libraries(test_cpu_kernel gtest_main cpu_kernel_lib ${Python3_LIBRARIES})
include(GoogleTest)
gtest_discover_tests(test_cpu_kernel)

# Build VecBuffer tests
# add_executable(test_vecbuffer test_vecbuffer.cpp)
# target_include_directories(test_vecbuffer PRIVATE ${pybind11_SOURCE_DIR}/include ${Python3_INCLUDE_DIRS})
# target_link_libraries(test_vecbuffer gtest_main cpu_kernel_lib ${Python3_LIBRARIES})
# gtest_discover_tests(test_vecbuffer)

# # Build DTypeEnum tests
# add_executable(test_dtype_enum test_dtype_enum.cpp)
# target_include_directories(test_dtype_enum PRIVATE ${pybind11_SOURCE_DIR}/include ${Python3_INCLUDE_DIRS})
# target_link_libraries(test_dtype_enum gtest_main cpu_kernel_lib ${Python3_LIBRARIES})
# gtest_discover_tests(test_dtype_enum)
