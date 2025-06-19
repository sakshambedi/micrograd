# Build CPU Kernel tests
include(GoogleTest)
add_executable(test_cpu_kernel test_cpu_kernel.cpp)
target_include_directories(test_cpu_kernel PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_cpu_kernel gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
gtest_discover_tests(test_cpu_kernel)

# Build test_buff_elementwise Tests
# add_executable(test_buffer_elementwise test_buffer_elementwise.cpp)
# target_include_directories(test_buffer_elementwise PRIVATE
#                             ${pybind11_SOURCE_DIR}/include
#                             ${Python3_INCLUDE_DIRS}
#                             ${XSIMD_INCLUDE_DIR})
# target_link_libraries(test_buffer_elementwise gtest_main
#                     cpu_kernel_lib ${Python3_LIBRARIES})
# gtest_discover_tests(test_buffer_elementwise)


# Build VecBuffer Arthimethic Tests
# add_executable(test_vecbuffer_arithmetic test_vecbuffer_arithmetic.cpp)
# target_include_directories(test_vecbuffer_arithmetic PRIVATE
#                             ${pybind11_SOURCE_DIR}/include
#                             ${Python3_INCLUDE_DIRS}
#                             ${XSIMD_INCLUDE_DIR})
# target_link_libraries(test_vecbuffer_arithmetic gtest_main
#                     cpu_kernel_lib ${Python3_LIBRARIES})
# gtest_discover_tests(test_vecbuffer_arithmetic)


# Build VecBuffer tests
add_executable(test_vecbuffer test_vecbuffer.cpp)
target_include_directories(test_vecbuffer PRIVATE
                            ${pybind11_SOURCE_DIR}/include
                            ${Python3_INCLUDE_DIRS}
                            ${XSIMD_INCLUDE_DIR})
target_link_libraries(test_vecbuffer gtest_main
                    cpu_kernel_lib ${Python3_LIBRARIES})
gtest_discover_tests(test_vecbuffer)

# # Build DTypeEnum tests
# add_executable(test_dtype_enum test_dtype_enum.cpp)
# target_include_directories(test_dtype_enum PRIVATE
#                         ${pybind11_SOURCE_DIR}/include
#                         ${Python3_INCLUDE_DIRS}
#                         ${XSIMD_INCLUDE_DIR})
# target_link_libraries(test_dtype_enum gtest_main
#                     cpu_kernel_lib ${Python3_LIBRARIES})
# gtest_discover_tests(test_dtype_enum)
