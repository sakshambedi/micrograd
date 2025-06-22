// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "../../kernels/vecbuffer.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

TEST(VecBufferTest, FloatConstruction) {
  VecBuffer<float> vec(5);
  EXPECT_EQ(vec.size(), 5);

  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_FLOAT_EQ(vec[i], 0.0f);
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<float>(i) * 1.5f;
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_FLOAT_EQ(vec[i], static_cast<float>(i) * 1.5f);
  }
}

TEST(VecBufferTest, ConstructionFromArray) {
  std::array<float, 3> data = {1.0f, 2.0f, 3.0f};
  VecBuffer<float> vec(data.data(), 3);

  EXPECT_EQ(vec.size(), 3);
  EXPECT_FLOAT_EQ(vec[0], 1.0f);
  EXPECT_FLOAT_EQ(vec[1], 2.0f);
  EXPECT_FLOAT_EQ(vec[2], 3.0f);
}

TEST(VecBufferTest, DataAccess) {
  VecBuffer<float> vec(3);
  vec[0] = 1.5f;
  vec[1] = 2.5f;
  vec[2] = 3.5f;

  float *raw_data = vec.data();
  EXPECT_FLOAT_EQ(raw_data[0], 1.5f);
  EXPECT_FLOAT_EQ(raw_data[1], 2.5f);
  EXPECT_FLOAT_EQ(raw_data[2], 3.5f);

  raw_data[1] = 10.0f;
  EXPECT_FLOAT_EQ(vec[1], 10.0f);

  const VecBuffer<float> &const_vec = vec;
  const float *const_data = const_vec.data();
  EXPECT_FLOAT_EQ(const_data[0], 1.5f);
  EXPECT_FLOAT_EQ(const_data[1], 10.0f);
  EXPECT_FLOAT_EQ(const_data[2], 3.5f);
}

// Tests binary kernel operations with Addition
// TEST(VecBufferTest, AdditionKernel) {
//   constexpr size_t size = 16; // Ensure size is enough to test SIMD path
//   VecBuffer<float> vec1(size);
//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<float>(i) * 1.5f;
//     vec2[i] = static_cast<float>(i) * 0.5f;
//   }

//   // Apply binary_kernel with AddOp
//   binary_kernel<float, AddOp>(vec1.data(), vec2.data(), result.data(), size);

//   // Verify results
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] + vec2[i]);
//   }
// }

// Tests binary kernel operations with Subtraction
// TEST(VecBufferTest, SubtractionKernel) {
//   constexpr size_t size = 16; // Ensure size is enough to test SIMD path
//   VecBuffer<float> vec1(size);
//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<float>(i) * 1.5f;
//     vec2[i] = static_cast<float>(i) * 0.5f;
//   }

//   // Apply binary_kernel with SubOp
//   binary_kernel<float, SubOp>(vec1.data(), vec2.data(), result.data(), size);

//   // Verify results
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] - vec2[i]);
//   }
// }

// Tests binary kernel operations with Multiplication
// TEST(VecBufferTest, MultiplicationKernel) {
//   constexpr size_t size = 16; // Ensure size is enough to test SIMD path
//   VecBuffer<float> vec1(size);
//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<float>(i) * 1.5f;
//     vec2[i] = static_cast<float>(i) * 0.5f;
//   }

//   // Apply binary_kernel with MulOp
//   binary_kernel<float, MulOp>(vec1.data(), vec2.data(), result.data(), size);

//   // Verify results
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] * vec2[i]);
//   }
// }

// Tests binary kernel operations with Division
// TEST(VecBufferTest, DivisionKernel) {
//   constexpr size_t size = 16; // Ensure size is enough to test SIMD path
//   VecBuffer<float> vec1(size);
//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<float>(i + 1) * 1.5f; // Avoid division by zero
//     vec2[i] = static_cast<float>(i + 1) * 0.5f; // Avoid division by zero
//   }

//   // Apply binary_kernel with DivOp
//   binary_kernel<float, DivOp>(vec1.data(), vec2.data(), result.data(), size);

//   // Verify results
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] / vec2[i]);
//   }
// }

// Tests binary kernels with non-SIMD size to ensure scalar fallback works
// TEST(VecBufferTest, NonSIMDSizeOperations) {
//   constexpr size_t size = 7; // Non-multiple of most SIMD widths
//   VecBuffer<float> vec1(size);
//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<float>(i + 1) * 1.5f;
//     vec2[i] = static_cast<float>(i + 1) * 0.5f;
//   }

//   // Test all operations
//   binary_kernel<float, AddOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] + vec2[i]);
//   }

//   binary_kernel<float, SubOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] - vec2[i]);
//   }

//   binary_kernel<float, MulOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] * vec2[i]);
//   }

//   binary_kernel<float, DivOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(result[i], vec1[i] / vec2[i]);
//   }
// }

// Tests VecBuffer with various data types (double, int, bool)
TEST(VecBufferTest, DifferentTypes) {
  {
    VecBuffer<double> vec(2);
    vec[0] = 1.23456789;
    vec[1] = 9.87654321;
    EXPECT_DOUBLE_EQ(vec[0], 1.23456789);
    EXPECT_DOUBLE_EQ(vec[1], 9.87654321);
  }

  {
    VecBuffer<int> vec(2);
    vec[0] = -42;
    vec[1] = 100;
    EXPECT_EQ(vec[0], -42);
    EXPECT_EQ(vec[1], 100);
  }
}

// Tests binary kernels with double precision
// TEST(VecBufferTest, DoublePrecisionOperations) {
//   constexpr size_t size = 16;
//   VecBuffer<double> vec1(size);
//   VecBuffer<double> vec2(size);
//   VecBuffer<double> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<double>(i + 1) * 1.5;
//     vec2[i] = static_cast<double>(i + 1) * 0.5;
//   }

//   binary_kernel<double, AddOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_DOUBLE_EQ(result[i], vec1[i] + vec2[i]);
//   }

//   binary_kernel<double, SubOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_DOUBLE_EQ(result[i], vec1[i] - vec2[i]);
//   }

//   binary_kernel<double, MulOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_DOUBLE_EQ(result[i], vec1[i] * vec2[i]);
//   }

//   binary_kernel<double, DivOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_DOUBLE_EQ(result[i], vec1[i] / vec2[i]);
//   }
// }

// Tests binary kernels with integer types
// TEST(VecBufferTest, IntegerOperations) {
//   constexpr size_t size = 16;
//   VecBuffer<int32_t> vec1(size);
//   VecBuffer<int32_t> vec2(size);
//   VecBuffer<int32_t> result(size);

//   // Initialize test data
//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = static_cast<int32_t>(i + 1) * 3;
//     vec2[i] = static_cast<int32_t>(i + 1);
//   }

//   binary_kernel<int32_t, AddOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_EQ(result[i], vec1[i] + vec2[i]);
//   }

//   binary_kernel<int32_t, SubOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_EQ(result[i], vec1[i] - vec2[i]);
//   }

//   binary_kernel<int32_t, MulOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_EQ(result[i], vec1[i] * vec2[i]);
//   }

//   binary_kernel<int32_t, DivOp>(vec1.data(), vec2.data(), result.data(),
//   size); for (size_t i = 0; i < size; ++i) {
//     EXPECT_EQ(result[i], vec1[i] / vec2[i]);
//   }
// }

// Tests empty vector behavior
TEST(VecBufferTest, ZeroSize) {
  VecBuffer<float> vec(0);
  EXPECT_EQ(vec.size(), 0);

  EXPECT_NE(vec.data(), nullptr);
}

// TEST(VecBufferTest, HalfPrecision) {
//   constexpr size_t size = 8;
//   VecBuffer<Eigen::half> vec1(size);
//   VecBuffer<Eigen::half> vec2(size);
//   VecBuffer<Eigen::half> result(size);

//   for (size_t i = 0; i < size; ++i) {
//     vec1[i] = Eigen::half(static_cast<float>(i + 1) * 1.5f);
//     vec2[i] = Eigen::half(static_cast<float>(i + 1) * 0.5f);
//   }

//   binary_kernel<AddOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     // Need to convert to float for comparison due to half precision
//     limitations float expected = static_cast<float>(vec1[i]) +
//     static_cast<float>(vec2[i]); auto actual = static_cast<float>(result[i]);
//     EXPECT_NEAR(actual, expected, 0.01f);
//   }

//   binary_kernel<SubOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     float expected = static_cast<float>(vec1[i]) -
//     static_cast<float>(vec2[i]); auto actual = static_cast<float>(result[i]);
//     EXPECT_NEAR(actual, expected, 0.01f);
//   }

//   binary_kernel<MulOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     float expected = static_cast<float>(vec1[i]) *
//     static_cast<float>(vec2[i]); auto actual = static_cast<float>(result[i]);
//     EXPECT_NEAR(actual, expected, 0.01f);
//   }

//   binary_kernel<DivOp>(vec1.data(), vec2.data(), result.data(), size);
//   for (size_t i = 0; i < size; ++i) {
//     float expected = static_cast<float>(vec1[i]) /
//     static_cast<float>(vec2[i]); auto actual = static_cast<float>(result[i]);
//     EXPECT_NEAR(actual, expected, 0.01f);
//   }
// }

// TEST(VecBufferTest, LargeArray) {
//   const size_t size = 1000;
//   VecBuffer<float> vec(size);

//   for (size_t i = 0; i < size; ++i) {
//     vec[i] = static_cast<float>(i);
//   }

//   for (size_t i = 0; i < size; ++i) {
//     EXPECT_FLOAT_EQ(vec[i], static_cast<float>(i));
//   }

//   float sum = 0.0f;
//   for (size_t i = 0; i < size; ++i) {
//     sum += vec[i];
//   }
//   float expected_sum = size * (size - 1) / 2.0f;
//   EXPECT_FLOAT_EQ(sum, expected_sum);

//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   for (size_t i = 0; i < size; ++i) {
//     vec2[i] = static_cast<float>(i) * 0.5f;
//   }

//   binary_kernel<float, AddOp>(vec.data(), vec2.data(), result.data(), size);

//   for (size_t i = 0; i < size; i += 100) {
//     EXPECT_FLOAT_EQ(result[i], vec[i] + vec2[i]);
//   }
// }

// TEST(VecBufferTest, SmallArray) {
//   const size_t size = 1;
//   VecBuffer<float> vec1(size);
//   VecBuffer<float> vec2(size);
//   VecBuffer<float> result(size);

//   vec1[0] = 42.0f;
//   vec2[0] = 24.0f;

//   binary_kernel<float, AddOp>(vec1.data(), vec2.data(), result.data(), size);
//   EXPECT_FLOAT_EQ(result[0], 66.0f);

//   binary_kernel<float, SubOp>(vec1.data(), vec2.data(), result.data(), size);
//   EXPECT_FLOAT_EQ(result[0], 18.0f);

//   binary_kernel<float, MulOp>(vec1.data(), vec2.data(), result.data(), size);
//   EXPECT_FLOAT_EQ(result[0], 1008.0f);

//   binary_kernel<float, DivOp>(vec1.data(), vec2.data(), result.data(), size);
//   EXPECT_FLOAT_EQ(result[0], 1.75f);
// }

TEST(VecBufferTest, MemoryAlignment) {
  constexpr size_t size = 16;
  VecBuffer<float> vec(size);

  // Check if the pointer is aligned
  auto addr = reinterpret_cast<uintptr_t>(vec.data());
  constexpr size_t alignment = 16; // Common SIMD alignment requirement
  EXPECT_EQ(addr % alignment, 0)
      << "Memory not aligned to " << alignment << " bytes";
}
