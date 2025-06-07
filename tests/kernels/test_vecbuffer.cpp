// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>

// Tests basic VecBuffer construction and element access
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

// Tests construction from existing array
TEST(VecBufferTest, ConstructionFromArray) {
  std::array<float, 3> data = {1.0f, 2.0f, 3.0f};
  VecBuffer<float> vec(data.data(), 3);

  EXPECT_EQ(vec.size(), 3);
  EXPECT_FLOAT_EQ(vec[0], 1.0f);
  EXPECT_FLOAT_EQ(vec[1], 2.0f);
  EXPECT_FLOAT_EQ(vec[2], 3.0f);
}

// Tests raw data access and modification
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

// Tests vector arithmetic operations (+= -= cwiseMul dot)
TEST(VecBufferTest, ArithmeticOperations) {
  VecBuffer<float> vec1(3);
  VecBuffer<float> vec2(3);

  for (size_t i = 0; i < 3; ++i) {
    vec1[i] = static_cast<float>(i + 1);
    vec2[i] = static_cast<float>(i * 2);
  }

  VecBuffer<float> result1 = vec1;
  result1 += vec2;
  EXPECT_FLOAT_EQ(result1[0], 1.0f + 0.0f);
  EXPECT_FLOAT_EQ(result1[1], 2.0f + 2.0f);
  EXPECT_FLOAT_EQ(result1[2], 3.0f + 4.0f);

  VecBuffer<float> result2 = vec1;
  result2 -= vec2;
  EXPECT_FLOAT_EQ(result2[0], 1.0f - 0.0f);
  EXPECT_FLOAT_EQ(result2[1], 2.0f - 2.0f);
  EXPECT_FLOAT_EQ(result2[2], 3.0f - 4.0f);

  VecBuffer<float> result3 = vec1.cwiseMul(vec2);
  EXPECT_FLOAT_EQ(result3[0], 1.0f * 0.0f);
  EXPECT_FLOAT_EQ(result3[1], 2.0f * 2.0f);
  EXPECT_FLOAT_EQ(result3[2], 3.0f * 4.0f);

  float dot_result = vec1.dot(vec2);
  EXPECT_FLOAT_EQ(dot_result, 1.0f * 0.0f + 2.0f * 2.0f + 3.0f * 4.0f);
}

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

  {
    VecBuffer<bool> vec(2);
    vec[0] = true;
    vec[1] = false;
    EXPECT_TRUE(vec[0]);
    EXPECT_FALSE(vec[1]);
  }
}

// Tests empty vector behavior
TEST(VecBufferTest, ZeroSize) {
  VecBuffer<float> vec(0);
  EXPECT_EQ(vec.size(), 0);
}

// Tests reference-based access functionality
TEST(VecBufferTest, ReferenceAccess) {
  VecBuffer<float> vec(3);
  vec[0] = 1.0f;
  vec[1] = 2.0f;
  vec[2] = 3.0f;

  auto ref = vec.ref();
  ref(1) = 10.0f;
  EXPECT_FLOAT_EQ(vec[1], 10.0f);

  const VecBuffer<float> &const_vec = vec;
  auto const_ref = const_vec.ref();
  EXPECT_FLOAT_EQ(const_ref(0), 1.0f);
  EXPECT_FLOAT_EQ(const_ref(1), 10.0f);
  EXPECT_FLOAT_EQ(const_ref(2), 3.0f);
}

// Tests VecBuffer with large data sets
TEST(VecBufferTest, LargeArray) {
  const size_t size = 1000;
  VecBuffer<float> vec(size);

  for (size_t i = 0; i < size; ++i) {
    vec[i] = static_cast<float>(i);
  }

  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(vec[i], static_cast<float>(i));
  }

  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    sum += vec[i];
  }
  float expected_sum = size * (size - 1) / 2.0f;
  EXPECT_FLOAT_EQ(sum, expected_sum);
}

// Tests VecBuffer operations with half-precision floating point
TEST(VecBufferTest, HalfPrecision) {
  VecBuffer<Eigen::half> vec(3);

  vec[0] = Eigen::half(1.5f);
  vec[1] = Eigen::half(2.5f);
  vec[2] = Eigen::half(3.5f);

  EXPECT_NEAR(static_cast<float>(vec[0]), 1.5f, 0.01f);
  EXPECT_NEAR(static_cast<float>(vec[1]), 2.5f, 0.01f);
  EXPECT_NEAR(static_cast<float>(vec[2]), 3.5f, 0.01f);
}
