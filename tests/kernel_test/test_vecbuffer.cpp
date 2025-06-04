#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

// Direct tests for the VecBuffer template class
// Note: These tests access the internal VecBuffer implementation
// through template instantiation

// Test VecBuffer construction and basic operations with float
TEST(VecBufferTest, FloatConstruction) {
  VecBuffer<float> vec(5);
  EXPECT_EQ(vec.size(), 5);
  
  // Test initial values (should be zero-initialized)
  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_FLOAT_EQ(vec[i], 0.0f);
  }
  
  // Test modification of elements
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<float>(i) * 1.5f;
  }
  
  // Verify values
  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_FLOAT_EQ(vec[i], static_cast<float>(i) * 1.5f);
  }
}

// Test VecBuffer construction from raw array
TEST(VecBufferTest, ConstructionFromArray) {
  float data[3] = {1.0f, 2.0f, 3.0f};
  VecBuffer<float> vec(data, 3);
  
  EXPECT_EQ(vec.size(), 3);
  EXPECT_FLOAT_EQ(vec[0], 1.0f);
  EXPECT_FLOAT_EQ(vec[1], 2.0f);
  EXPECT_FLOAT_EQ(vec[2], 3.0f);
}

// Test VecBuffer data access
TEST(VecBufferTest, DataAccess) {
  VecBuffer<float> vec(3);
  vec[0] = 1.5f;
  vec[1] = 2.5f;
  vec[2] = 3.5f;
  
  // Test data() method
  float* raw_data = vec.data();
  EXPECT_FLOAT_EQ(raw_data[0], 1.5f);
  EXPECT_FLOAT_EQ(raw_data[1], 2.5f);
  EXPECT_FLOAT_EQ(raw_data[2], 3.5f);
  
  // Modify through raw pointer
  raw_data[1] = 10.0f;
  EXPECT_FLOAT_EQ(vec[1], 10.0f);
  
  // Test const data access
  const VecBuffer<float>& const_vec = vec;
  const float* const_data = const_vec.data();
  EXPECT_FLOAT_EQ(const_data[0], 1.5f);
  EXPECT_FLOAT_EQ(const_data[1], 10.0f);
  EXPECT_FLOAT_EQ(const_data[2], 3.5f);
}

// Test VecBuffer arithmetic operations
TEST(VecBufferTest, ArithmeticOperations) {
  VecBuffer<float> vec1(3);
  VecBuffer<float> vec2(3);
  
  // Initialize vectors
  for (size_t i = 0; i < 3; ++i) {
    vec1[i] = static_cast<float>(i + 1);
    vec2[i] = static_cast<float>(i * 2);
  }
  
  // Test operator+=
  VecBuffer<float> result1 = vec1;
  result1 += vec2;
  EXPECT_FLOAT_EQ(result1[0], 1.0f + 0.0f);
  EXPECT_FLOAT_EQ(result1[1], 2.0f + 2.0f);
  EXPECT_FLOAT_EQ(result1[2], 3.0f + 4.0f);
  
  // Test operator-=
  VecBuffer<float> result2 = vec1;
  result2 -= vec2;
  EXPECT_FLOAT_EQ(result2[0], 1.0f - 0.0f);
  EXPECT_FLOAT_EQ(result2[1], 2.0f - 2.0f);
  EXPECT_FLOAT_EQ(result2[2], 3.0f - 4.0f);
  
  // Test cwiseMul
  VecBuffer<float> result3 = vec1.cwiseMul(vec2);
  EXPECT_FLOAT_EQ(result3[0], 1.0f * 0.0f);
  EXPECT_FLOAT_EQ(result3[1], 2.0f * 2.0f);
  EXPECT_FLOAT_EQ(result3[2], 3.0f * 4.0f);
  
  // Test dot product
  float dot_result = vec1.dot(vec2);
  EXPECT_FLOAT_EQ(dot_result, 1.0f*0.0f + 2.0f*2.0f + 3.0f*4.0f);
}

// Test VecBuffer with different types
TEST(VecBufferTest, DifferentTypes) {
  // Test with double
  {
    VecBuffer<double> vec(2);
    vec[0] = 1.23456789;
    vec[1] = 9.87654321;
    EXPECT_DOUBLE_EQ(vec[0], 1.23456789);
    EXPECT_DOUBLE_EQ(vec[1], 9.87654321);
  }
  
  // Test with int
  {
    VecBuffer<int> vec(2);
    vec[0] = -42;
    vec[1] = 100;
    EXPECT_EQ(vec[0], -42);
    EXPECT_EQ(vec[1], 100);
  }
  
  // Test with bool
  {
    VecBuffer<bool> vec(2);
    vec[0] = true;
    vec[1] = false;
    EXPECT_TRUE(vec[0]);
    EXPECT_FALSE(vec[1]);
  }
}

// Test zero-size VecBuffer
TEST(VecBufferTest, ZeroSize) {
  VecBuffer<float> vec(0);
  EXPECT_EQ(vec.size(), 0);
}

// Test VecBuffer reference access
TEST(VecBufferTest, ReferenceAccess) {
  VecBuffer<float> vec(3);
  vec[0] = 1.0f;
  vec[1] = 2.0f;
  vec[2] = 3.0f;
  
  // Get mutable reference and modify
  auto ref = vec.ref();
  ref(1) = 10.0f;
  EXPECT_FLOAT_EQ(vec[1], 10.0f);
  
  // Get const reference
  const VecBuffer<float>& const_vec = vec;
  auto const_ref = const_vec.ref();
  EXPECT_FLOAT_EQ(const_ref(0), 1.0f);
  EXPECT_FLOAT_EQ(const_ref(1), 10.0f);
  EXPECT_FLOAT_EQ(const_ref(2), 3.0f);
}

// Test VecBuffer with large arrays
TEST(VecBufferTest, LargeArray) {
  const size_t size = 1000;
  VecBuffer<float> vec(size);
  
  // Fill with values
  for (size_t i = 0; i < size; ++i) {
    vec[i] = static_cast<float>(i);
  }
  
  // Verify
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(vec[i], static_cast<float>(i));
  }
  
  // Test sum of elements
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    sum += vec[i];
  }
  float expected_sum = size * (size - 1) / 2.0f;  // Sum of arithmetic sequence
  EXPECT_FLOAT_EQ(sum, expected_sum);
}

// Test VecBuffer operations with Eigen::half type
TEST(VecBufferTest, HalfPrecision) {
  VecBuffer<Eigen::half> vec(3);
  
  // Set values
  vec[0] = Eigen::half(1.5f);
  vec[1] = Eigen::half(2.5f);
  vec[2] = Eigen::half(3.5f);
  
  // Verify values (accounting for half precision)
  EXPECT_NEAR(static_cast<float>(vec[0]), 1.5f, 0.01f);
  EXPECT_NEAR(static_cast<float>(vec[1]), 2.5f, 0.01f);
  EXPECT_NEAR(static_cast<float>(vec[2]), 3.5f, 0.01f);
}