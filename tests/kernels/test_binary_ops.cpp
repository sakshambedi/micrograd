// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include "../../kernels/operations.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
namespace py = pybind11;

// Test Fixture for Binary Operations
class BinaryOpsTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter();
    }
  }

  void TearDown() override {
    if (Py_IsInitialized()) {
      py::finalize_interpreter();
    }
  }

  // Helper function to generate random values
  template <typename T>
  std::vector<T> generateRandomData(size_t size, T min_val, T max_val) {
    std::vector<T> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dist(min_val, max_val);
      for (auto &val : data) {
        val = dist(gen);
      }
    } else {
      std::uniform_int_distribution<int> dist(static_cast<int>(min_val),
                                              static_cast<int>(max_val));
      for (auto &val : data) {
        val = static_cast<T>(dist(gen));
      }
    }
    return data;
  }

  // Helper to verify addition results
  template <typename T>
  void verifyAddition(const std::vector<T> &a, const std::vector<T> &b,
                      const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_NEAR(result[i], a[i] + b[i],
                  std::numeric_limits<T>::epsilon() * 10)
          << "Failure at index " << i;
    }
  }

  // Helper to verify subtraction results
  template <typename T>
  void verifySubtraction(const std::vector<T> &a, const std::vector<T> &b,
                         const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_NEAR(result[i], a[i] - b[i],
                  std::numeric_limits<T>::epsilon() * 10)
          << "Failure at index " << i;
    }
  }

  // Helper to verify multiplication results
  template <typename T>
  void verifyMultiplication(const std::vector<T> &a, const std::vector<T> &b,
                            const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_NEAR(result[i], a[i] * b[i],
                  std::numeric_limits<T>::epsilon() * 10)
          << "Failure at index " << i;
    }
  }

  // Helper to verify division results
  template <typename T>
  void verifyDivision(const std::vector<T> &a, const std::vector<T> &b,
                      const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      if (b[i] != 0) {
        // Use higher tolerance for division operations for errors that commonly
        // occur during division operations
        EXPECT_NEAR(result[i], a[i] / b[i],
                    std::numeric_limits<T>::epsilon() * 100)
            << "Failure at index " << i;
      }
    }
  }
};

// Test addition operation with different sizes and alignments
TEST_F(BinaryOpsTest, AddFloatVaryingSizes) {
  const size_t sizes[] = {1,  2,  3,  7,   8,   15,  16,  31,
                          32, 63, 64, 100, 128, 500, 1000};

  for (size_t size : sizes) {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);

    verifyAddition(a, b, result);
  }
}

// Test addition with various data types
TEST_F(BinaryOpsTest, AddWithVariousTypes) {
  const size_t size = 128;

  // Test with float
  {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // Test with double
  {
    std::vector<double> a = generateRandomData<double>(size, -100.0, 100.0);
    std::vector<double> b = generateRandomData<double>(size, -100.0, 100.0);
    std::vector<double> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // Test with int32
  {
    std::vector<int32_t> a = generateRandomData<int32_t>(size, -100, 100);
    std::vector<int32_t> b = generateRandomData<int32_t>(size, -100, 100);
    std::vector<int32_t> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // Test with int64
  {
    std::vector<int64_t> a = generateRandomData<int64_t>(size, -100, 100);
    std::vector<int64_t> b = generateRandomData<int64_t>(size, -100, 100);
    std::vector<int64_t> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }
}

// Test addition with misaligned pointers
TEST_F(BinaryOpsTest, AddWithMisalignment) {
  const size_t size = 1000;
  const size_t offsets[] = {1, 2, 3, 4, 7};

  std::vector<float> base_a =
      generateRandomData<float>(size + 8, -100.0f, 100.0f);
  std::vector<float> base_b =
      generateRandomData<float>(size + 8, -100.0f, 100.0f);
  std::vector<float> base_result(size + 8);

  for (size_t offset_a : offsets) {
    for (size_t offset_b : offsets) {
      for (size_t offset_result : offsets) {
        float *a_ptr = base_a.data() + offset_a;
        float *b_ptr = base_b.data() + offset_b;
        float *result_ptr = base_result.data() + offset_result;

        // Extract the actual data for verification
        std::vector<float> a_data(size);
        std::vector<float> b_data(size);

        for (size_t i = 0; i < size; i++) {
          a_data[i] = a_ptr[i];
          b_data[i] = b_ptr[i];
        }

        // Perform addition
        simd_ops::add(a_ptr, b_ptr, result_ptr, size);

        // Copy results for verification
        std::vector<float> result_data(size);
        for (size_t i = 0; i < size; i++) {
          result_data[i] = result_ptr[i];
        }

        verifyAddition(a_data, b_data, result_data);
      }
    }
  }
}

// Test edge cases for addition
TEST_F(BinaryOpsTest, AddEdgeCases) {
  // Test with zeros
  {
    std::vector<float> a(128, 0.0f);
    std::vector<float> b(128, 0.0f);
    std::vector<float> result(128);

    simd_ops::add(a.data(), b.data(), result.data(), 128);

    for (size_t i = 0; i < 128; i++) {
      EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
  }

  // Test with extremely large values
  {
    std::vector<float> a(64, std::numeric_limits<float>::max() / 2);
    std::vector<float> b(64, std::numeric_limits<float>::max() / 2);
    std::vector<float> result(64);

    simd_ops::add(a.data(), b.data(), result.data(), 64);

    for (size_t i = 0; i < 64; i++) {
      // Check the result is very large rather than using strict equality
      // which might fail due to compiler optimizations
      EXPECT_GT(result[i], std::numeric_limits<float>::max() / 3);
    }
  }

  // Test with NaN and Inf
  {
    std::vector<double> a = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(), 1.0};
    std::vector<double> b = {1.0, 1.0, 1.0,
                             std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> result(4);

    simd_ops::add(a.data(), b.data(), result.data(), 4);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isinf(result[1]) && result[1] > 0);
    EXPECT_TRUE(std::isinf(result[2]) && result[2] < 0);
    EXPECT_TRUE(std::isnan(result[3]));
  }
}

// Test performance of addition
TEST_F(BinaryOpsTest, AddPerformance) {
  const size_t size = 10000000; // 10 million elements

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> result(size);

  auto start = std::chrono::high_resolution_clock::now();
  simd_ops::add(a.data(), b.data(), result.data(), size);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time to add " << size << " elements: " << elapsed.count()
            << " seconds" << '\n';

  // Verify a few elements to make sure the computation was correct
  for (size_t i = 0; i < 100; i++) {
    size_t idx = i * (size / 100);
    EXPECT_NEAR(result[idx], a[idx] + b[idx],
                std::numeric_limits<float>::epsilon() * 10);
  }
}

// Test subtraction operation
TEST_F(BinaryOpsTest, SubtractBasic) {
  const size_t size = 128;

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> result(size);

  simd_ops::subtract(a.data(), b.data(), result.data(), size);

  verifySubtraction(a, b, result);
}

// Test multiplication operation
TEST_F(BinaryOpsTest, MultiplyBasic) {
  const size_t size = 128;

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> result(size);

  simd_ops::multiply(a.data(), b.data(), result.data(), size);
  verifyMultiplication(a, b, result);
}

// Test division operation
TEST_F(BinaryOpsTest, DivideBasic) {
  const size_t size = 128;

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b =
      generateRandomData<float>(size, 0.1f, 100.0f); // Avoid division by zero
  std::vector<float> result(size);

  simd_ops::divide(a.data(), b.data(), result.data(), size);

  verifyDivision(a, b, result);
}

// Test buffer level operations
TEST_F(BinaryOpsTest, BufferAddBasic) {
  const size_t size = 128;

  // Fill with random data
  std::vector<float> a_data = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b_data = generateRandomData<float>(size, -100.0f, 100.0f);

  // Create Buffer objects directly with proper size and type
  Buffer a_buf(size, "float32");
  Buffer b_buf(size, "float32");

  // Copy data to Buffers
  auto &a_vec = std::get<VecBuffer<float>>(a_buf.raw());
  auto &b_vec = std::get<VecBuffer<float>>(b_buf.raw());

  for (size_t i = 0; i < size; i++) {
    a_vec[i] = a_data[i];
    b_vec[i] = b_data[i];
  }

  // Perform addition
  Buffer result = simd_ops::buffer_add(a_buf, b_buf, "float32");

  // Verify results
  auto &result_vec = std::get<VecBuffer<float>>(result.raw());

  for (size_t i = 0; i < size; i++) {
    EXPECT_NEAR(result_vec[i], a_data[i] + b_data[i],
                std::numeric_limits<float>::epsilon() * 10);
  }
}

// Test buffer addition with type casting
TEST_F(BinaryOpsTest, BufferAddWithCasting) {
  // Create buffers of different types
  std::vector<float> a_data = generateRandomData<float>(128, -100.0f, 100.0f);
  std::vector<int32_t> b_data = generateRandomData<int32_t>(128, -100, 100);

  // Create Buffer objects directly with proper size and type
  Buffer a_buf(128, "float32");
  Buffer b_buf(128, "int32");

  // Copy data to Buffers
  auto &a_raw = std::get<VecBuffer<float>>(a_buf.raw());
  auto &b_raw = std::get<VecBuffer<int32_t>>(b_buf.raw());

  for (size_t i = 0; i < 128; i++) {
    a_raw[i] = a_data[i];
    b_raw[i] = b_data[i];
  }

  // Perform addition with specified result type
  Buffer result = simd_ops::buffer_add(a_buf, b_buf, "float32");

  // Verify results
  auto &result_vec = std::get<VecBuffer<float>>(result.raw());

  for (size_t i = 0; i < 128; i++) {
    EXPECT_NEAR(result_vec[i], a_data[i] + static_cast<float>(b_data[i]),
                std::numeric_limits<float>::epsilon() * 10);
  }
}

// Test buffer add with different sizes
TEST_F(BinaryOpsTest, BufferAddMismatchedSizes) {
  // Create buffers of different sizes
  std::vector<float> a_data = generateRandomData<float>(128, -100.0f, 100.0f);
  std::vector<float> b_data = generateRandomData<float>(64, -100.0f, 100.0f);

  // Create Buffer objects directly with proper size and type
  Buffer a_buf(128, "float32");
  Buffer b_buf(64, "float32");

  // Copy data to Buffers
  auto &a_raw = std::get<VecBuffer<float>>(a_buf.raw());
  auto &b_raw = std::get<VecBuffer<float>>(b_buf.raw());

  for (size_t i = 0; i < 128; i++) {
    if (i < 64) {
      b_raw[i] = b_data[i];
    }
    a_raw[i] = a_data[i];
  }

  // Addition should fail with different sizes
  EXPECT_THROW(simd_ops::buffer_add(a_buf, b_buf, "float32"),
               std::runtime_error);
}

// Additional testing with half precision
TEST_F(BinaryOpsTest, AddWithHalfPrecision) {
  const size_t size = 64;
  std::vector<half> a(size);
  std::vector<half> b(size);
  std::vector<half> result(size);

  // Fill with some values
  for (size_t i = 0; i < size; i++) {
    a[i] = half(static_cast<float>(i) / 10.0f);
    b[i] = half(static_cast<float>(i) / 5.0f);
  }

  // Perform addition
  simd_ops::add(a.data(), b.data(), result.data(), size);

  // Verify results
  for (size_t i = 0; i < size; i++) {
    float expected =
        static_cast<float>(i) / 10.0f + static_cast<float>(i) / 5.0f;
    float actual = static_cast<float>(result[i]);
    EXPECT_NEAR(actual, expected, 0.02f); // Higher tolerance for half precision
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
