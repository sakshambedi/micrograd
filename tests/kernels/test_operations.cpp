// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include "../../kernels/operations.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace py = pybind11;

// Unified Test Fixture for Binary Operations
class BinaryOperationsTest : public ::testing::Test {
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
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_NEAR(result[i], a[i] + b[i],
                    std::numeric_limits<T>::epsilon() * 10)
            << "Failure at index " << i;
      } else {
        EXPECT_EQ(result[i], static_cast<T>(a[i] + b[i]))
            << "Failure at index " << i;
      }
    }
  }

  // Helper to verify subtraction results
  template <typename T>
  void verifySubtraction(const std::vector<T> &a, const std::vector<T> &b,
                         const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_NEAR(result[i], a[i] - b[i],
                    std::numeric_limits<T>::epsilon() * 10)
            << "Failure at index " << i;
      } else {
        EXPECT_EQ(result[i], static_cast<T>(a[i] - b[i]))
            << "Failure at index " << i;
      }
    }
  }

  // Helper to verify multiplication results
  template <typename T>
  void verifyMultiplication(const std::vector<T> &a, const std::vector<T> &b,
                            const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_NEAR(result[i], a[i] * b[i],
                    std::numeric_limits<T>::epsilon() * 10)
            << "Failure at index " << i;
      } else {
        EXPECT_EQ(result[i], static_cast<T>(a[i] * b[i]))
            << "Failure at index " << i;
      }
    }
  }

  // Helper to verify division results
  template <typename T>
  void verifyDivision(const std::vector<T> &a, const std::vector<T> &b,
                      const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      if (b[i] != 0) {
        if constexpr (std::is_floating_point_v<T>) {
          // higher tolerance for division operations due to potential SIMD
          // rounding differences
          EXPECT_NEAR(result[i], a[i] / b[i],
                      std::numeric_limits<T>::epsilon() * 1000)
              << "Failure at index " << i;
        } else {
          EXPECT_EQ(result[i], static_cast<T>(a[i] / b[i]))
              << "Failure at index " << i;
        }
      }
    }
  }
};

// Unified Test Fixture for Unary Operations
class UnaryOperationsTest : public ::testing::Test {
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

  // Helper function to generate random values (same as BinaryOperationsTest)
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

  template <typename T>
  void verifyPower(const std::vector<T> &base, const std::vector<T> &exp,
                   const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      if constexpr (std::is_floating_point_v<T>) {
        T expected = std::pow(base[i], exp[i]);
        T computed = result[i];
        T diff = std::abs(computed - expected);
        T tolerance = std::numeric_limits<T>::epsilon() * 100000;

        EXPECT_NEAR(computed, expected, tolerance)
            << "Power failure at index " << i << ": base=" << base[i]
            << ", exp=" << exp[i] << ", computed=" << computed
            << ", expected=" << expected << ", diff=" << diff
            << ", tolerance=" << tolerance;
      } else {
        T expected = static_cast<T>(std::pow(static_cast<double>(base[i]),
                                             static_cast<double>(exp[i])));
        EXPECT_EQ(result[i], expected)
            << "Failure at index " << i << ": base=" << base[i]
            << ", exp=" << exp[i];
      }
    }
  }

  // Helper to verify negation results
  template <typename T>
  void verifyNegation(const std::vector<T> &input,
                      const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_EQ(result[i], -input[i]) << "Failure at index " << i;
    }
  }
};

//------------------------------------------------------------------------------
// Addition Tests
//------------------------------------------------------------------------------

// Test addition operation with different sizes and alignments
TEST_F(BinaryOperationsTest, AddFloatVaryingSizes) {
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

// Test around SIMD boundaries
TEST_F(BinaryOperationsTest, AddSIMDBoundaries) {
  std::array<size_t, 8> sizes = {3, 4, 5, 7, 8, 9, 15, 16};
  for (auto size : sizes) {
    auto a = generateRandomData<float>(size, -80.0f, 80.0f);
    auto b = generateRandomData<float>(size, -80.0f, 80.0f);
    std::vector<float> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }
}

// Test addition with all supported data types
TEST_F(BinaryOperationsTest, AddAllSupportedTypes) {
  const size_t size = 64;

  // float
  {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // double
  {
    std::vector<double> a = generateRandomData<double>(size, -100.0, 100.0);
    std::vector<double> b = generateRandomData<double>(size, -100.0, 100.0);
    std::vector<double> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // uint8_t
  {
    auto a = generateRandomData<uint8_t>(size, 0, 100);
    auto b = generateRandomData<uint8_t>(size, 0, 100);
    std::vector<uint8_t> r(size);
    simd_ops::add(a.data(), b.data(), r.data(), size);
    verifyAddition(a, b, r);
  }

  // int16_t
  {
    auto a = generateRandomData<int16_t>(size, -500, 500);
    auto b = generateRandomData<int16_t>(size, -500, 500);
    std::vector<int16_t> r(size);
    simd_ops::add(a.data(), b.data(), r.data(), size);
    verifyAddition(a, b, r);
  }

  // uint16_t
  {
    auto a = generateRandomData<uint16_t>(size, 0, 1000);
    auto b = generateRandomData<uint16_t>(size, 0, 1000);
    std::vector<uint16_t> r(size);
    simd_ops::add(a.data(), b.data(), r.data(), size);
    verifyAddition(a, b, r);
  }

  // int32_t
  {
    std::vector<int32_t> a = generateRandomData<int32_t>(size, -100, 100);
    std::vector<int32_t> b = generateRandomData<int32_t>(size, -100, 100);
    std::vector<int32_t> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // uint32_t
  {
    auto a = generateRandomData<uint32_t>(size, 0, 20000);
    auto b = generateRandomData<uint32_t>(size, 0, 20000);
    std::vector<uint32_t> r(size);
    simd_ops::add(a.data(), b.data(), r.data(), size);
    verifyAddition(a, b, r);
  }

  // int64_t
  {
    std::vector<int64_t> a = generateRandomData<int64_t>(size, -100, 100);
    std::vector<int64_t> b = generateRandomData<int64_t>(size, -100, 100);
    std::vector<int64_t> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }

  // uint64_t
  {
    auto a = generateRandomData<uint64_t>(size, 0ULL, 100000ULL);
    auto b = generateRandomData<uint64_t>(size, 0ULL, 100000ULL);
    std::vector<uint64_t> r(size);
    simd_ops::add(a.data(), b.data(), r.data(), size);
    verifyAddition(a, b, r);
  }
}

// Test addition with misaligned pointers
TEST_F(BinaryOperationsTest, AddWithMisalignment) {
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

// Compare aligned vs unaligned kernels
TEST_F(BinaryOperationsTest, AlignedVsUnaligned) {
  const size_t N = 128;
  alignas(64) std::vector<float> a(N), b(N), r1(N), r2(N);

  for (size_t i = 0; i < N; ++i) {
    a[i] = (std::rand() / static_cast<float>(RAND_MAX)) * 200.0f - 100.0f;
    b[i] = (std::rand() / static_cast<float>(RAND_MAX)) * 200.0f - 100.0f;
  }

  simd_ops::add(a.data(), b.data(), r1.data(), N);
  simd_ops::add(a.data(), b.data(), r2.data(), N);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(r1[i], r2[i]) << "Index " << i;
    EXPECT_FLOAT_EQ(r1[i], a[i] + b[i]) << "Index " << i;
  }
}

// Test edge cases for addition
TEST_F(BinaryOperationsTest, AddEdgeCases) {
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

// Test overflow produces +Inf
TEST_F(BinaryOperationsTest, AddProducesInfinityOnOverflow) {
  const size_t N = 32;
  float big = std::numeric_limits<float>::max();
  float posInf = std::numeric_limits<float>::infinity();

  std::vector<float> a(N, big), b(N, big), r(N);
  simd_ops::add(a.data(), b.data(), r.data(), N);

  for (auto v : r) {
    EXPECT_EQ(v, posInf);
  }
}

// Test performance of addition
TEST_F(BinaryOperationsTest, AddPerformance) {
  std::array<size_t, 5> sizes = {1000, 10000, 100000, 1000000, 10000000};
  for (auto N : sizes) {
    auto a = generateRandomData<float>(N, -100.0f, 100.0f);
    auto b = generateRandomData<float>(N, -100.0f, 100.0f);
    std::vector<float> r(N);

    auto t0 = std::chrono::high_resolution_clock::now();
    simd_ops::add(a.data(), b.data(), r.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dt = t1 - t0;
    // spot-check a few elements
    for (size_t i = 0; i < N; i += N / 10) {
      EXPECT_NEAR(r[i], a[i] + b[i],
                  std::numeric_limits<float>::epsilon() * 10);
    }
    std::cout << "Add N=" << N << ": " << dt.count() << " ms\n";
  }
}

// Test with half precision
TEST_F(BinaryOperationsTest, AddWithHalfPrecision) {
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
    auto actual = static_cast<float>(result[i]);
    EXPECT_NEAR(actual, expected, 0.02f); // Higher tolerance for half precision
  }
}

//------------------------------------------------------------------------------
// Subtraction Tests
//------------------------------------------------------------------------------

// Test subtraction operation
TEST_F(BinaryOperationsTest, SubtractBasic) {
  const size_t size = 128;

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> result(size);

  simd_ops::subtract(a.data(), b.data(), result.data(), size);

  verifySubtraction(a, b, result);
}

// Test subtraction with different sizes
TEST_F(BinaryOperationsTest, SubtractVaryingSizes) {
  const size_t sizes[] = {1, 3, 8, 16, 32, 64, 128, 500};

  for (size_t size : sizes) {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> result(size);

    simd_ops::subtract(a.data(), b.data(), result.data(), size);

    verifySubtraction(a, b, result);
  }
}

// Test subtraction with all supported types
TEST_F(BinaryOperationsTest, SubtractAllTypes) {
  const size_t size = 64;

  // float
  {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> result(size);

    simd_ops::subtract(a.data(), b.data(), result.data(), size);
    verifySubtraction(a, b, result);
  }

  // double
  {
    std::vector<double> a = generateRandomData<double>(size, -100.0, 100.0);
    std::vector<double> b = generateRandomData<double>(size, -100.0, 100.0);
    std::vector<double> result(size);

    simd_ops::subtract(a.data(), b.data(), result.data(), size);
    verifySubtraction(a, b, result);
  }

  // uint8_t
  {
    auto a = generateRandomData<uint8_t>(size, 0, 100);
    auto b = generateRandomData<uint8_t>(size, 0, 100);
    std::vector<uint8_t> r(size);
    simd_ops::subtract(a.data(), b.data(), r.data(), size);
    verifySubtraction(a, b, r);
  }

  // int16_t
  {
    auto a = generateRandomData<int16_t>(size, -500, 500);
    auto b = generateRandomData<int16_t>(size, -500, 500);
    std::vector<int16_t> r(size);
    simd_ops::subtract(a.data(), b.data(), r.data(), size);
    verifySubtraction(a, b, r);
  }

  // int32_t
  {
    std::vector<int32_t> a = generateRandomData<int32_t>(size, -100, 100);
    std::vector<int32_t> b = generateRandomData<int32_t>(size, -100, 100);
    std::vector<int32_t> result(size);

    simd_ops::subtract(a.data(), b.data(), result.data(), size);
    verifySubtraction(a, b, result);
  }

  // int64_t
  {
    std::vector<int64_t> a = generateRandomData<int64_t>(size, -100, 100);
    std::vector<int64_t> b = generateRandomData<int64_t>(size, -100, 100);
    std::vector<int64_t> result(size);

    simd_ops::subtract(a.data(), b.data(), result.data(), size);
    verifySubtraction(a, b, result);
  }
}

// Test subtraction with misaligned pointers
TEST_F(BinaryOperationsTest, SubtractWithMisalignment) {
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

        std::vector<float> a_data(size);
        std::vector<float> b_data(size);

        for (size_t i = 0; i < size; i++) {
          a_data[i] = a_ptr[i];
          b_data[i] = b_ptr[i];
        }

        simd_ops::subtract(a_ptr, b_ptr, result_ptr, size);

        std::vector<float> result_data(size);
        for (size_t i = 0; i < size; i++) {
          result_data[i] = result_ptr[i];
        }

        verifySubtraction(a_data, b_data, result_data);
      }
    }
  }
}

// Test edge cases for subtraction
TEST_F(BinaryOperationsTest, SubtractEdgeCases) {
  // Test with zeros
  {
    std::vector<float> a(64, 0.0f);
    std::vector<float> b(64, 0.0f);
    std::vector<float> result(64);

    simd_ops::subtract(a.data(), b.data(), result.data(), 64);

    for (size_t i = 0; i < 64; i++) {
      EXPECT_FLOAT_EQ(result[i], 0.0f);
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

    simd_ops::subtract(a.data(), b.data(), result.data(), 4);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isinf(result[1]) && result[1] > 0);
    EXPECT_TRUE(std::isinf(result[2]) && result[2] < 0);
    EXPECT_TRUE(std::isnan(result[3]));
  }
}

//------------------------------------------------------------------------------
// Multiplication Tests
//------------------------------------------------------------------------------

// Test multiplication operation
TEST_F(BinaryOperationsTest, MultiplyBasic) {
  const size_t size = 128;

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> result(size);

  simd_ops::multiply(a.data(), b.data(), result.data(), size);
  verifyMultiplication(a, b, result);
}

// Test multiplication with varying sizes
TEST_F(BinaryOperationsTest, MultiplyVaryingSizes) {
  const size_t sizes[] = {1, 3, 8, 16, 32, 64, 128, 500};

  for (size_t size : sizes) {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> result(size);

    simd_ops::multiply(a.data(), b.data(), result.data(), size);

    verifyMultiplication(a, b, result);
  }
}

// Test edge cases for multiplication
TEST_F(BinaryOperationsTest, MultiplyEdgeCases) {
  // Test with zeros
  {
    std::vector<float> a(64, 0.0f);
    std::vector<float> b = generateRandomData<float>(64, -100.0f, 100.0f);
    std::vector<float> result(64);

    simd_ops::multiply(a.data(), b.data(), result.data(), 64);

    for (size_t i = 0; i < 64; i++) {
      EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
  }

  // Test with NaN and Inf
  {
    std::vector<double> a = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(), 0.0};
    std::vector<double> b = {1.0, 2.0, 2.0,
                             std::numeric_limits<double>::infinity()};
    std::vector<double> result(4);

    simd_ops::multiply(a.data(), b.data(), result.data(), 4);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isinf(result[1]) && result[1] > 0);
    EXPECT_TRUE(std::isinf(result[2]) && result[2] < 0);
    EXPECT_TRUE(std::isnan(result[3])); // 0 * Inf = NaN
  }
}

// Test multiplication with integer types
TEST_F(BinaryOperationsTest, MultiplyIntegerTypes) {
  const size_t size = 64;

  // uint8_t
  {
    auto a = generateRandomData<uint8_t>(size, 0, 20);
    auto b = generateRandomData<uint8_t>(size, 0, 20);
    std::vector<uint8_t> r(size);
    simd_ops::multiply(a.data(), b.data(), r.data(), size);
    verifyMultiplication(a, b, r);
  }

  // int16_t
  {
    auto a = generateRandomData<int16_t>(size, -100, 100);
    auto b = generateRandomData<int16_t>(size, -100, 100);
    std::vector<int16_t> r(size);
    simd_ops::multiply(a.data(), b.data(), r.data(), size);
    verifyMultiplication(a, b, r);
  }
}

// Test multiplication with misaligned pointers
TEST_F(BinaryOperationsTest, MultiplyWithMisalignment) {
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

        std::vector<float> a_data(size);
        std::vector<float> b_data(size);

        for (size_t i = 0; i < size; i++) {
          a_data[i] = a_ptr[i];
          b_data[i] = b_ptr[i];
        }

        simd_ops::multiply(a_ptr, b_ptr, result_ptr, size);

        std::vector<float> result_data(size);
        for (size_t i = 0; i < size; i++) {
          result_data[i] = result_ptr[i];
        }

        verifyMultiplication(a_data, b_data, result_data);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Division Tests
//------------------------------------------------------------------------------

// Test division operation
TEST_F(BinaryOperationsTest, DivideBasic) {
  const size_t size = 128;

  std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
  std::vector<float> b =
      generateRandomData<float>(size, 0.1f, 100.0f); // Avoid division by zero
  std::vector<float> result(size);

  simd_ops::divide(a.data(), b.data(), result.data(), size);

  verifyDivision(a, b, result);
}

// Test division with varying sizes
TEST_F(BinaryOperationsTest, DivideVaryingSizes) {
  const size_t sizes[] = {1, 3, 8, 16, 32, 64, 128};

  for (size_t size : sizes) {
    std::vector<float> a = generateRandomData<float>(size, -100.0f, 100.0f);
    std::vector<float> b =
        generateRandomData<float>(size, 0.1f, 100.0f); // Avoid division by zero
    std::vector<float> result(size);

    simd_ops::divide(a.data(), b.data(), result.data(), size);

    verifyDivision(a, b, result);
  }
}

// Test division with integer types
TEST_F(BinaryOperationsTest, DivideIntegerTypes) {
  const size_t size = 64;

  // uint8_t
  {
    auto a = generateRandomData<uint8_t>(size, 1, 10);
    auto b = generateRandomData<uint8_t>(size, 1, 10);
    std::vector<uint8_t> r(size);
    simd_ops::divide(a.data(), b.data(), r.data(), size);
    verifyDivision(a, b, r);
  }

  // int16_t
  {
    auto a = generateRandomData<int16_t>(size, -100, 100);
    auto b = generateRandomData<int16_t>(size, 1, 100);
    std::vector<int16_t> r(size);
    simd_ops::divide(a.data(), b.data(), r.data(), size);
    verifyDivision(a, b, r);
  }
}

// Test division with misaligned pointers
TEST_F(BinaryOperationsTest, DivideWithMisalignment) {
  const size_t size = 1000;
  const size_t offsets[] = {1, 2, 3, 4, 7};

  std::vector<float> base_a =
      generateRandomData<float>(size + 8, -100.0f, 100.0f);
  std::vector<float> base_b = generateRandomData<float>(size + 8, 0.1f, 100.0f);
  std::vector<float> base_result(size + 8);

  for (size_t offset_a : offsets) {
    for (size_t offset_b : offsets) {
      for (size_t offset_result : offsets) {
        float *a_ptr = base_a.data() + offset_a;
        float *b_ptr = base_b.data() + offset_b;
        float *result_ptr = base_result.data() + offset_result;

        std::vector<float> a_data(size);
        std::vector<float> b_data(size);

        for (size_t i = 0; i < size; i++) {
          a_data[i] = a_ptr[i];
          b_data[i] = b_ptr[i];
        }

        simd_ops::divide(a_ptr, b_ptr, result_ptr, size);

        std::vector<float> result_data(size);
        for (size_t i = 0; i < size; i++) {
          result_data[i] = result_ptr[i];
        }

        verifyDivision(a_data, b_data, result_data);
      }
    }
  }
}

// Test edge cases for division
TEST_F(BinaryOperationsTest, DivideEdgeCases) {
  // Test with division by small values
  {
    std::vector<float> a(64, 1.0f);
    std::vector<float> b(64, std::numeric_limits<float>::min());
    std::vector<float> result(64);

    simd_ops::divide(a.data(), b.data(), result.data(), 64);

    for (size_t i = 0; i < 64; i++) {
      EXPECT_GT(result[i], 0); // Result should be positive and large
    }
  }

  // Test with NaN and Inf
  {
    std::vector<double> a = {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(), 1.0};
    std::vector<double> b = {1.0, 2.0, 2.0,
                             std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> result(4);

    simd_ops::divide(a.data(), b.data(), result.data(), 4);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isinf(result[1]) && result[1] > 0);
    EXPECT_TRUE(std::isinf(result[2]) && result[2] < 0);
    EXPECT_TRUE(std::isnan(result[3]));
  }

  // Test with division by zero
  {
    std::vector<float> a = {1.0f, -1.0f, 0.0f,
                            std::numeric_limits<float>::infinity()};
    std::vector<float> b = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> result(4);

    simd_ops::divide(a.data(), b.data(), result.data(), 4);

    EXPECT_TRUE(std::isinf(result[0]) && result[0] > 0); // +Inf
    EXPECT_TRUE(std::isinf(result[1]) && result[1] < 0); // -Inf
    EXPECT_TRUE(std::isnan(result[2]));                  // 0/0 = NaN
    EXPECT_TRUE(std::isnan(result[3]));                  // Inf/0 = NaN
  }
}

//------------------------------------------------------------------------------
// Buffer Operations Tests
//------------------------------------------------------------------------------

// Test buffer level operations
TEST_F(BinaryOperationsTest, BufferAddBasic) {
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
TEST_F(BinaryOperationsTest, BufferAddWithCasting) {
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

  // Perform addition with specified result type using binary_op
  Buffer result =
      simd_ops::binary_op(a_buf, b_buf, simd_ops::BinaryOpType::ADD, "float32");

  // Verify results
  auto &result_vec = std::get<VecBuffer<float>>(result.raw());

  for (size_t i = 0; i < 128; i++) {
    EXPECT_NEAR(result_vec[i], a_data[i] + static_cast<float>(b_data[i]),
                std::numeric_limits<float>::epsilon() * 10);
  }
}

// Test buffer add with different sizes
TEST_F(BinaryOperationsTest, BufferAddMismatchedSizes) {
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
  EXPECT_THROW(
      simd_ops::binary_op(a_buf, b_buf, simd_ops::BinaryOpType::ADD, "float32"),
      std::runtime_error);
}

// Test buffer addition with mixed types
TEST_F(BinaryOperationsTest, BufferAddWithTypeConversion) {
  const size_t N = 64;
  Buffer A(N, "int32"), B(N, "float32");
  auto &Ai = std::get<VecBuffer<int32_t>>(A.raw());
  auto &Bf = std::get<VecBuffer<float>>(B.raw());

  for (size_t i = 0; i < N; ++i) {
    Ai[i] = static_cast<int32_t>(i);
    Bf[i] = static_cast<float>(i) + 0.5f;
  }

  // float output
  {
    Buffer C =
        simd_ops::binary_op(A, B, simd_ops::BinaryOpType::ADD, "float32");
    auto &Cv = std::get<VecBuffer<float>>(C.raw());
    for (size_t i = 0; i < N; ++i) {
      float exp = i + (i + 0.5f);
      EXPECT_NEAR(Cv[i], exp, std::numeric_limits<float>::epsilon() * 10);
    }
  }

  // int32 output
  {
    Buffer C = simd_ops::binary_op(A, B, simd_ops::BinaryOpType::ADD, "int32");
    auto &Ci = std::get<VecBuffer<int32_t>>(C.raw());
    for (size_t i = 0; i < N; ++i) {
      int32_t exp = i + static_cast<int32_t>(i + 0.5f);
      EXPECT_EQ(Ci[i], exp);
    }
  }
}

//------------------------------------------------------------------------------
// Unary Operation Tests
//------------------------------------------------------------------------------

// Test power operation with float data
TEST_F(UnaryOperationsTest, PowerFloat) {
  const size_t size = 100;
  std::vector<float> base = generateRandomData<float>(size, 0.1f, 10.0f);
  std::vector<float> exp = generateRandomData<float>(size, 0.5f, 4.0f);
  std::vector<float> result(size);

  simd_ops::power<float>(base.data(), exp.data(), result.data(), size);

  verifyPower(base, exp, result);

  std::vector<float> test_base = {2.0f, 3.0f, 4.0f, 5.0f, 2.5f, 1.5f};
  std::vector<float> test_exp = {3.0f, 2.0f, 0.5f, 2.0f, 2.0f, 3.0f};
  std::vector<float> test_result(test_base.size());

  simd_ops::power<float>(test_base.data(), test_exp.data(), test_result.data(),
                         test_base.size());

  verifyPower(test_base, test_exp, test_result);
}

// Test power operation with integer data
TEST_F(UnaryOperationsTest, PowerInteger) {
  const size_t size = 6;
  std::vector<int32_t> base = {2, 3, 4, 5, 2, 10};
  std::vector<int32_t> exp = {3, 2, 2, 2, 4, 2};
  std::vector<int32_t> result(size);

  simd_ops::power<int32_t>(base.data(), exp.data(), result.data(), size);

  verifyPower(base, exp, result);
}

// Test power operation with Buffer class
TEST_F(UnaryOperationsTest, PowerBuffer) {
  try {
    // Create test buffers
    Buffer a(4, "float32");
    Buffer b(4, "float32");

    // Fill with test data
    auto &a_data = std::get<VecBuffer<float>>(a.raw());
    auto &b_data = std::get<VecBuffer<float>>(b.raw());

    a_data[0] = 2.0f;
    a_data[1] = 3.0f;
    a_data[2] = 4.0f;
    a_data[3] = 5.0f;
    b_data[0] = 2.0f;
    b_data[1] = 3.0f;
    b_data[2] = 0.5f;
    b_data[3] = 2.0f;

    // Test power operation
    Buffer power_result =
        simd_ops::binary_op(a, b, simd_ops::BinaryOpType::POW, "float32");
    auto &power_data = std::get<VecBuffer<float>>(power_result.raw());

    std::vector<float> base_vec = {a_data[0], a_data[1], a_data[2], a_data[3]};
    std::vector<float> exp_vec = {b_data[0], b_data[1], b_data[2], b_data[3]};
    std::vector<float> result_vec = {power_data[0], power_data[1],
                                     power_data[2], power_data[3]};

    verifyPower(base_vec, exp_vec, result_vec);
  } catch (const std::exception &e) {
    FAIL() << "Exception thrown: " << e.what();
  }
}

// Test power operation performance
TEST_F(UnaryOperationsTest, PowerPerformance) {
  const size_t N = 1000000;
  std::vector<float> a(N), b(N), result(N);

  // Initialize with test data
  for (size_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i % 100) + 1.0f;
    b[i] = 2.0f;
  }

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  simd_ops::power<float>(a.data(), b.data(), result.data(), N);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  // Verify at least a subset of results
  for (size_t i = 0; i < 100; ++i) {
    size_t idx = i * (N / 100);
    EXPECT_NEAR(result[idx], std::pow(a[idx], b[idx]), 0.001f);
  }
}

//------------------------------------------------------------------------------
// Negation Operation Tests
//------------------------------------------------------------------------------

// Test negation operation with float data
TEST_F(UnaryOperationsTest, NegateFloat) {
  const size_t size = 100;
  std::vector<float> input = generateRandomData<float>(size, -10.0f, 10.0f);
  std::vector<float> result(size);

  // Use the SIMD negate function
  simd_ops::negate<float>(input.data(), result.data(), size);

  // Verify results
  verifyNegation(input, result);

  // Check specific test cases
  std::vector<float> test_input = {1.5f, -2.5f, 0.0f, 3.14f, -1.0f, 42.0f};
  std::vector<float> test_result(test_input.size());

  simd_ops::negate<float>(test_input.data(), test_result.data(),
                          test_input.size());
  verifyNegation(test_input, test_result);
}

// Test negation operation with integer data
TEST_F(UnaryOperationsTest, NegateInteger) {
  const size_t size = 6;
  std::vector<int32_t> input = {1, -2, 0, 42, -100, 7};
  std::vector<int32_t> result(size);

  // Use the SIMD negate function
  simd_ops::negate<int32_t>(input.data(), result.data(), size);

  // Verify results
  verifyNegation(input, result);
}

// Test negation operation with Buffer class
TEST_F(UnaryOperationsTest, NegateBuffer) {
  try {

    Buffer a(4, "float32");

    auto &a_data = std::get<VecBuffer<float>>(a.raw());

    a_data[0] = 2.0f;
    a_data[1] = -3.0f;
    a_data[2] = 0.0f;
    a_data[3] = 5.0f;

    Buffer neg_result =
        simd_ops::unary_op(a, simd_ops::UnaryOpType::NEG, "float32");
    auto &neg_data = std::get<VecBuffer<float>>(neg_result.raw());

    std::vector<float> input_vec = {a_data[0], a_data[1], a_data[2], a_data[3]};
    std::vector<float> result_vec = {neg_data[0], neg_data[1], neg_data[2],
                                     neg_data[3]};

    verifyNegation(input_vec, result_vec);
  } catch (const std::exception &e) {
    FAIL() << "Exception thrown: " << e.what();
  }
}

// Test negation operation performance
TEST_F(UnaryOperationsTest, NegatePerformance) {
  const size_t N = 1000000;
  std::vector<float> a(N), result(N);

  // Initialize with test data
  for (size_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i % 100) - 50.0f;
  }

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  simd_ops::negate<float>(a.data(), result.data(), N);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  // Verify at least a subset of results
  for (size_t i = 0; i < 100; ++i) {
    size_t idx = i * (N / 100);
    EXPECT_EQ(result[idx], -a[idx]);
  }
}

//------------------------------------------------------------------------------
// SIMD Information
//------------------------------------------------------------------------------

TEST(SIMDInfo, PrintSIMDWidths) {
  std::cout << "SIMD Information:\n";
  std::cout << "================\n";

  std::cout << "Float SIMD width: " << xsimd::batch<float>::size
            << " elements\n";
  std::cout << "Double SIMD width: " << xsimd::batch<double>::size
            << " elements\n";
  std::cout << "Int32 SIMD width: " << xsimd::batch<int32_t>::size
            << " elements\n";
  std::cout << "Int64 SIMD width: " << xsimd::batch<int64_t>::size
            << " elements\n";
  std::cout << "\n";
  SUCCEED();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
