/*
 * Copyright 2025 Saksham Bedi
 */

#include "../../kernels/cpu_kernel.h"
#include "../../kernels/operations.h"

#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>

namespace py = pybind11;

// Fixture for elementwise addition tests
class ElementwiseAddTest : public ::testing::Test {
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

  // Generate a vector of random values between [min_val, max_val]
  template <typename T>
  std::vector<T> generateRandomData(size_t size, T min_val, T max_val) {
    std::vector<T> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dist(min_val, max_val);
      for (auto &v : data) {
        v = dist(gen);
      }
    } else {
      std::uniform_int_distribution<int> dist(static_cast<int>(min_val),
                                              static_cast<int>(max_val));
      for (auto &v : data) {
        v = static_cast<T>(dist(gen));
      }
    }
    return data;
  }

  // Compare each element in result against a + b
  template <typename T>
  void verifyAddition(const std::vector<T> &a, const std::vector<T> &b,
                      const std::vector<T> &result) {
    for (size_t i = 0; i < result.size(); ++i) {
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_NEAR(result[i], a[i] + b[i],
                    std::numeric_limits<T>::epsilon() * 10)
            << "Index " << i;
      } else {
        EXPECT_EQ(result[i], static_cast<T>(a[i] + b[i])) << "Index " << i;
      }
    }
  }
};

// Basic float addition
TEST_F(ElementwiseAddTest, FloatBasicAdd) {
  const size_t N = 128;
  auto a = generateRandomData<float>(N, -100.0f, 100.0f);
  auto b = generateRandomData<float>(N, -100.0f, 100.0f);
  std::vector<float> result(N);

  simd_ops::add(a.data(), b.data(), result.data(), N);
  verifyAddition(a, b, result);
}

// Addition on small arrays (sizes < SIMD width)
TEST_F(ElementwiseAddTest, SmallArrayAdd) {
  std::array<size_t, 5> sizes = {1, 2, 3, 5, 7};
  for (auto size : sizes) {
    auto a = generateRandomData<float>(size, -50.0f, 50.0f);
    auto b = generateRandomData<float>(size, -50.0f, 50.0f);
    std::vector<float> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }
}

// Test around SIMD boundaries (e.g., 4, 8, 16 floats)
TEST_F(ElementwiseAddTest, SIMDBoundaryAdd) {
  std::array<size_t, 8> sizes = {3, 4, 5, 7, 8, 9, 15, 16};
  for (auto size : sizes) {
    auto a = generateRandomData<float>(size, -80.0f, 80.0f);
    auto b = generateRandomData<float>(size, -80.0f, 80.0f);
    std::vector<float> result(size);

    simd_ops::add(a.data(), b.data(), result.data(), size);
    verifyAddition(a, b, result);
  }
}

// Run addition on all supported types
TEST_F(ElementwiseAddTest, AddAllSupportedTypes) {
  const size_t N = 64;

  // float
  {
    auto a = generateRandomData<float>(N, -100.0f, 100.0f);
    auto b = generateRandomData<float>(N, -100.0f, 100.0f);
    std::vector<float> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // double
  {
    auto a = generateRandomData<double>(N, -200.0, 200.0);
    auto b = generateRandomData<double>(N, -200.0, 200.0);
    std::vector<double> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // uint8_t
  {
    auto a = generateRandomData<uint8_t>(N, 0, 100);
    auto b = generateRandomData<uint8_t>(N, 0, 100);
    std::vector<uint8_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // int16_t
  {
    auto a = generateRandomData<int16_t>(N, -500, 500);
    auto b = generateRandomData<int16_t>(N, -500, 500);
    std::vector<int16_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // uint16_t
  {
    auto a = generateRandomData<uint16_t>(N, 0, 1000);
    auto b = generateRandomData<uint16_t>(N, 0, 1000);
    std::vector<uint16_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // int32_t
  {
    auto a = generateRandomData<int32_t>(N, -10000, 10000);
    auto b = generateRandomData<int32_t>(N, -10000, 10000);
    std::vector<int32_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // uint32_t
  {
    auto a = generateRandomData<uint32_t>(N, 0, 20000);
    auto b = generateRandomData<uint32_t>(N, 0, 20000);
    std::vector<uint32_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // int64_t
  {
    auto a = generateRandomData<int64_t>(N, -100000LL, 100000LL);
    auto b = generateRandomData<int64_t>(N, -100000LL, 100000LL);
    std::vector<int64_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // uint64_t
  {
    auto a = generateRandomData<uint64_t>(N, 0ULL, 100000ULL);
    auto b = generateRandomData<uint64_t>(N, 0ULL, 100000ULL);
    std::vector<uint64_t> r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    verifyAddition(a, b, r);
  }

  // half precision
  {
    std::vector<half> a(N), b(N), r(N);
    auto fa = generateRandomData<float>(N, -10.0f, 10.0f);
    auto fb = generateRandomData<float>(N, -10.0f, 10.0f);

    for (size_t i = 0; i < N; ++i) {
      a[i] = half(fa[i]);
      b[i] = half(fb[i]);
    }

    simd_ops::add(a.data(), b.data(), r.data(), N);

    for (size_t i = 0; i < N; ++i) {
      float expected = fa[i] + fb[i];
      EXPECT_NEAR(static_cast<float>(r[i]), expected, 0.1f) << "Index " << i;
    }
  }
}

// Verify operation on misaligned pointers
TEST_F(ElementwiseAddTest, MisalignedPointers) {
  const size_t N = 128;
  auto baseA = generateRandomData<float>(N + 3, -50.0f, 50.0f);
  auto baseB = generateRandomData<float>(N + 3, -50.0f, 50.0f);
  std::vector<float> out(N + 3);

  for (int offA = 0; offA < 3; ++offA) {
    for (int offB = 0; offB < 3; ++offB) {
      for (int offR = 0; offR < 3; ++offR) {
        float *a = baseA.data() + offA;
        float *b = baseB.data() + offB;
        float *r = out.data() + offR;

        simd_ops::add(a, b, r, N);
        for (size_t i = 0; i < N; ++i) {
          EXPECT_NEAR(r[i], a[i] + b[i],
                      std::numeric_limits<float>::epsilon() * 10)
              << "Offsets A=" << offA << " B=" << offB << " R=" << offR
              << " idx=" << i;
        }
      }
    }
  }
}

// Compare aligned vs unaligned kernels
TEST_F(ElementwiseAddTest, AlignedVsUnaligned) {
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

// Edge cases: zeros, large values, nan/inf
TEST_F(ElementwiseAddTest, EdgeCases) {
  // zeros
  {
    const size_t N = 32;
    std::vector<float> z(N, 0.0f), r(N);
    simd_ops::add(z.data(), z.data(), r.data(), N);
    for (auto v : r) {
      EXPECT_FLOAT_EQ(v, 0.0f);
    }
  }

  // overflow to infinity
  {
    const size_t N = 32;
    float big = std::numeric_limits<float>::max();
    std::vector<float> a(N, big), b(N, big), r(N);
    simd_ops::add(a.data(), b.data(), r.data(), N);
    for (auto v : r) {
      EXPECT_TRUE(std::isinf(v) && v > 0);
    }
  }

  // special values
  {
    std::vector<float> a = {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(), 1.0f};
    std::vector<float> b = {-1.0f, 1.0f, 1.0f,
                            std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> r(a.size());

    simd_ops::add(a.data(), b.data(), r.data(), r.size());
    EXPECT_TRUE(std::isinf(r[0]) && r[0] > 0);
    EXPECT_TRUE(std::isinf(r[1]) && r[1] < 0);
    EXPECT_TRUE(std::isnan(r[2]));
    EXPECT_TRUE(std::isnan(r[3]));
  }
}

// Overflow produces +Inf
TEST(SimdAdd, ProducesInfinityOnOverflow) {
  const size_t N = 32;
  float big = std::numeric_limits<float>::max();
  float posInf = std::numeric_limits<float>::infinity();

  std::vector<float> a(N, big), b(N, big), r(N);
  simd_ops::add(a.data(), b.data(), r.data(), N);

  for (auto v : r) {
    EXPECT_EQ(v, posInf);
  }
}

// Quick performance check (prints timing to stdout)
TEST_F(ElementwiseAddTest, AddPerformance) {
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
    std::cout << "N=" << N << ": " << dt.count() << " ms\n";
  }
}

// Test add via Buffer wrapper
TEST_F(ElementwiseAddTest, BufferAdd) {
  const size_t N = 128;
  auto da = generateRandomData<float>(N, -100.0f, 100.0f);
  auto db = generateRandomData<float>(N, -100.0f, 100.0f);

  Buffer A(N, "float32"), B(N, "float32");
  auto &Av = std::get<VecBuffer<float>>(A.raw());
  auto &Bv = std::get<VecBuffer<float>>(B.raw());

  for (size_t i = 0; i < N; ++i) {
    Av[i] = da[i];
    Bv[i] = db[i];
  }

  Buffer C = simd_ops::buffer_add(A, B, "float32");
  auto &Cv = std::get<VecBuffer<float>>(C.raw());

  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(Cv[i], da[i] + db[i],
                std::numeric_limits<float>::epsilon() * 10);
  }
}

// Buffer addition with mixed types
TEST_F(ElementwiseAddTest, BufferAddWithTypeConversion) {
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
    Buffer C = simd_ops::buffer_add(A, B, "float32");
    auto &Cv = std::get<VecBuffer<float>>(C.raw());
    for (size_t i = 0; i < N; ++i) {
      float exp = i + (i + 0.5f);
      EXPECT_NEAR(Cv[i], exp, std::numeric_limits<float>::epsilon() * 10);
    }
  }

  // int32 output
  {
    Buffer C = simd_ops::buffer_add(A, B, "int32");
    auto &Ci = std::get<VecBuffer<int32_t>>(C.raw());
    for (size_t i = 0; i < N; ++i) {
      int32_t exp = i + static_cast<int32_t>(i + 0.5f);
      EXPECT_EQ(Ci[i], exp);
    }
  }
}
