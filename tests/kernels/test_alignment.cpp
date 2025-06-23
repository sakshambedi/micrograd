// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "../../kernels/operations.h"
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>

TEST(AlignmentTest, FloatAlignment) {
  using T = float;
  const std::size_t alignment = simd_ops::simd_alignment<T>();
  T *ptr = static_cast<T *>(std::aligned_alloc(alignment, alignment * 2));
  ASSERT_NE(ptr, nullptr);

  EXPECT_TRUE(simd_ops::is_aligned<T>(ptr));
  EXPECT_EQ(simd_ops::align_offset<T>(ptr), 0U);

  char *mis_bytes = reinterpret_cast<char *>(ptr) + 1;
  T *mis_ptr = reinterpret_cast<T *>(mis_bytes);
  EXPECT_FALSE(simd_ops::is_aligned<T>(mis_ptr));

  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(mis_ptr);
  std::size_t expected = alignment - (addr % alignment);
  if (expected == alignment)
    expected = 0;
  expected /= sizeof(T);
  EXPECT_EQ(simd_ops::align_offset<T>(mis_ptr), expected);

  std::free(ptr);
}

TEST(AlignmentTest, Int32Alignment) {
  using T = int32_t;
  const std::size_t alignment = simd_ops::simd_alignment<T>();
  T *ptr = static_cast<T *>(std::aligned_alloc(alignment, alignment * 2));
  ASSERT_NE(ptr, nullptr);

  EXPECT_TRUE(simd_ops::is_aligned<T>(ptr));
  EXPECT_EQ(simd_ops::align_offset<T>(ptr), 0U);

  char *mis_bytes = reinterpret_cast<char *>(ptr) + 1;
  T *mis_ptr = reinterpret_cast<T *>(mis_bytes);
  EXPECT_FALSE(simd_ops::is_aligned<T>(mis_ptr));

  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(mis_ptr);
  std::size_t expected = alignment - (addr % alignment);
  if (expected == alignment)
    expected = 0;
  expected /= sizeof(T);
  EXPECT_EQ(simd_ops::align_offset<T>(mis_ptr), expected);

  std::free(ptr);
}

TEST(AlignmentTest, SimdWidthAndBytes) {
  EXPECT_EQ(simd_ops::simd_width<float>(), xsimd::batch<float>::size);
  EXPECT_EQ(simd_ops::simd_bytes<float>(),
            xsimd::batch<float>::size * sizeof(float));

  EXPECT_EQ(simd_ops::simd_width<int32_t>(), xsimd::batch<int32_t>::size);
  EXPECT_EQ(simd_ops::simd_bytes<int32_t>(),
            xsimd::batch<int32_t>::size * sizeof(int32_t));
}
