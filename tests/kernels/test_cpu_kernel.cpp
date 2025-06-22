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
#include <vector>
namespace py = pybind11;
TEST(BufferTest, ConstructorAndSize) {
  py::scoped_interpreter guard{};

  Buffer fbuf(5, "float32");
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.dtype(), "float32");

  Buffer ibuf(3, "int64");
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.dtype(), "int64");

  Buffer bbuf(2, "uint8");
  EXPECT_EQ(bbuf.size(), 2);
  EXPECT_EQ(bbuf.dtype(), "uint8");
}

// Test dtypes
TEST(BufferTest, DTypeCheck) {
  py::scoped_interpreter guard{};

  Buffer f16buf(1, "float16");
  EXPECT_EQ(f16buf.dtype(), "float16");

  Buffer f32buf(1, "float32");
  EXPECT_EQ(f32buf.dtype(), "float32");

  Buffer f64buf(1, "float64");
  EXPECT_EQ(f64buf.dtype(), "float64");

  Buffer i8buf(1, "int8");
  EXPECT_EQ(i8buf.dtype(), "int8");

  Buffer u8buf(1, "uint8");
  EXPECT_EQ(u8buf.dtype(), "uint8");

  Buffer i16buf(1, "int16");
  EXPECT_EQ(i16buf.dtype(), "int16");

  Buffer u16buf(1, "uint16");
  EXPECT_EQ(u16buf.dtype(), "uint16");

  Buffer i32buf(1, "int32");
  EXPECT_EQ(i32buf.dtype(), "int32");

  Buffer u32buf(1, "uint32");
  EXPECT_EQ(u32buf.dtype(), "uint32");

  Buffer i64buf(1, "int64");
  EXPECT_EQ(i64buf.dtype(), "int64");

  Buffer u64buf(1, "uint64");
  EXPECT_EQ(u64buf.dtype(), "uint64");
}

// Test set and get operations for various dtypes
// TEST(BufferTest, SetAndGet) {
//   py::scoped_interpreter guard{};
//
//   Buffer fbuf(1, "float32");
//   fbuf.set_item(0, 3.5);
//   EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 3.5f, 1e-6f);
//
//   Buffer ibuf(1, "int32");
//   ibuf.set_item(0, 7);
//   EXPECT_EQ(py::cast<int>(ibuf.get_item(0)), 7);
// }

// Test initializer list constructor
TEST(BufferTest, InitializerListConstructor) {
  py::scoped_interpreter guard{};

  // Float32 initialization
  Buffer fbuf({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, "float32");
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.dtype(), "float32");
  // Check values
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 1.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(1)), 2.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(2)), 3.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(3)), 4.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(4)), 5.0f, 1e-6f);

  // Int64 initialization
  Buffer ibuf({-1, 0, 1}, "int64");
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.dtype(), "int64");
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(0)), -1);
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(1)), 0);
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(2)), 1);

  // UInt8 initialization
  Buffer u8buf({0, 128, 255}, "uint8");
  EXPECT_EQ(u8buf.size(), 3);
  EXPECT_EQ(u8buf.dtype(), "uint8");
  EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(0)), 0);
  EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(1)), 128);
  EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(2)), 255);

  Buffer dbuf({1.5, 2.5, 3.14159}, "float64");
  EXPECT_EQ(dbuf.size(), 3);
  EXPECT_EQ(dbuf.dtype(), "float64");
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(0)), 1.5, 1e-9);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(1)), 2.5, 1e-9);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(2)), 3.14159, 1e-9);

  Buffer empty_buf({}, "int16");
  EXPECT_EQ(empty_buf.size(), 0);
  EXPECT_EQ(empty_buf.dtype(), "int16");

  Buffer conv_buf({1.1f, 2.9f, 3.5f}, "int32");
  EXPECT_EQ(conv_buf.size(), 3);
  EXPECT_EQ(conv_buf.dtype(), "int32");

  EXPECT_EQ(py::cast<int32_t>(conv_buf.get_item(0)), 1);
  EXPECT_EQ(py::cast<int32_t>(conv_buf.get_item(1)), 2);
  EXPECT_EQ(py::cast<int32_t>(conv_buf.get_item(2)), 3);
}

// Test py::buffer constructor
TEST(BufferTest, PyBufferConstructor) {
  py::scoped_interpreter guard{};

  py::module np = py::module::import("numpy");

  py::array_t<float> np_float =
      np.attr("array")(py::make_tuple(1.0f, 2.0f, 3.0f, 4.0f), "float32");
  Buffer float_buf(np_float, "float32");
  EXPECT_EQ(float_buf.size(), 4);
  EXPECT_EQ(float_buf.dtype(), "float32");

  // Test with int32 array
  py::array_t<int32_t> np_int =
      np.attr("array")(py::make_tuple(10, 20, 30), "int32");
  Buffer int_buf(np_int, "int32");
  EXPECT_EQ(int_buf.size(), 3);
  EXPECT_EQ(int_buf.dtype(), "int32");

  // Test with empty array
  py::array_t<double> np_empty = np.attr("array")(py::make_tuple(), "float64");
  Buffer empty_buf(np_empty, "float64");
  EXPECT_EQ(empty_buf.size(), 0);
  EXPECT_EQ(empty_buf.dtype(), "float64");

  // Test type conversion (uint8 buffer from int array)
  py::array_t<int> np_mix =
      np.attr("array")(py::make_tuple(0, 128, 255), "int32");
  Buffer uint_buf(np_mix, "uint8");
  EXPECT_EQ(uint_buf.size(), 3);
  EXPECT_EQ(uint_buf.dtype(), "uint8");
}

// Test exception handling in constructors
TEST(BufferTest, ConstructorExceptions) {
  py::scoped_interpreter guard{};

  // Test invalid dtype
  EXPECT_THROW(Buffer(5, "invalid_type"), std::runtime_error);
  EXPECT_THROW(Buffer({1, 2, 3}, "float128"), std::runtime_error);
}

// // Test buffer initialization with same value across different dtypes
// TEST(BufferTest, InitializeWithValue) {
//   py::scoped_interpreter guard{};
//
//   // Float buffers with same value
//   Buffer f32buf(5, "float32", py::cast(3.14f));
//   for (std::size_t i = 0; i < f32buf.size(); ++i) {
//     EXPECT_NEAR(py::cast<float>(f32buf.get_item(i)), 3.14f, 1e-6f);
//   }
//
//   Buffer f64buf(3, "float64", py::cast(2.71828));
//   for (std::size_t i = 0; i < f64buf.size(); ++i) {
//     EXPECT_NEAR(py::cast<double>(f64buf.get_item(i)), 2.71828, 1e-9);
//   }
//
//   // float16 buffer test
//   Buffer f16buf(4, "float16", py::cast(1.5f));
//   for (std::size_t i = 0; i < f16buf.size(); ++i) {
//     EXPECT_NEAR(py::cast<float>(f16buf.get_item(i)), 1.5f, 1e-3f);
//   }
//
//   // Integer buffers with same value
//   Buffer i32buf(4, "int32", py::cast(42));
//   for (std::size_t i = 0; i < i32buf.size(); ++i) {
//     EXPECT_EQ(py::cast<int>(i32buf.get_item(i)), 42);
//   }
//
//   // Use a large negative int64_t value that can be reliably represented
//   int64_t large_neg_value = -9223372036854775800LL;
//   Buffer i64buf(4, "int64", py::cast(large_neg_value));
//   for (std::size_t i = 0; i < i64buf.size(); ++i) {
//     EXPECT_EQ(py::cast<int64_t>(i64buf.get_item(i)), large_neg_value);
//   }
//
//   Buffer u8buf(3, "uint8", py::cast(uint8_t(255)));
//   for (std::size_t i = 0; i < u8buf.size(); ++i) {
//     EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(i)), uint8_t(255));
//   }
// }

// Out-of-bounds accesses should throw an exception
TEST(BufferTest, GetItemOutOfBounds) {
  py::scoped_interpreter guard{};
  Buffer buf(1, "float32");
  EXPECT_THROW(
      {
        auto result = buf.get_item(2);
        static_cast<void>(result);
      },
      std::out_of_range);
}

// Tests for VecBuffer's cast method directly
TEST(CastBufferTest, Int32ToFloat32) {
  VecBuffer<int32_t> in({10, -5, 0, 123});
  auto out = in.cast<float>();
  ASSERT_EQ(out.size(), in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(in[i]), out[i]);
  }
}

TEST(CastBufferTest, Float32ToInt16) {
  VecBuffer<float> in({1.9f, -2.1f, 0.0f, 32767.0f});
  auto out = in.cast<int16_t>();
  ASSERT_EQ(out.size(), in.size());
  EXPECT_EQ(out[0], static_cast<int16_t>(1.9f));
  EXPECT_EQ(out[1], static_cast<int16_t>(-2.1f));
  EXPECT_EQ(out[2], static_cast<int16_t>(0.0f));
  EXPECT_EQ(out[3], static_cast<int16_t>(32767.0f));
}

TEST(CastBufferTest, UInt8ToFloat64) {
  VecBuffer<uint8_t> in({0, 128, 255});
  auto out = in.cast<double>();
  ASSERT_EQ(out.size(), in.size());
  EXPECT_DOUBLE_EQ(static_cast<double>(in[0]), out[0]);
  EXPECT_DOUBLE_EQ(static_cast<double>(in[1]), out[1]);
  EXPECT_DOUBLE_EQ(static_cast<double>(in[2]), out[2]);
}

TEST(CastBufferTest, Float64ToInt64) {
  VecBuffer<double> in(
      {1.1, -2.9, 0.0, 9223372036854774000.0, -9223372036854774000.0});
  auto out = in.cast<int64_t>();
  ASSERT_EQ(out.size(), in.size());
  EXPECT_EQ(out[0], static_cast<int64_t>(1.1));
  EXPECT_EQ(out[1], static_cast<int64_t>(-2.9));
  EXPECT_EQ(out[2], static_cast<int64_t>(0.0));
  EXPECT_EQ(out[3], static_cast<int64_t>(9223372036854774000.0));
  EXPECT_EQ(out[4], static_cast<int64_t>(-9223372036854774000.0));
}

TEST(CastBufferTest, Int16ToFloat16) {
  VecBuffer<int16_t> in({-32768, -1, 0, 1, 32767});
  auto out = in.cast<half>();
  ASSERT_EQ(out.size(), in.size());

  // Convert half back to float for comparison
  std::array<float, 5> values;
  for (size_t i = 0; i < out.size(); ++i) {
    values[i] = static_cast<float>(out[i]);
  }

  EXPECT_NEAR(values[0], static_cast<float>(-32768),
              8.0f); // Allow some precision loss
  EXPECT_NEAR(values[1], -1.0f, 0.01f);
  EXPECT_NEAR(values[2], 0.0f, 0.01f);
  EXPECT_NEAR(values[3], 1.0f, 0.01f);
  EXPECT_NEAR(values[4], static_cast<float>(32767), 8.0f);
}

TEST(CastBufferTest, UInt32ToInt8) {
  VecBuffer<uint32_t> in({0, 127, 128, 255, 256, 1000000});
  auto out = in.cast<int8_t>();
  ASSERT_EQ(out.size(), in.size());

  EXPECT_EQ(out[0], static_cast<int8_t>(0));
  EXPECT_EQ(out[1], static_cast<int8_t>(127));
  EXPECT_EQ(out[2], static_cast<int8_t>(128));
  EXPECT_EQ(out[3], static_cast<int8_t>(255));
  EXPECT_EQ(out[4], static_cast<int8_t>(256));
  EXPECT_EQ(out[5], static_cast<int8_t>(1000000));
}

TEST(CastBufferTest, Int8ToUInt16) {
  VecBuffer<int8_t> in({-128, -1, 0, 1, 127});
  auto out = in.cast<uint16_t>();
  ASSERT_EQ(out.size(), in.size());

  EXPECT_EQ(out[0], static_cast<uint16_t>(-128));
  EXPECT_EQ(out[1], static_cast<uint16_t>(-1));
  EXPECT_EQ(out[2], static_cast<uint16_t>(0));
  EXPECT_EQ(out[3], static_cast<uint16_t>(1));
  EXPECT_EQ(out[4], static_cast<uint16_t>(127));
}

TEST(CastBufferTest, FloatSpecialValues) {
  VecBuffer<float> in({std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN(), -0.0f,
                       std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::max()});

  auto double_out = in.cast<double>();
  ASSERT_EQ(double_out.size(), in.size());
  EXPECT_TRUE(std::isinf(double_out[0]) && double_out[0] > 0);
  EXPECT_TRUE(std::isinf(double_out[1]) && double_out[1] < 0);
  EXPECT_TRUE(std::isnan(double_out[2]));
  EXPECT_DOUBLE_EQ(double_out[3], -0.0);
  EXPECT_DOUBLE_EQ(double_out[4],
                   static_cast<double>(std::numeric_limits<float>::min()));
  EXPECT_DOUBLE_EQ(double_out[5],
                   static_cast<double>(std::numeric_limits<float>::max()));

  auto int_out = in.cast<int32_t>();
  ASSERT_EQ(int_out.size(), in.size());
  // Implementation-defined behavior for inf/nan-to-int, but we can at least
  // verify the ordinary values
  EXPECT_EQ(int_out[3], static_cast<int32_t>(-0.0f));
  EXPECT_EQ(int_out[4],
            static_cast<int32_t>(std::numeric_limits<float>::min()));
}

// Test for chained casting operations
TEST(CastBufferTest, ChainedCasting) {
  VecBuffer<int32_t> in({-1000, 0, 1000});

  // int32 -> float32 -> uint8 -> int16
  auto float_out = in.cast<float>();
  auto uint8_out = float_out.cast<uint8_t>();
  auto int16_out = uint8_out.cast<int16_t>();

  ASSERT_EQ(int16_out.size(), in.size());

  // We can't make exact predictions without knowing the implementation details
  // Just verify that the values went through the conversion chain
  EXPECT_EQ(int16_out.size(), in.size());
}

TEST(CastBufferTest, ExtremeValueCasting) {

  // uint64 max to other types
  VecBuffer<uint64_t> uint64_max({std::numeric_limits<uint64_t>::max()});

  auto uint64_to_int32 = uint64_max.cast<int32_t>();
  EXPECT_EQ(uint64_to_int32[0],
            static_cast<int32_t>(std::numeric_limits<uint64_t>::max()));

  auto uint64_to_float = uint64_max.cast<float>();
  // Precision loss expected, can only check if it's a large positive number
  EXPECT_GT(uint64_to_float[0], 1.0e10f);

  // int64 min to other types
  VecBuffer<int64_t> int64_min({std::numeric_limits<int64_t>::min()});

  auto int64min_to_uint32 = int64_min.cast<uint32_t>();
  EXPECT_EQ(int64min_to_uint32[0],
            static_cast<uint32_t>(std::numeric_limits<int64_t>::min()));

  auto int64min_to_double = int64_min.cast<double>();
  EXPECT_DOUBLE_EQ(int64min_to_double[0],
                   static_cast<double>(std::numeric_limits<int64_t>::min()));
}

TEST(CastBufferTest, EmptyBufferCasting) {
  VecBuffer<float> empty_float;
  auto empty_int = empty_float.cast<int32_t>();
  EXPECT_EQ(empty_int.size(), 0);

  VecBuffer<int16_t> empty_int16(0);
  auto empty_uint8 = empty_int16.cast<uint8_t>();
  EXPECT_EQ(empty_uint8.size(), 0);
}

TEST(CastBufferTest, SameTypeCasting) {
  VecBuffer<int32_t> original({-42, 0, 42});
  auto copy = original.cast<int32_t>();

  ASSERT_EQ(copy.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(copy[i], original[i]);
  }
}

TEST(CastBufferTest, LargeBufferCasting) {
  constexpr size_t size = 1000;
  VecBuffer<float> large(size);

  // filling the buffer
  for (size_t i = 0; i < size; ++i) {
    large[i] = static_cast<float>(i) - 500.0f;
  }

  auto large_int = large.cast<int32_t>();
  ASSERT_EQ(large_int.size(), size);

  // Check a sampling of values
  EXPECT_EQ(large_int[0], -500);
  EXPECT_EQ(large_int[500], 0);
  EXPECT_EQ(large_int[999], 499);
}

// Test for precision loss in floating point casting
TEST(CastBufferTest, FloatPrecisionLoss) {
  VecBuffer<double> precise({0.1234567890123456789, 1e20, -1e-20});
  auto less_precise = precise.cast<float>();
  auto back_to_double = less_precise.cast<double>();

  ASSERT_EQ(back_to_double.size(), precise.size());

  // Values should be different due to precision loss
  EXPECT_NE(back_to_double[0], precise[0]);
  EXPECT_NE(back_to_double[1], precise[1]);
  // Very small values might become zero
  EXPECT_NEAR(back_to_double[2], 0.0, 1e-20);
}

// Test specifically for half-precision float conversions
TEST(CastBufferTest, HalfPrecisionConversions) {
  // Test float32 to float16 to float32 round trip
  VecBuffer<float> orig_float({-65504.0f, // Min normal float16
                               65504.0f,  // Max normal float16
                               0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                               0.1f,    // Will lose precision
                               1.0e-8f, // Will be flushed to zero
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()});

  auto half_cast = orig_float.cast<half>();
  auto back_to_float = half_cast.cast<float>();

  ASSERT_EQ(back_to_float.size(), orig_float.size());

  // Values that should be preserved exactly
  EXPECT_NEAR(back_to_float[2], 0.0f, 1e-6f);
  EXPECT_NEAR(back_to_float[3], 1.0f, 1e-6f);
  EXPECT_NEAR(back_to_float[4], -1.0f, 1e-6f);
  EXPECT_NEAR(back_to_float[5], 0.5f, 1e-6f);
  EXPECT_NEAR(back_to_float[6], -0.5f, 1e-6f);

  // Values at float16 limits should be preserved
  EXPECT_NEAR(back_to_float[0], -65504.0f, 1.0f);
  EXPECT_NEAR(back_to_float[1], 65504.0f, 1.0f);

  // Value with precision loss
  EXPECT_NE(back_to_float[7], orig_float[7]);
  EXPECT_NEAR(back_to_float[7], 0.1f, 0.01f);

  // Value that should flush to zero
  EXPECT_NEAR(back_to_float[8], 0.0f, 1e-6f);

  // Special values
  EXPECT_TRUE(std::isinf(back_to_float[9]) && back_to_float[9] > 0);
  EXPECT_TRUE(std::isinf(back_to_float[10]) && back_to_float[10] < 0);
  EXPECT_TRUE(std::isnan(back_to_float[11]));

  // Test half to int16 conversion
  auto half_to_int16 = half_cast.cast<int16_t>();
  EXPECT_EQ(half_to_int16[2], 0);
  EXPECT_EQ(half_to_int16[3], 1);
  EXPECT_EQ(half_to_int16[4], -1);
  // Implementation-dependent, don't test exact values
  // Just verify we got some values back from the conversion
  EXPECT_EQ(half_to_int16.size(), half_cast.size());
}

// Test for signed/unsigned integer conversions
TEST(CastBufferTest, SignedUnsignedConversions) {
  // Test various signed to unsigned conversions
  VecBuffer<int32_t> signed_ints({-100, -1, 0, 1, 100,
                                  std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::max()});

  // int32 to uint32
  auto uint32_cast = signed_ints.cast<uint32_t>();
  ASSERT_EQ(uint32_cast.size(), signed_ints.size());
  EXPECT_EQ(uint32_cast[0], static_cast<uint32_t>(-100));
  EXPECT_EQ(uint32_cast[1], static_cast<uint32_t>(-1)); // Wraps to max uint32
  EXPECT_EQ(uint32_cast[2], 0u);
  EXPECT_EQ(uint32_cast[3], 1u);
  EXPECT_EQ(uint32_cast[4], 100u);
  EXPECT_EQ(uint32_cast[5],
            static_cast<uint32_t>(std::numeric_limits<int32_t>::min()));
  EXPECT_EQ(uint32_cast[6],
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));

  // int32 to uint16 (truncation)
  auto uint16_cast = signed_ints.cast<uint16_t>();
  ASSERT_EQ(uint16_cast.size(), signed_ints.size());
  EXPECT_EQ(uint16_cast[0], static_cast<uint16_t>(-100));
  EXPECT_EQ(uint16_cast[1], static_cast<uint16_t>(-1)); // Wraps to 0xFFFF
  EXPECT_EQ(uint16_cast[2], 0u);
  EXPECT_EQ(uint16_cast[3], 1u);
  EXPECT_EQ(uint16_cast[4], 100u);
  // The min/max values will be truncated to uint16
  EXPECT_EQ(uint16_cast[5],
            static_cast<uint16_t>(std::numeric_limits<int32_t>::min()));
  EXPECT_EQ(uint16_cast[6],
            static_cast<uint16_t>(std::numeric_limits<int32_t>::max()));

  // Test unsigned to signed conversions
  VecBuffer<uint32_t> unsigned_ints({
      0, 1, 100,
      0x7FFFFFFF,                          // Max int32 value
      0x80000000,                          // Just above max int32 value
      std::numeric_limits<uint32_t>::max() // Max uint32
  });

  // uint32 to int32
  auto int32_cast = unsigned_ints.cast<int32_t>();
  ASSERT_EQ(int32_cast.size(), unsigned_ints.size());
  EXPECT_EQ(int32_cast[0], 0);
  EXPECT_EQ(int32_cast[1], 1);
  EXPECT_EQ(int32_cast[2], 100);
  EXPECT_EQ(int32_cast[3], static_cast<int32_t>(0x7FFFFFFF)); // OK, max int32
  EXPECT_EQ(int32_cast[4],
            static_cast<int32_t>(0x80000000)); // Overflows to min int32
  EXPECT_EQ(int32_cast[5],
            static_cast<int32_t>(std::numeric_limits<uint32_t>::max())); // -1

  // uint32 to int16
  auto int16_cast = unsigned_ints.cast<int16_t>();
  ASSERT_EQ(int16_cast.size(), unsigned_ints.size());
  EXPECT_EQ(int16_cast[0], 0);
  EXPECT_EQ(int16_cast[1], 1);
  EXPECT_EQ(int16_cast[2], 100);
  // Larger values will be truncated to int16
  EXPECT_EQ(int16_cast[3], static_cast<int16_t>(0x7FFFFFFF));
  EXPECT_EQ(int16_cast[4], static_cast<int16_t>(0x80000000));
  EXPECT_EQ(int16_cast[5],
            static_cast<int16_t>(std::numeric_limits<uint32_t>::max()));
}

// Test for overflow and underflow in casting
TEST(CastBufferTest, OverflowUnderflowCasting) {
  // Test overflow from float to int
  VecBuffer<float> big_floats(
      {1.0e10f, -1.0e10f,
       static_cast<float>(std::numeric_limits<int32_t>::max()) * 2.0f,
       static_cast<float>(std::numeric_limits<int32_t>::min()) * 2.0f,
       std::numeric_limits<float>::max(),
       std::numeric_limits<float>::lowest()});

  auto int32_from_big_float = big_floats.cast<int32_t>();
  ASSERT_EQ(int32_from_big_float.size(), big_floats.size());

  // Implementation-defined behavior, but typically:
  // Values too large become INT_MAX, too small become INT_MIN
  // or might wrap around (undefined behavior in C++)
  EXPECT_TRUE(int32_from_big_float[0] == std::numeric_limits<int32_t>::max() ||
              int32_from_big_float[0] < 0); // Overflow case

  EXPECT_TRUE(int32_from_big_float[1] == std::numeric_limits<int32_t>::min() ||
              int32_from_big_float[1] > 0); // Underflow case

  // Test small float values to unsigned int (negative values)
  VecBuffer<float> neg_floats({-1.0f, -0.5f, -0.0f});
  auto uint32_from_neg = neg_floats.cast<uint32_t>();
  ASSERT_EQ(uint32_from_neg.size(), neg_floats.size());

  // Implementation-defined behavior for negative float to unsigned int
  // We can at least check the last value
  EXPECT_EQ(uint32_from_neg[2], 0u); // -0.0f should become 0
}

// Test for fractional value conversions to integers
TEST(CastBufferTest, FractionalToIntegerCasting) {
  VecBuffer<double> fractions({0.0, 0.1, 0.49, 0.5, 0.51, 0.99, 1.0, -0.1,
                               -0.49, -0.5, -0.51, -0.99, -1.0});
  auto int_cast = fractions.cast<int32_t>();

  ASSERT_EQ(int_cast.size(), fractions.size());

  // Check truncation behavior (C++ standard truncates toward zero)
  EXPECT_EQ(int_cast[0], 0); // 0.0 -> 0
  EXPECT_EQ(int_cast[1], 0); // 0.1 -> 0
  EXPECT_EQ(int_cast[2], 0); // 0.49 -> 0
  EXPECT_EQ(int_cast[3], 0); // 0.5 -> 0
  EXPECT_EQ(int_cast[4], 0); // 0.51 -> 0
  EXPECT_EQ(int_cast[5], 0); // 0.99 -> 0
  EXPECT_EQ(int_cast[6], 1); // 1.0 -> 1

  EXPECT_EQ(int_cast[7], 0);   // -0.1 -> 0
  EXPECT_EQ(int_cast[8], 0);   // -0.49 -> 0
  EXPECT_EQ(int_cast[9], 0);   // -0.5 -> 0
  EXPECT_EQ(int_cast[10], 0);  // -0.51 -> 0
  EXPECT_EQ(int_cast[11], 0);  // -0.99 -> 0
  EXPECT_EQ(int_cast[12], -1); // -1.0 -> -1
}

TEST(CastBufferTest, MixedPrecisionCastChains) {
  // Create values that span the full range of uint8
  VecBuffer<uint8_t> uint8_values(256);
  for (int i = 0; i < 256; i++) {
    uint8_values[i] = static_cast<uint8_t>(i);
  }

  // Test chain: uint8 -> float32 -> float16 -> float64 -> int32
  auto to_float32 = uint8_values.cast<float>();
  auto to_float16 = to_float32.cast<half>();
  auto to_float64 = to_float16.cast<double>();
  auto to_int32 = to_float64.cast<int32_t>();

  ASSERT_EQ(to_int32.size(), uint8_values.size());

  // Check that basic values survived the conversion chain
  EXPECT_EQ(to_int32[0], 0);     // 0 should survive all conversions
  EXPECT_EQ(to_int32[1], 1);     // 1 should survive all conversions
  EXPECT_EQ(to_int32[255], 255); // 255 should survive all conversions

  // Test another chain: int32 -> uint64 -> float32 -> int16
  VecBuffer<int32_t> int32_values({-1000000, -1, 0, 1, 1000000});
  auto to_uint64 = int32_values.cast<uint64_t>();
  auto back_to_float = to_uint64.cast<float>();
  auto to_int16 = back_to_float.cast<int16_t>();

  ASSERT_EQ(to_int16.size(), int32_values.size());

  // Negative values will become large positive values in uint64, then might
  // overflow in float->int16
  EXPECT_EQ(to_int16[2], 0); // 0 should survive all conversions
  EXPECT_EQ(to_int16[3], 1); // 1 should survive all conversions
}

// Test for subnormal floating-point values
TEST(CastBufferTest, SubnormalFloatCasting) {
  // Create some subnormal float32 values
  float min_normal = std::numeric_limits<float>::min();
  VecBuffer<float> subnormals({
      0.0f,
      std::numeric_limits<float>::denorm_min(), // Smallest positive subnormal
      min_normal / 2,                           // A subnormal value
      min_normal                                // Smallest normal value
  });

  auto to_double = subnormals.cast<double>();
  ASSERT_EQ(to_double.size(), subnormals.size());
  EXPECT_DOUBLE_EQ(to_double[0], 0.0);
  EXPECT_DOUBLE_EQ(to_double[1], static_cast<double>(
                                     std::numeric_limits<float>::denorm_min()));
  EXPECT_DOUBLE_EQ(to_double[2], static_cast<double>(min_normal / 2));
  EXPECT_DOUBLE_EQ(to_double[3], static_cast<double>(min_normal));

  // Cast to half (some values may flush to zero)
  auto to_half = subnormals.cast<half>();
  auto back_to_float = to_half.cast<float>();
  ASSERT_EQ(back_to_float.size(), subnormals.size());
  EXPECT_FLOAT_EQ(back_to_float[0], 0.0f);
}

// Test for integer rounding during floating point casting
TEST(CastBufferTest, IntegerRoundingInFloatCast) {
  // Large integers that can't be exactly represented in floating point
  VecBuffer<int64_t> large_ints({
      1000000000000000000LL, // 10^18, beyond exact float32 precision
      1000000000000000001LL, // 10^18 + 1, will round in float
      9007199254740992LL,    // 2^53, max exact double precision integer
      9007199254740993LL     // 2^53 + 1, will round in double
  });

  auto to_float = large_ints.cast<float>();
  auto back_to_int64 = to_float.cast<int64_t>();

  ASSERT_EQ(back_to_int64.size(), large_ints.size());

  EXPECT_NE(back_to_int64[0], large_ints[0]);
  EXPECT_NE(back_to_int64[1], large_ints[1]);

  auto to_double = large_ints.cast<double>();
  auto double_to_int64 = to_double.cast<int64_t>();

  ASSERT_EQ(double_to_int64.size(), large_ints.size());

  // Double can represent 2^53 exactly, but not 2^53+1
  EXPECT_EQ(double_to_int64[2], large_ints[2]);
  EXPECT_NE(double_to_int64[3], large_ints[3]);
}

TEST(CastBufferTest, ComprehensiveTypeConversionGrid) {

  // int8 conversions
  {
    VecBuffer<int8_t> val({42});
    EXPECT_EQ(val.cast<int16_t>()[0], static_cast<int16_t>(42));
    EXPECT_EQ(val.cast<int32_t>()[0], static_cast<int32_t>(42));
    EXPECT_EQ(val.cast<int64_t>()[0], static_cast<int64_t>(42));
    EXPECT_EQ(val.cast<uint8_t>()[0], static_cast<uint8_t>(42));
    EXPECT_EQ(val.cast<uint16_t>()[0], static_cast<uint16_t>(42));
    EXPECT_EQ(val.cast<uint32_t>()[0], static_cast<uint32_t>(42));
    EXPECT_EQ(val.cast<uint64_t>()[0], static_cast<uint64_t>(42));
    EXPECT_NEAR(static_cast<float>(val.cast<half>()[0]), 42.0f, 0.01f);
    EXPECT_FLOAT_EQ(val.cast<float>()[0], 42.0f);
    EXPECT_DOUBLE_EQ(val.cast<double>()[0], 42.0);
  }

  // Test negative int8 conversion to unsigned types
  {
    VecBuffer<int8_t> neg_val({-42});
    EXPECT_EQ(neg_val.cast<int16_t>()[0], static_cast<int16_t>(-42));
    EXPECT_EQ(neg_val.cast<uint8_t>()[0],
              static_cast<uint8_t>(-42)); // Will wrap around
    EXPECT_EQ(neg_val.cast<uint16_t>()[0],
              static_cast<uint16_t>(-42)); // Will wrap around
  }

  // uint64 max value conversions
  {
    VecBuffer<uint64_t> max_val({std::numeric_limits<uint64_t>::max()});

    // To signed integers (implementation-defined, but usually modulo
    // conversion)
    EXPECT_EQ(max_val.cast<int8_t>()[0],
              static_cast<int8_t>(std::numeric_limits<uint64_t>::max()));
    EXPECT_EQ(max_val.cast<int16_t>()[0],
              static_cast<int16_t>(std::numeric_limits<uint64_t>::max()));
    EXPECT_EQ(max_val.cast<int32_t>()[0],
              static_cast<int32_t>(std::numeric_limits<uint64_t>::max()));
    EXPECT_EQ(max_val.cast<int64_t>()[0],
              static_cast<int64_t>(std::numeric_limits<uint64_t>::max()));

    // To floating point (will lose precision)
    EXPECT_GT(max_val.cast<float>()[0], 1.0e19f);
    EXPECT_GT(max_val.cast<double>()[0], 1.0e19);
  }

  // Check double->float->half conversion precision chain
  {
    VecBuffer<double> pi_val({3.14159265358979323846});
    auto pi_float = pi_val.cast<float>();
    auto pi_half = pi_float.cast<half>();

    // Verify the expected precision loss
    EXPECT_NE(static_cast<double>(pi_float[0]), pi_val[0]);
    EXPECT_NEAR(pi_float[0], 3.14159265f, 1e-7f);

    auto half_as_float = static_cast<float>(pi_half[0]);
    EXPECT_NE(half_as_float, pi_float[0]);
    EXPECT_NEAR(half_as_float, 3.14f, 0.01f);
  }
}

// TEST(BufferTest, AddOperation) {
//   py::scoped_interpreter guard{};

//   Buffer a({1.0f, 2.0f, 3.0f}, "float32");
//   Buffer b({4.0f, 5.0f, 6.0f}, "float32");
//   Buffer c = simd_ops::(a, b, "float32");
//   std::vector<float> expected = {5.0f, 7.0f, 9.0f};
//   const auto &c_buf = std::get<VecBuffer<float>>(c.raw());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_NEAR(c_buf[i], expected[i], 1e-6);
// }
// }

// TEST(BufferDeathTest, SetItemOutOfBounds) {
//   Buffer buf(1, "float32");
//   EXPECT_DEATH({ buf.set_item(3, 1.0); }, "index");
// }

// // Validate type conversions and data preservation
// TEST(BufferTest, TypeConversions) {
//   py::scoped_interpreter guard{};

//   Buffer dbuf(1, "float64");
//   dbuf.set_item(0, 3.14159);
//   EXPECT_NEAR(py::cast<double>(dbuf.get_item(0)), 3.14159, 1e-9);

//   Buffer ibuf(1, "int64");
//   ibuf.set_item(0, -5);
//   EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(0)), -5);

//   Buffer fbuf(1, "float32");
//   fbuf.set_item(0, 42);
//   EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 42.0f, 1e-6f);

//   fbuf.set_item(0, 5.5);
//   EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 5.5f, 1e-6f);
// }
