// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
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

// // Test set and get operations for various dtypes
//  TEST(BufferTest, SetAndGet) {
//   py::scoped_interpreter guard{};

//   Buffer fbuf(1, "float32");
//   fbuf.set_item(0, 3.5);
//   EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 3.5f, 1e-6f);

//   Buffer ibuf(1, "int32");
//   ibuf.set_item(0, 7);
//   EXPECT_EQ(py::cast<int>(ibuf.get_item(0)), 7);

//   Buffer bbuf(1, "bool");
//   bbuf.set_item(0, 1.0);
//   EXPECT_TRUE(py::cast<bool>(bbuf.get_item(0)));
// }

// Test initializer list constructor
TEST(BufferTest, InitializerListConstructor) {
  py::scoped_interpreter guard{};

  // Float32 initialization
  Buffer fbuf({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, "float32");
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.dtype(), "float32");

  // Int64 initialization
  Buffer ibuf({-1, 0, 1}, "int64");
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.dtype(), "int64");

  // UInt8 initialization
  Buffer u8buf({0, 128, 255}, "uint8");
  EXPECT_EQ(u8buf.size(), 3);
  EXPECT_EQ(u8buf.dtype(), "uint8");

  // Float64 initialization with mixed values
  Buffer dbuf({1.5, 2.5, 3.14159}, "float64");
  EXPECT_EQ(dbuf.size(), 3);
  EXPECT_EQ(dbuf.dtype(), "float64");

  // Empty initialization
  Buffer empty_buf({}, "int32");
  EXPECT_EQ(empty_buf.size(), 0);
  EXPECT_EQ(empty_buf.dtype(), "int32");

  // Type conversion during initialization
  Buffer conv_buf({1.1f, 2.9f, 3.5f}, "int32");
  EXPECT_EQ(conv_buf.size(), 3);
  EXPECT_EQ(conv_buf.dtype(), "int32");
  // Values should be converted from float to int
}

// Test py::buffer constructor
TEST(BufferTest, PyBufferConstructor) {
  py::scoped_interpreter guard{};

  // Create a NumPy array through Python
  py::module np = py::module::import("numpy");

  // Test with float32 array
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

//   // Float buffers with same value
//   Buffer f32buf(5, "float32", py::cast(3.14f));
//   for (std::size_t i = 0; i < f32buf.size(); ++i) {
//     EXPECT_NEAR(py::cast<float>(f32buf.get_item(i)), 3.14f, 1e-6f);
//   }

//   Buffer f64buf(3, "float64", py::cast(2.71828));
//   for (std::size_t i = 0; i < f64buf.size(); ++i) {
//     EXPECT_NEAR(py::cast<double>(f64buf.get_item(i)), 2.71828, 1e-9);
//   }

//   // float16 buffer test
//   Buffer f16buf(4, "float16", py::cast(1.5f));
//   for (std::size_t i = 0; i < f16buf.size(); ++i) {
//     EXPECT_NEAR(py::cast<float>(f16buf.get_item(i)), 1.5f, 1e-3f);
//   }

//   // Integer buffers with same value
//   Buffer i32buf(4, "int32", py::cast(42));
//   for (std::size_t i = 0; i < i32buf.size(); ++i) {
//     EXPECT_EQ(py::cast<int>(i32buf.get_item(i)), 42);
//   }

//   // Use a large negative int64_t value that can be reliably represented
//   int64_t large_neg_value = -9223372036854775800LL;
//   Buffer i64buf(4, "int64", py::cast(large_neg_value));
//   for (std::size_t i = 0; i < i64buf.size(); ++i) {
//     EXPECT_EQ(py::cast<int64_t>(i64buf.get_item(i)), large_neg_value);
//   }

//   Buffer u8buf(3, "uint8", py::cast(uint8_t(255)));
//   for (std::size_t i = 0; i < u8buf.size(); ++i) {
//     EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(i)), uint8_t(255));
//   }

//   // Boolean buffers with different representations
//   Buffer bbuf1(5, "bool", py::cast(true));
//   for (std::size_t i = 0; i < bbuf1.size(); ++i) {
//     EXPECT_TRUE(py::cast<bool>(bbuf1.get_item(i)));
//   }

//   Buffer bbuf2(5, "bool", py::cast(1));
//   for (std::size_t i = 0; i < bbuf2.size(); ++i) {
//     EXPECT_TRUE(py::cast<bool>(bbuf2.get_item(i)));
//   }

//   Buffer bbuf3(5, "bool", py::cast(false));
//   for (std::size_t i = 0; i < bbuf3.size(); ++i) {
//     EXPECT_FALSE(py::cast<bool>(bbuf3.get_item(i)));
//   }

//   Buffer bbuf4(5, "bool", py::cast(0));
//   for (std::size_t i = 0; i < bbuf4.size(); ++i) {
//     EXPECT_FALSE(py::cast<bool>(bbuf4.get_item(i)));
//   }
// }

// // Out-of-bounds accesses should trigger a debug assertion
// TEST(BufferDeathTest, GetItemOutOfBounds) {
//   py::scoped_interpreter guard{};
//   Buffer buf(1, "float32");
//   EXPECT_DEATH({ buf.get_item(2); }, "index");
// }

// // Additional tests for boolean buffer initialization
// TEST(BufferTest, BooleanBufferConversions) {
//   py::scoped_interpreter guard{};

//   // Test conversion from floating point to boolean
//   Buffer bbuf1(3, "bool", py::cast(0.0f));
//   for (std::size_t i = 0; i < bbuf1.size(); ++i) {
//     EXPECT_FALSE(py::cast<bool>(bbuf1.get_item(i)));
//   }

//   Buffer bbuf2(3, "bool", py::cast(1.0f));
//   for (std::size_t i = 0; i < bbuf2.size(); ++i) {
//     EXPECT_TRUE(py::cast<bool>(bbuf2.get_item(i)));
//   }

//   Buffer bbuf3(3, "bool", py::cast(0.1));
//   for (std::size_t i = 0; i < bbuf3.size(); ++i) {
//     EXPECT_TRUE(py::cast<bool>(bbuf3.get_item(i)));
//   }

//   Buffer bbuf4(3, "bool", py::cast(-1));
//   for (std::size_t i = 0; i < bbuf4.size(); ++i) {
//     EXPECT_TRUE(py::cast<bool>(bbuf4.get_item(i)));
//   }
// }

// // Test for float16 buffer specifically
// TEST(BufferTest, Float16Buffer) {
//   py::scoped_interpreter guard{};

//   Buffer f16buf(5, "float16");
//   EXPECT_EQ(f16buf.get_dtype(), "float16");
//   EXPECT_EQ(f16buf.size(), 5);

//   // Test value initialization and conversion
//   Buffer f16buf_val(3, "float16", py::cast(3.14159f));
//   for (std::size_t i = 0; i < f16buf_val.size(); ++i) {
//     // we need a larger epsilon, since precision float16 << float32
//     float val = py::cast<float>(f16buf_val.get_item(i));
//     EXPECT_NEAR(val, 3.14159f, 5e-3f); // Higher epsilon for float16
//   }

//   // Test limits of float16 (±65504)
//   Buffer f16buf_max(1, "float16", py::cast(65504.0f));
//   float max_val = py::cast<float>(f16buf_max.get_item(0));
//   EXPECT_NEAR(max_val, 65504.0f, 1.0f);

//   f16buf.set_item(0, 42.5);
//   float read_val = py::cast<float>(f16buf.get_item(0));
//   EXPECT_NEAR(read_val, 42.5f, 1e-2f);
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
