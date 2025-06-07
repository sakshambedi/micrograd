// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <pybind11/embed.h>
namespace py = pybind11;

TEST(BufferTest, ConstructorAndSize) {
  py::scoped_interpreter guard{};

  Buffer fbuf(5, "float32");
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.get_dtype(), "float32");

  Buffer ibuf(3, "int64");
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.get_dtype(), "int64");

  Buffer bbuf(2, "bool");
  EXPECT_EQ(bbuf.size(), 2);
  EXPECT_EQ(bbuf.get_dtype(), "bool");
}

// Test set and get operations for various dtypes
TEST(BufferTest, SetAndGet) {
  py::scoped_interpreter guard{};

  Buffer fbuf(1, "float32");
  fbuf.set_item(0, 3.5);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 3.5f, 1e-6f);

  Buffer ibuf(1, "int32");
  ibuf.set_item(0, 7);
  EXPECT_EQ(py::cast<int>(ibuf.get_item(0)), 7);

  Buffer bbuf(1, "bool");
  bbuf.set_item(0, 1.0);
  EXPECT_TRUE(py::cast<bool>(bbuf.get_item(0)));
}

// Test buffer initialization with same value across different dtypes
TEST(BufferTest, InitializeWithValue) {
  py::scoped_interpreter guard{};

  // Float buffers with same value
  Buffer f32buf(5, "float32", py::cast(3.14f));
  for (std::size_t i = 0; i < f32buf.size(); ++i) {
    EXPECT_NEAR(py::cast<float>(f32buf.get_item(i)), 3.14f, 1e-6f);
  }

  Buffer f64buf(3, "float64", py::cast(2.71828));
  for (std::size_t i = 0; i < f64buf.size(); ++i) {
    EXPECT_NEAR(py::cast<double>(f64buf.get_item(i)), 2.71828, 1e-9);
  }

  // float16 buffer test
  Buffer f16buf(4, "float16", py::cast(1.5f));
  for (std::size_t i = 0; i < f16buf.size(); ++i) {
    EXPECT_NEAR(py::cast<float>(f16buf.get_item(i)), 1.5f, 1e-3f);
  }

  // Integer buffers with same value
  Buffer i32buf(4, "int32", py::cast(42));
  for (std::size_t i = 0; i < i32buf.size(); ++i) {
    EXPECT_EQ(py::cast<int>(i32buf.get_item(i)), 42);
  }

  Buffer i64buf(4, "int64",
                py::cast(static_cast<int64_t>(-9223372036854775807)));
  for (std::size_t i = 0; i < i64buf.size(); ++i) {
    EXPECT_EQ(py::cast<int64_t>(i64buf.get_item(i)),
              static_cast<int64_t>(-9223372036854775807));
  }

  Buffer u8buf(3, "uint8", py::cast(uint8_t(255)));
  for (std::size_t i = 0; i < u8buf.size(); ++i) {
    EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(i)), uint8_t(255));
  }

  // Boolean buffers with different representations
  Buffer bbuf1(5, "bool", py::cast(true));
  for (std::size_t i = 0; i < bbuf1.size(); ++i) {
    EXPECT_TRUE(py::cast<bool>(bbuf1.get_item(i)));
  }

  Buffer bbuf2(5, "bool", py::cast(1));
  for (std::size_t i = 0; i < bbuf2.size(); ++i) {
    EXPECT_TRUE(py::cast<bool>(bbuf2.get_item(i)));
  }

  Buffer bbuf3(5, "bool", py::cast(false));
  for (std::size_t i = 0; i < bbuf3.size(); ++i) {
    EXPECT_FALSE(py::cast<bool>(bbuf3.get_item(i)));
  }

  Buffer bbuf4(5, "bool", py::cast(0));
  for (std::size_t i = 0; i < bbuf4.size(); ++i) {
    EXPECT_FALSE(py::cast<bool>(bbuf4.get_item(i)));
  }
}

// Out-of-bounds accesses should trigger a debug assertion
TEST(BufferDeathTest, GetItemOutOfBounds) {
  py::scoped_interpreter guard{};
  Buffer buf(1, "float32");
  EXPECT_DEATH({ buf.get_item(2); }, "index");
}

// Additional tests for boolean buffer initialization
TEST(BufferTest, BooleanBufferConversions) {
  py::scoped_interpreter guard{};

  // Test conversion from floating point to boolean
  Buffer bbuf1(3, "bool", py::cast(0.0f));
  for (std::size_t i = 0; i < bbuf1.size(); ++i) {
    EXPECT_FALSE(py::cast<bool>(bbuf1.get_item(i)));
  }

  Buffer bbuf2(3, "bool", py::cast(1.0f));
  for (std::size_t i = 0; i < bbuf2.size(); ++i) {
    EXPECT_TRUE(py::cast<bool>(bbuf2.get_item(i)));
  }

  Buffer bbuf3(3, "bool",
               py::cast(0.1)); // Non-zero value should convert to true
  for (std::size_t i = 0; i < bbuf3.size(); ++i) {
    EXPECT_TRUE(py::cast<bool>(bbuf3.get_item(i)));
  }

  Buffer bbuf4(3, "bool",
               py::cast(-1)); // Negative values should convert to true
  for (std::size_t i = 0; i < bbuf4.size(); ++i) {
    EXPECT_TRUE(py::cast<bool>(bbuf4.get_item(i)));
  }
}

// Test for float16 buffer specifically
TEST(BufferTest, Float16Buffer) {
  py::scoped_interpreter guard{};

  Buffer f16buf(5, "float16");
  EXPECT_EQ(f16buf.get_dtype(), "float16");
  EXPECT_EQ(f16buf.size(), 5);

  // Test value initialization and conversion
  Buffer f16buf_val(3, "float16", py::cast(3.14159f));
  for (std::size_t i = 0; i < f16buf_val.size(); ++i) {
    // we need a larger epsilon, since precision float16 << float32
    float val = py::cast<float>(f16buf_val.get_item(i));
    EXPECT_NEAR(val, 3.14159f, 5e-3f); // Higher epsilon for float16
  }

  // Test limits of float16 (±65504)
  Buffer f16buf_max(1, "float16", py::cast(65504.0f));
  float max_val = py::cast<float>(f16buf_max.get_item(0));
  EXPECT_NEAR(max_val, 65504.0f, 1.0f);

  f16buf.set_item(0, 42.5);
  float read_val = py::cast<float>(f16buf.get_item(0));
  EXPECT_NEAR(read_val, 42.5f, 1e-2f);
}

TEST(BufferDeathTest, SetItemOutOfBounds) {
  Buffer buf(1, "float32");
  EXPECT_DEATH({ buf.set_item(3, 1.0); }, "index");
}

// Validate type conversions and data preservation
TEST(BufferTest, TypeConversions) {
  py::scoped_interpreter guard{};

  Buffer dbuf(1, "float64");
  dbuf.set_item(0, 3.14159);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(0)), 3.14159, 1e-9);

  Buffer ibuf(1, "int64");
  ibuf.set_item(0, -5);
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(0)), -5);

  Buffer fbuf(1, "float32");
  fbuf.set_item(0, 42);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 42.0f, 1e-6f);

  fbuf.set_item(0, 5.5);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 5.5f, 1e-6f);
}
