#include "../../kernels/cpu_kernel.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

// Test Buffer size functionality
TEST(BufferTest, Size) {
  Buffer buf("float32", 5);
  EXPECT_EQ(buf.size(), 5);

  Buffer empty_buf("int32", 0);
  EXPECT_EQ(empty_buf.size(), 0);

  Buffer large_buf("bool", 1000);
  EXPECT_EQ(large_buf.size(), 1000);
}

// Test set_item with different overloads
TEST(BufferTest, SetItem) {
  // Test double overload
  Buffer float_buf("float32", 1);
  float_buf.set_item_double(0, 42.5);

  // Test int64 overload
  Buffer int_buf("int32", 1);
  int_buf.set_item_int64(0, 42);

  // Test bool overload
  Buffer bool_buf("bool", 1);
  bool_buf.set_item_bool(0, true);
}

// Test get_dtype functionality
TEST(BufferTest, GetDtype) {
  Buffer buf_f32("float32", 1);
  EXPECT_EQ(buf_f32.get_dtype(), "float32");

  Buffer buf_f64("float64
