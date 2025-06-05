#include "../../kernels/cpu_kernel.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <pybind11/embed.h>
namespace py = pybind11;

// Verify constructor and dtype reporting

TEST(BufferTest, ConstructorAndSize) {
  py::scoped_interpreter guard{};

  Buffer fbuf("float32", 5);
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.get_dtype(), "float32");

  Buffer ibuf("int64", 3);
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.get_dtype(), "int64");

  Buffer bbuf("bool", 2);
  EXPECT_EQ(bbuf.size(), 2);
  EXPECT_EQ(bbuf.get_dtype(), "bool");
}

// Test set and get operations for various dtypes
TEST(BufferTest, SetAndGet) {
  py::scoped_interpreter guard{};

  Buffer fbuf("float32", 1);
  fbuf.set_item(0, 3.5);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 3.5f, 1e-6f);

  Buffer ibuf("int32", 1);
  ibuf.set_item(0, 7);
  EXPECT_EQ(py::cast<int>(ibuf.get_item(0)), 7);

  Buffer bbuf("bool", 1);
  bbuf.set_item(0, 1.0);
  EXPECT_TRUE(py::cast<bool>(bbuf.get_item(0)));
}

// Out-of-bounds accesses should trigger a debug assertion
TEST(BufferDeathTest, GetItemOutOfBounds) {
  py::scoped_interpreter guard{};
  Buffer buf("float32", 1);
  EXPECT_DEATH({ buf.get_item(2); }, "index");
}

TEST(BufferDeathTest, SetItemOutOfBounds) {
  Buffer buf("float32", 1);
  EXPECT_DEATH({ buf.set_item(3, 1.0); }, "index");
}

// Validate type conversions and data preservation
TEST(BufferTest, TypeConversions) {
  py::scoped_interpreter guard{};

  Buffer dbuf("float64", 1);
  dbuf.set_item(0, 3.14159);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(0)), 3.14159, 1e-9);

  Buffer ibuf("int64", 1);
  ibuf.set_item(0, -5);
  EXPECT_EQ(py::cast<long long>(ibuf.get_item(0)), -5);

  Buffer fbuf("float32", 1);
  fbuf.set_item(0, 42);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 42.0f, 1e-6f);

  fbuf.set_item(0, 5.5);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 5.5f, 1e-6f);
}
