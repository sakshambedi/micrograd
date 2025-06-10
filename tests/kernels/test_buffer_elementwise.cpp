// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>
#include <pybind11/embed.h>
namespace py = pybind11;

// Forward declare the initialization function so we can call it directly in our
// tests
void init_kernel_table();

class BufferAddTestSetup : public ::testing::Test {
protected:
  void SetUp() override {
    // Explicitly initialize the kernel table before each test
    init_kernel_table();
  }
};

TEST_F(BufferAddTestSetup, SameTypeAddition) {
  // Initialize Python interpreter
  py::scoped_interpreter guard{};

  // Ensure kernel table is initialized
  init_kernel_table();

  Buffer buf1(4, "float32");
  Buffer buf2(4, "float32");

  for (size_t i = 0; i < 4; ++i) {
    buf1.set_item(i, static_cast<float>(i));
    buf2.set_item(i, static_cast<float>(2 * i + 1));
  }

  Buffer result = add(buf1, buf2);
  EXPECT_EQ(result.get_dtype(), "float32");
  EXPECT_EQ(result.size(), 4);

  for (size_t i = 0; i < 4; ++i) {
    float expected = static_cast<float>(i) + static_cast<float>(2 * i + 1);
    double result_val = result.get_item(i).cast<double>();
    EXPECT_FLOAT_EQ(result_val, expected);

    // Verify original buffers are unchanged
    EXPECT_FLOAT_EQ(buf1.get_item(i).cast<double>(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(buf2.get_item(i).cast<double>(),
                    static_cast<float>(2 * i + 1));
  }
  // Test int32 addition
  {
    Buffer buf1(3, "int32");
    Buffer buf2(3, "int32");

    buf1.set_item(0, 10);
    buf1.set_item(1, 20);
    buf1.set_item(2, 30);

    buf2.set_item(0, 1);
    buf2.set_item(1, 2);
    buf2.set_item(2, 3);

    Buffer result = add(buf1, buf2);
    EXPECT_EQ(result.get_dtype(), "int32");
    EXPECT_EQ(result.size(), 3);

    EXPECT_EQ(result.get_item(0).cast<int>(), 11);
    EXPECT_EQ(result.get_item(1).cast<int>(), 22);
    EXPECT_EQ(result.get_item(2).cast<int>(), 33);
  }

  // Test float64 (double) addition
  {
    Buffer buf1(3, "float64");
    Buffer buf2(3, "float64");

    buf1.set_item(0, 1.5);
    buf1.set_item(1, 2.5);
    buf1.set_item(2, 3.5);

    buf2.set_item(0, 0.5);
    buf2.set_item(1, 1.0);
    buf2.set_item(2, 1.5);

    Buffer result = add(buf1, buf2);
    EXPECT_EQ(result.get_dtype(), "float64");

    EXPECT_DOUBLE_EQ(result.get_item(0).cast<double>(), 2.0);
    EXPECT_DOUBLE_EQ(result.get_item(1).cast<double>(), 3.5);
    EXPECT_DOUBLE_EQ(result.get_item(2).cast<double>(), 5.0);
  }

  // Test bool addition
  {
    Buffer buf1(4, "bool");
    Buffer buf2(4, "bool");

    buf1.set_item(0, 0); // false
    buf1.set_item(1, 1); // true
    buf1.set_item(2, 0); // false
    buf1.set_item(3, 1); // true

    buf2.set_item(0, 0); // false
    buf2.set_item(1, 0); // false
    buf2.set_item(2, 1); // true
    buf2.set_item(3, 1); // true

    Buffer result = add(buf1, buf2);
    EXPECT_EQ(result.get_dtype(), "bool");

    EXPECT_EQ(result.get_item(0).cast<bool>(), false); // false + false = false
    EXPECT_EQ(result.get_item(1).cast<bool>(), true);  // true + false = true
    EXPECT_EQ(result.get_item(2).cast<bool>(), true);  // false + true = true
    EXPECT_EQ(result.get_item(3).cast<bool>(), true);  // true + true = true
  }

  // Test empty buffers
  {
    Buffer buf1(0, "float32");
    Buffer buf2(0, "float32");

    Buffer result = add(buf1, buf2);
    EXPECT_EQ(result.size(), 0);
    EXPECT_EQ(result.get_dtype(), "float32");
  }
}

TEST_F(BufferAddTestSetup, DifferentTypeAddition) {
  // Initialize Python interpreter
  py::scoped_interpreter guard{};

  // Ensure kernel table is initialized
  init_kernel_table();

  // Test int32 + float32 -> float32
  {
    Buffer buf1(3, "int32");
    Buffer buf2(3, "float32");

    buf1.set_item(0, 10);
    buf1.set_item(1, 20);
    buf1.set_item(2, 30);

    buf2.set_item(0, 0.5);
    buf2.set_item(1, 1.5);
    buf2.set_item(2, 2.5);

    Buffer result = add(buf1, buf2, "float32");
    EXPECT_EQ(result.get_dtype(), "float32");

    EXPECT_FLOAT_EQ(result.get_item(0).cast<float>(), 10.5f);
    EXPECT_FLOAT_EQ(result.get_item(1).cast<float>(), 21.5f);
    EXPECT_FLOAT_EQ(result.get_item(2).cast<float>(), 32.5f);
  }

  // Test float32 + float64 -> float64 (upcast)
  {
    Buffer buf1(3, "float32");
    Buffer buf2(3, "float64");

    buf1.set_item(0, 1.25f);
    buf1.set_item(1, 2.25f);
    buf1.set_item(2, 3.25f);

    buf2.set_item(0, 0.75);
    buf2.set_item(1, 1.75);
    buf2.set_item(2, 2.75);

    Buffer result = add(buf1, buf2, "float64");
    EXPECT_EQ(result.get_dtype(), "float64");

    EXPECT_DOUBLE_EQ(result.get_item(0).cast<double>(), 2.0);
    EXPECT_DOUBLE_EQ(result.get_item(1).cast<double>(), 4.0);
    EXPECT_DOUBLE_EQ(result.get_item(2).cast<double>(), 6.0);
  }

  // Test int8 + uint8 -> int16 (cross type with explicit output type)
  {
    Buffer buf1(3, "int8");
    Buffer buf2(3, "uint8");

    buf1.set_item(0, 10);
    buf1.set_item(1, 20);
    buf1.set_item(2, 30);

    buf2.set_item(0, 5);
    buf2.set_item(1, 10);
    buf2.set_item(2, 15);

    EXPECT_NO_THROW({
      Buffer result = add(buf1, buf2, "int16");
      EXPECT_EQ(result.get_dtype(), "int16");

      EXPECT_EQ(result.get_item(0).cast<int16_t>(), 15);
      EXPECT_EQ(result.get_item(1).cast<int16_t>(), 30);
      EXPECT_EQ(result.get_item(2).cast<int16_t>(), 45);
    });
  }

  // Test bool + int32 -> int32
  {
    Buffer buf1(4, "bool");
    Buffer buf2(4, "int32");

    buf1.set_item(0, 0); // false
    buf1.set_item(1, 1); // true
    buf1.set_item(2, 0); // false
    buf1.set_item(3, 1); // true

    buf2.set_item(0, 100);
    buf2.set_item(1, 200);
    buf2.set_item(2, 300);
    buf2.set_item(3, 400);

    EXPECT_NO_THROW({
      Buffer result = add(buf1, buf2, "int32");
      EXPECT_EQ(result.get_dtype(), "int32");

      EXPECT_EQ(result.get_item(0).cast<int32_t>(), 100); // false + 100 = 100
      EXPECT_EQ(result.get_item(1).cast<int32_t>(), 201); // true + 200 = 201
      EXPECT_EQ(result.get_item(2).cast<int32_t>(), 300); // false + 300 = 300
      EXPECT_EQ(result.get_item(3).cast<int32_t>(), 401); // true + 400 = 401
    });
  }

  // Test float16 + float32 -> float32
  {
    Buffer buf1(3, "float16");
    Buffer buf2(3, "float32");

    buf1.set_item(0, 1.0);
    buf1.set_item(1, 2.0);
    buf1.set_item(2, 3.0);

    buf2.set_item(0, 0.5);
    buf2.set_item(1, 1.5);
    buf2.set_item(2, 2.5);

    EXPECT_NO_THROW({
      Buffer result = add(buf1, buf2, "float32");
      EXPECT_EQ(result.get_dtype(), "float32");

      // Using near because float16 has limited precision
      EXPECT_NEAR(result.get_item(0).cast<float>(), 1.5f, 0.01f);
      EXPECT_NEAR(result.get_item(1).cast<float>(), 3.5f, 0.01f);
      EXPECT_NEAR(result.get_item(2).cast<float>(), 5.5f, 0.01f);
    });
  }
}

TEST_F(BufferAddTestSetup, ErrorCases) {
  // Initialize Python interpreter
  py::scoped_interpreter guard{};

  // Ensure kernel table is initialized
  init_kernel_table();

  // Test size mismatch
  {
    Buffer buf1(3, "float32");
    Buffer buf2(4, "float32");

    EXPECT_THROW(add(buf1, buf2), std::runtime_error);
  }

  // Test with invalid output dtype
  {
    Buffer buf1(3, "float32");
    Buffer buf2(3, "float32");

    EXPECT_THROW(add(buf1, buf2, "invalid_dtype"), std::runtime_error);
  }
}
