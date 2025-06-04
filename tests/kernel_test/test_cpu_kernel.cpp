#include "../../kernels/cpu_kernel.h" // For direct access to Buffer
#include <arm_neon.h>
#include <gtest/gtest.h>

namespace py = pybind11;
TEST(BufferTest, Size) {
  Buffer buf("float32", 5);
  EXPECT_EQ(buf.size(), 5);
}

TEST(BufferTest, SetAndGet) {
  Buffer buf("float32", 1);
  buf.set_item(0, 42.0);
  // get_item returns a py::object, so cast to float
  // float value = py::cast<float>(buf.get_item(0));
  float32_t value = buf.get_item(0);
  EXPECT_EQ(value, 42.0);
}

TEST(BufferTest, Dtype) {
  Buffer buf("float32", 1);
  EXPECT_EQ(buf.get_dtype(), "float32");
}

// Add more tests as needed!
