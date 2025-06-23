#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace py = pybind11;

TEST(BufferMembersTest, RoundTripAndBounds) {
  py::scoped_interpreter guard{};

  Buffer fbuf(2, "float32");
  fbuf.set_item<float>(0, 1.25f);
  fbuf.set_item<float>(1, -2.5f);
  EXPECT_FLOAT_EQ(py::cast<float>(fbuf.get_item(0)), 1.25f);
  EXPECT_FLOAT_EQ(py::cast<float>(fbuf.get_item(1)), -2.5f);
  EXPECT_THROW(fbuf.get_item(2), std::out_of_range);
  EXPECT_THROW(fbuf.set_item<float>(2, 0.f), std::out_of_range);

  Buffer ibuf(1, "int32");
  ibuf.set_item<int32_t>(0, 42);
  EXPECT_EQ(py::cast<int32_t>(ibuf.get_item(0)), 42);
}

TEST(BufferMembersTest, ReprContainsInfo) {
  py::scoped_interpreter guard{};
  Buffer buf(3, "int64");
  std::string r = buf.repr();
  EXPECT_NE(r.find("dtype=int64"), std::string::npos);
  EXPECT_NE(r.find("size=3"), std::string::npos);
}

TEST(BufferMembersTest, ArrayInterfaceBasics) {
  py::scoped_interpreter guard{};
  Buffer buf(5, "float32");
  auto ai = buf.array_interface();
  EXPECT_EQ(ai["shape"].cast<py::tuple>()[0].cast<std::size_t>(), 5);
  EXPECT_EQ(ai["typestr"].cast<std::string>(), "<f4");

  Buffer ibuf(2, "int32");
  auto ai2 = ibuf.array_interface();
  EXPECT_EQ(ai2["typestr"].cast<std::string>(), "<i4");
}

TEST(BufferMembersTest, CastBetweenTypes) {
  py::scoped_interpreter guard{};
  Buffer fbuf({1.5f, -2.0f}, "float32");
  Buffer ibuf = fbuf.cast("int32");
  EXPECT_EQ(py::cast<int32_t>(ibuf.get_item(0)), static_cast<int32_t>(1.5f));
  EXPECT_EQ(py::cast<int32_t>(ibuf.get_item(1)), static_cast<int32_t>(-2.0f));

  Buffer i64buf({1, 2, 3}, "int64");
  Buffer f64buf = i64buf.cast("float64");
  EXPECT_DOUBLE_EQ(py::cast<double>(f64buf.get_item(0)), 1.0);
  EXPECT_DOUBLE_EQ(py::cast<double>(f64buf.get_item(1)), 2.0);
  EXPECT_DOUBLE_EQ(py::cast<double>(f64buf.get_item(2)), 3.0);
}

