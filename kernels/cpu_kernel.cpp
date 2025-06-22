// Copyright 2025 Saksham Bedi

#include "cpu_kernel.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace py = pybind11;

static const std::unordered_map<std::string, DType> str_to_dtype = {
    // {"bool", DType::BOOL},    // Commented out bool type for now
    {"int8", DType::INT8},       {"uint8", DType::UINT8},
    {"int16", DType::INT16},     {"uint16", DType::UINT16},
    {"int32", DType::INT32},     {"uint32", DType::UINT32},
    {"int64", DType::INT64},     {"uint64", DType::UINT64},
    {"float16", DType::FLOAT16}, {"float32", DType::FLOAT32},
    {"float64", DType::FLOAT64},
};

DType dtype_from_string(const std::string &s) {
  auto it = str_to_dtype.find(s);
  if (it == str_to_dtype.end())
    throw std::runtime_error("unknown dtype");
  return it->second;
}

std::string dtype_to_string(DType t) {
  for (auto &kv : str_to_dtype)
    if (kv.second == t)
      return kv.first;
  return "";
}

// -----------------------------------------------------------------------------
Buffer::Buffer(std::size_t n, const std::string &dtype) {
  init(n, dtype_from_string(dtype));
}

Buffer::Buffer(const py::buffer &view, const std::string &dtype) {
  auto info = view.request();
  init(info.size, dtype_from_string(dtype));
  std::visit(
      [&](auto &buf) {
        using T = std::decay_t<decltype(buf[0])>;
        const T *src = static_cast<const T *>(info.ptr);
        std::copy(src, src + info.size, buf.data());
      },
      data_);
}

std::size_t Buffer::size() const {
  return std::visit([](auto &b) { return b.size(); }, data_);
}

const BufferVariant &Buffer::raw() const { return data_; }
BufferVariant &Buffer::raw() { return data_; }

py::dict Buffer::array_interface() const {
  py::dict d;
  std::visit(
      [&, this](auto &b) {
        using T = std::decay_t<decltype(b[0])>;
        d["shape"] = py::make_tuple(b.size());
        if constexpr (std::is_same_v<T, uint8_t>)
          // if (this->dtype_ == DType::BOOL)
          //   d["typestr"] = "|b1"; // NumPy bool
          // else
          d["typestr"] = "|u1"; // uint8
        else if constexpr (std::is_same_v<T, int8_t>)
          d["typestr"] = "|i1";
        else if constexpr (std::is_same_v<T, int16_t>)
          d["typestr"] = "<i2";
        else if constexpr (std::is_same_v<T, uint16_t>)
          d["typestr"] = "<u2";
        else if constexpr (std::is_same_v<T, int32_t>)
          d["typestr"] = "<i4";
        else if constexpr (std::is_same_v<T, uint32_t>)
          d["typestr"] = "<u4";
        else if constexpr (std::is_same_v<T, int64_t>)
          d["typestr"] = "<i8";
        else if constexpr (std::is_same_v<T, uint64_t>)
          d["typestr"] = "<u8";
        else if constexpr (std::is_same_v<T, half>)
          d["typestr"] = "<f2";
        else if constexpr (std::is_same_v<T, float>)
          d["typestr"] = "<f4";
        else if constexpr (std::is_same_v<T, double>)
          d["typestr"] = "<f8";
        d["data"] =
            py::make_tuple(reinterpret_cast<std::uintptr_t>(b.data()), false);
      },
      data_);
  d["version"] = 3;
  return d;
}

std::string Buffer::dtype() const { return dtype_to_string(dtype_); }

void Buffer::init(std::size_t n, DType t) {
  dtype_ = t;
  switch (t) {
  // case DType::BOOL:
  //   data_ = VecBuffer<int8_t>(n);
  //   break;
  case DType::INT8:
    data_ = VecBuffer<int8_t>(n);
    break;
  case DType::UINT8:
    data_ = VecBuffer<uint8_t>(n);
    break;
  case DType::INT16:
    data_ = VecBuffer<int16_t>(n);
    break;
  case DType::UINT16:
    data_ = VecBuffer<uint16_t>(n);
    break;
  case DType::INT32:
    data_ = VecBuffer<int32_t>(n);
    break;
  case DType::UINT32:
    data_ = VecBuffer<uint32_t>(n);
    break;
  case DType::INT64:
    data_ = VecBuffer<int64_t>(n);
    break;
  case DType::UINT64:
    data_ = VecBuffer<uint64_t>(n);
    break;
  case DType::FLOAT16:
    data_ = VecBuffer<half>(n);
    break; // or half if using half.hpp
  case DType::FLOAT32:
    data_ = VecBuffer<float>(n);
    break;
  case DType::FLOAT64:
    data_ = VecBuffer<double>(n);
    break;
  default:
    throw std::runtime_error("bad dtype");
  }
}

std::string Buffer::repr() const {
  std::ostringstream oss;
  oss << "Buffer(dtype=" << dtype_to_string(dtype_) << ", size=" << size()
      << ", data=[";

  const size_t max_display_elements = 10;
  const bool truncated = size() > max_display_elements;
  const size_t display_count = truncated ? max_display_elements : size();

  std::visit(
      [&](auto &b) {
        using T = std::decay_t<decltype(b[0])>;

        for (size_t i = 0; i < display_count; ++i) {
          if (i > 0)
            oss << ", ";

          if constexpr (std::is_same_v<T, int8_t>)
            oss << static_cast<int>(b[i]); // Display as number, not char
          else if constexpr (std::is_same_v<T, uint8_t>)
            oss << static_cast<unsigned>(b[i]); // Display as number, not char
          else if constexpr (std::is_same_v<T, half>)
            oss << static_cast<float>(b[i]) << "f";
          else if constexpr (std::is_same_v<T, float>)
            oss << b[i] << "f";
          else if constexpr (std::is_same_v<T, double>)
            oss << b[i];
          else
            oss << b[i];
        }

        if (truncated) {
          oss << ", ...";
        }
      },
      data_);
  oss << "])";
  return oss.str();
}

Buffer Buffer::cast(const std::string &new_dtype) const {
  DType target_dtype = dtype_from_string(new_dtype);
  if (target_dtype == dtype_) {
    return *this;
  }

  Buffer result(size(), new_dtype);

  std::visit(
      [&](const auto &src_buf) {
        using SrcType = std::decay_t<decltype(src_buf[0])>;

        std::visit(
            [&](auto &dst_buf) {
              using DstType = std::decay_t<decltype(dst_buf[0])>;

              if constexpr (!std::is_same_v<SrcType, DstType>) {
                // Use VecBuffer's cast method directly
                dst_buf = src_buf.template cast<DstType>();
              }
            },
            result.raw());
      },
      data_);

  return result;
}

py::object Buffer::get_item(size_t index) const {
  if (index >= size()) {
    throw std::out_of_range("Buffer index out of range");
  }

  return std::visit(
      [index](auto &b) -> py::object {
        using ValueType = std::decay_t<decltype(b[0])>;
        if constexpr (std::is_same_v<ValueType, half>) {
          return py::cast(
              static_cast<float>(b[index])); // Convert half to float for Python
        } else {
          return py::cast(b[index]);
        }
      },
      data_);
}

// -----------------------------------------------------------------------------
PYBIND11_MODULE(cpu_kernel, m) {
  py::class_<Buffer>(m, "Buffer")
      .def(py::init([](py::list l, const std::string &dtype) {
        py::object np = py::module_::import("numpy");

        py::object arr = np.attr("array")(l, py::arg("dtype") = dtype);
        // Ensure the array is C-contiguous (row-major order)
        // arr = arr.attr("ravel")().attr("copy")();
        if (py::cast<int>(arr.attr("ndim")) > 1) {
          arr = arr.attr("flatten")();
        }

        return Buffer(arr, dtype);
      }))
      .def(py::init<std::size_t, const std::string &>())
      .def(py::init<py::buffer, const std::string &>())
      .def("size", &Buffer::size)
      .def("get_dtype", &Buffer::dtype)
      .def("__repr__", &Buffer::repr)
      .def("__getitem__", &Buffer::get_item)
      .def("get_item", &Buffer::get_item)
      .def("cast", &Buffer::cast)
      .def_property_readonly("__array_interface__", &Buffer::array_interface);
}
