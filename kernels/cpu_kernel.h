// Copyright 2025 Saksham Bedi

#ifndef KERNELS_CPU_KERNEL_H_
#define KERNELS_CPU_KERNEL_H_

#include "pybind11/pytypes.h"
#include "vecbuffer.h"
#include <algorithm>
#include <array>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <variant>

namespace py = pybind11;

enum class DType : uint8_t {
  // BOOL,  // Commented out bool type for now
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  NUM_TYPES
};

DType dtype_from_string(const std::string &s);

// DType -> string
std::string dtype_to_string(DType t);

using BufferVariant =
    std::variant</* VecBuffer<bool>, */ VecBuffer<int8_t>, VecBuffer<uint8_t>,
                 VecBuffer<int16_t>, VecBuffer<uint16_t>, VecBuffer<int32_t>,
                 VecBuffer<uint32_t>, VecBuffer<int64_t>, VecBuffer<uint64_t>,
                 VecBuffer<half>, VecBuffer<float>, VecBuffer<double>>;

struct Buffer {
  // Constructor for creating a buffer with specified size and dtype
  Buffer(std::size_t n, const std::string &dtype);

  // Constructor for initializing from data list (mainly for testing)
  template <typename T>
  Buffer(std::initializer_list<T> data, const std::string &dtype);

  // Constructor taking a Python buffer view
  Buffer(const py::buffer &view, const std::string &dtype);

  [[nodiscard]] std::size_t size() const;

  // Access to the raw buffer variant
  [[nodiscard]] const BufferVariant &raw() const;
  BufferVariant &raw();

  // Returns the NumPy array interface dictionary for interoperability
  [[nodiscard]] py::dict array_interface() const;

  [[nodiscard]] std::string dtype() const;
  [[nodiscard]] std::string repr() const;

  // Get item at the specified index
  [[nodiscard]] py::object get_item(size_t index) const;
  [[nodiscard]] py::object set_item(size_t index, py::object val) const;

  // Set item at the specified index
  template <typename T> void set_item(size_t index, T value);

private:
  void init(std::size_t n, DType t);
  BufferVariant data_;
  DType dtype_;
};

template <typename T>
Buffer::Buffer(std::initializer_list<T> data, const std::string &dtype) {
  init(data.size(), dtype_from_string(dtype));
  std::visit(
      [&](auto &buf) {
        using BufT = std::decay_t<decltype(buf[0])>;
        std::transform(data.begin(), data.end(), buf.data(),
                       [](const T &val) { return static_cast<BufT>(val); });
      },
      data_);
}

#endif // KERNELS_CPU_KERNEL_H_
