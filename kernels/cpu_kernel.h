// Copyright 2025 Saksham Bedi

#ifndef KERNELS_CPU_KERNEL_H_
#define KERNELS_CPU_KERNEL_H_

#include "vecbuffer.h"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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
  Buffer(std::initializer_list<T> data, const std::string &dtype) {
    init(data.size(), dtype_from_string(dtype));

    // Convert the input data to the target type and copy it into our buffer
    std::visit(
        [&](auto &buf) {
          using DestT = std::decay_t<decltype(buf[0])>;
          size_t i = 0;
          for (const T &val : data) {
            buf[i++] = static_cast<DestT>(val);
          }
        },
        data_);
  }

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

  // Cast the buffer to a different data type
  [[nodiscard]] Buffer cast(const std::string &new_dtype) const;

private:
  void init(std::size_t n, DType t);
  BufferVariant data_;
  DType dtype_;
};

template <typename T>
inline void Buffer::set_item(size_t index, T value) {
  if (index >= size()) {
    throw std::out_of_range("Buffer index out of range");
  }

  std::visit(
      [&](auto &b) {
        using DestType = std::decay_t<decltype(b[0])>;
        b[index] = static_cast<DestType>(value);
      },
      data_);
}

// Buffer add(const Buffer &lhs, const Buffer &rhs,
//            const std::vector<std::size_t> &lhs_shape,
//            const std::vector<std::size_t> &rhs_shape,
//            const std::vector<std::size_t> &out_shape, const std::string
//            &dtype);

// No external cast_buffer function needed, VecBuffer has built-in cast
// functionality

#endif // KERNELS_CPU_KERNEL_H_
