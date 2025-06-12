// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <variant>

namespace py = pybind11;

// --- VecBuffer ---
template <typename T> class VecBuffer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Array = Eigen::Array<T, Eigen::Dynamic, 1>;

  explicit VecBuffer(std::size_t n = 0);
  VecBuffer(const T *src, std::size_t n);

  T &operator[](std::size_t i);
  const T &operator[](std::size_t i) const;
  [[nodiscard]] std::size_t size() const;
  T *data();
  [[nodiscard]] const T *data() const;

  VecBuffer &operator+=(const VecBuffer &rhs);
  VecBuffer &operator-=(const VecBuffer &rhs);
  [[nodiscard]] VecBuffer cwiseMul(const VecBuffer &rhs) const;
  [[nodiscard]] T dot(const VecBuffer &rhs) const;

  friend VecBuffer operator+(const VecBuffer &lhs, const VecBuffer &rhs) {
    VecBuffer result = lhs;
    result += rhs;
    return result;
  }

  friend VecBuffer operator-(const VecBuffer &lhs, const VecBuffer &rhs) {
    VecBuffer result = lhs;
    result -= rhs;
    return result;
  }

  Eigen::Ref<Array> ref();
  [[nodiscard]] Eigen::Ref<const Array> ref() const;

private:
  Array data_;
  explicit VecBuffer(const Array &a);
};

// Element-wise operations
enum class EwOp : uint8_t { ADD, SUB, MUL, DIV };

// Supported data types
enum class DType : uint8_t {
  BOOL,
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
  FLOAT64
};

// Get DType from string representation
DType dtype_from_string(std::string_view dtype_str);
std::string dtype_to_string(DType dtype);

// Buffer class that can hold vectors of different types
class Buffer {
public:
  using BufferVariant =
      std::variant<VecBuffer<bool>, VecBuffer<int8_t>, VecBuffer<uint8_t>,
                   VecBuffer<int16_t>, VecBuffer<uint16_t>, VecBuffer<int32_t>,
                   VecBuffer<uint32_t>, VecBuffer<int64_t>, VecBuffer<uint64_t>,
                   VecBuffer<Eigen::half>, VecBuffer<float>, VecBuffer<double>>;

  // Constructors
  explicit Buffer(std::size_t size, std::string_view dtype_str);
  Buffer(std::size_t size, std::string_view dtype_str, const py::object &val);
  Buffer(const py::sequence &seq, std::string_view dtype_str);

  // Basic operations
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] py::object get_item(std::size_t i) const;
  void set_item(std::size_t i, double val);
  [[nodiscard]] std::string get_dtype() const;
  [[nodiscard]] DType dtype() const;

  // Element-wise operations
  [[nodiscard]] Buffer elementwise_op(const Buffer &other, EwOp op,
                                      std::string_view out_dtype = "") const;
  [[nodiscard]] Buffer add(const Buffer &other,
                           std::string_view out_dtype = "") const;
  [[nodiscard]] Buffer sub(const Buffer &other,
                           std::string_view out_dtype = "") const;
  [[nodiscard]] Buffer mul(const Buffer &other,
                           std::string_view out_dtype = "") const;
  [[nodiscard]] Buffer div(const Buffer &other,
                           std::string_view out_dtype = "") const;

  [[nodiscard]] const BufferVariant &get_buffer() const { return buffer_; }
  BufferVariant &get_buffer() { return buffer_; }

private:
  void initialize_buffer(std::size_t size, DType dtype);
  BufferVariant buffer_;
  DType dtype_;
};

// Standalone operations
[[nodiscard]] Buffer add(const Buffer &lhs, const Buffer &rhs,
                         std::string_view out_dtype = "");
[[nodiscard]] Buffer sub(const Buffer &lhs, const Buffer &rhs,
                         std::string_view out_dtype = "");
[[nodiscard]] Buffer mul(const Buffer &lhs, const Buffer &rhs,
                         std::string_view out_dtype = "");
[[nodiscard]] Buffer div(const Buffer &lhs, const Buffer &rhs,
                         std::string_view out_dtype = "");
