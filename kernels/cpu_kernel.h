// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include "pybind11/pytypes.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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

  // Standalone addition and subtraction operators
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
  // std::shared_ptr<Array> data_;
  explicit VecBuffer(const Array &a);
};

enum class EwOp : std::uint8_t { ADD, MUL, DIV, SUB, COUNT };

using KernFn = void (*)(const void *, const void *, void *, std::size_t);
constexpr std::size_t T = 13;

static std::array<
    std::array<std::array<std::array<KernFn, T>, T>, T>, // [op][lhs][rhs][out]
    static_cast<std::size_t>(EwOp::COUNT)>
    TABLE = {};

enum class DTypeEnum : std::uint8_t {
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
  FLOAT64,
  UNKNOWN
};

template <EwOp OP, class A, class B, class R>
static void ew_kernel(const void *a_raw, const void *b_raw, void *r_raw,
                      std::size_t n);

template <EwOp OP, class A, class B, class R>
static void reg(DTypeEnum a, DTypeEnum b, DTypeEnum r);

DTypeEnum get_dtype_enum(std::string_view dtype);

extern const std::unordered_map<DTypeEnum,
                                std::function<void(class Buffer &, size_t)>>
    factory_table;

class Buffer {
public:
  // Define BufferVariant type
  using BufferVariant =
      std::variant<VecBuffer<bool>, VecBuffer<std::int8_t>,
                   VecBuffer<std::uint8_t>, VecBuffer<int16_t>,
                   VecBuffer<uint16_t>, VecBuffer<int32_t>, VecBuffer<uint32_t>,
                   VecBuffer<int64_t>, VecBuffer<uint64_t>, VecBuffer<float>,
                   VecBuffer<double>, VecBuffer<Eigen::half>>;
  explicit Buffer(std::size_t size, const std::string &dtype);
  explicit Buffer(std::size_t size, const std::string &dtype, py::object val);
  explicit Buffer(py::sequence seq, std::string_view fmt);

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] py::object get_item(std::size_t i) const;
  void set_item(std::size_t i, double val);
  [[nodiscard]] std::string get_dtype() const;

  // elementwise operations
  template <EwOp OP>
  [[nodiscard]] Buffer ewise(const Buffer &other,
                             const std::string &out_dtype) const;
  [[nodiscard]] Buffer mul(const Buffer &other,
                           const std::string &out_dtype) const;
  [[nodiscard]] Buffer div(const Buffer &other,
                           const std::string &out_dtype) const;

  template <typename T> void set_buffer(std::size_t n) {
    buffer_ = VecBuffer<T>(n);
  }

  [[nodiscard]] const BufferVariant &get_buffer() const { return buffer_; }
  BufferVariant &get_buffer() { return buffer_; }

private:
  void initialize_buffer(std::size_t size, const std::string &dtype);
  void initialize_buffer(std::size_t size, std::string_view dtype);

  BufferVariant buffer_;
};

[[nodiscard]] Buffer add(const Buffer &lhs, const Buffer &rhs,
                         const std::string &out_dtype = "");
