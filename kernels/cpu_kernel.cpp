// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "cpu_kernel.h"

// #include <cstddef>
// #include <cstdint>
#include <stdexcept>
// #include <string_view>
#include <algorithm>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
// #include "Eigen/src/Core/Map.h"
// #include "Eigen/src/Core/Ref.h"
// #include "Eigen/src/Core/arch/Default/Half.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;

// pybind11 type_caster for Eigen::half
namespace pybind11::detail {

template <> struct type_caster<Eigen::half> {
  Eigen::half value;

public:
  static constexpr auto name = _<Eigen::half>();

  // Python -> C++
  bool load(handle src, bool convert) {
    if (!src)
      return false;
    if (PyFloat_Check(src.ptr())) {
      value = static_cast<Eigen::half>(PyFloat_AsDouble(src.ptr()));
      return true;
    }
    return false;
  }

  // C++ -> Python
  static handle cast(Eigen::half src, return_value_policy policy,
                     handle parent) {
    return PyFloat_FromDouble(static_cast<double>(src));
  }

  operator Eigen::half() { return value; }
};

} // namespace pybind11::detail

// --- VecBuffer Implementation ---
template <typename T> VecBuffer<T>::VecBuffer(std::size_t n) : data_(n) {}

template <typename T>
VecBuffer<T>::VecBuffer(const T *src, std::size_t n) : data_(n) {
  std::copy(src, src + n, data_.data());
}

template <typename T> T &VecBuffer<T>::operator[](std::size_t i) {
  return data_[i];
}

template <typename T> const T &VecBuffer<T>::operator[](std::size_t i) const {
  return data_[i];
}

template <typename T> std::size_t VecBuffer<T>::size() const {
  return data_.size();
}

template <typename T> T *VecBuffer<T>::data() { return data_.data(); }

template <typename T> const T *VecBuffer<T>::data() const {
  return data_.data();
}

template <typename T>
VecBuffer<T> &VecBuffer<T>::operator+=(const VecBuffer<T> &rhs) {
  data_ += rhs.data_;
  return *this;
}

template <typename T>
VecBuffer<T> &VecBuffer<T>::operator-=(const VecBuffer<T> &rhs) {
  data_ -= rhs.data_;
  return *this;
}

template <typename T>
VecBuffer<T> VecBuffer<T>::cwiseMul(const VecBuffer<T> &rhs) const {
  VecBuffer<T> result(size());
  result.data_ = data_ * rhs.data_;
  return result;
}

template <typename T> T VecBuffer<T>::dot(const VecBuffer<T> &rhs) const {
  return (data_ * rhs.data_).sum();
}

template <typename T>
Eigen::Ref<typename VecBuffer<T>::Array> VecBuffer<T>::ref() {
  return Eigen::Ref<Array>(data_);
}

template <typename T>
Eigen::Ref<const typename VecBuffer<T>::Array> VecBuffer<T>::ref() const {
  return Eigen::Ref<const Array>(data_);
}

template <typename T> VecBuffer<T>::VecBuffer(const Array &a) : data_(a) {}

// --- DType string conversion ---
DType dtype_from_string(std::string_view dtype_str) {
  static const std::unordered_map<std::string_view, DType> dtype_map = {
      // Full names
      {"bool", DType::BOOL},      {"int8", DType::INT8},
      {"uint8", DType::UINT8},    {"int16", DType::INT16},
      {"uint16", DType::UINT16},  {"int32", DType::INT32},
      {"uint32", DType::UINT32},  {"int64", DType::INT64},
      {"uint64", DType::UINT64},  {"float16", DType::FLOAT16},
      {"half", DType::FLOAT16},   {"float32", DType::FLOAT32},
      {"float", DType::FLOAT32},  {"float64", DType::FLOAT64},
      {"double", DType::FLOAT64},

      {"?", DType::BOOL},         {"b", DType::INT8},
      {"B", DType::UINT8},        {"h", DType::INT16},
      {"H", DType::UINT16},       {"i", DType::INT32},
      {"I", DType::UINT32},       {"q", DType::INT64},
      {"Q", DType::UINT64},       {"e", DType::FLOAT16},
      {"f", DType::FLOAT32},      {"d", DType::FLOAT64}};

  auto it = dtype_map.find(dtype_str);
  if (it == dtype_map.end()) {
    throw std::runtime_error(std::string("Unknown dtype: ") +
                             std::string(dtype_str));
  }
  return it->second;
}

std::string dtype_to_string(DType dtype) {
  switch (dtype) {
  case DType::BOOL:
    return "bool";
  case DType::INT8:
    return "int8";
  case DType::UINT8:
    return "uint8";
  case DType::INT16:
    return "int16";
  case DType::UINT16:
    return "uint16";
  case DType::INT32:
    return "int32";
  case DType::UINT32:
    return "uint32";
  case DType::INT64:
    return "int64";
  case DType::UINT64:
    return "uint64";
  case DType::FLOAT16:
    return "float16";
  case DType::FLOAT32:
    return "float32";
  case DType::FLOAT64:
    return "float64";
  default:
    throw std::runtime_error("Invalid dtype");
  }
}

// --- Buffer implementation ---
void Buffer::initialize_buffer(std::size_t size, DType dtype) {
  dtype_ = dtype;

  switch (dtype) {
  case DType::BOOL:
    buffer_ = VecBuffer<bool>(size);
    break;
  case DType::INT8:
    buffer_ = VecBuffer<int8_t>(size);
    break;
  case DType::UINT8:
    buffer_ = VecBuffer<uint8_t>(size);
    break;
  case DType::INT16:
    buffer_ = VecBuffer<int16_t>(size);
    break;
  case DType::UINT16:
    buffer_ = VecBuffer<uint16_t>(size);
    break;
  case DType::INT32:
    buffer_ = VecBuffer<int32_t>(size);
    break;
  case DType::UINT32:
    buffer_ = VecBuffer<uint32_t>(size);
    break;
  case DType::INT64:
    buffer_ = VecBuffer<int64_t>(size);
    break;
  case DType::UINT64:
    buffer_ = VecBuffer<uint64_t>(size);
    break;
  case DType::FLOAT16:
    buffer_ = VecBuffer<Eigen::half>(size);
    break;
  case DType::FLOAT32:
    buffer_ = VecBuffer<float>(size);
    break;
  case DType::FLOAT64:
    buffer_ = VecBuffer<double>(size);
    break;
  }
}

Buffer::Buffer(std::size_t size, std::string_view dtype_str) {
  initialize_buffer(size, dtype_from_string(dtype_str));
}

Buffer::Buffer(std::size_t size, std::string_view dtype_str,
               const py::object &val) {
  DType dtype = dtype_from_string(dtype_str);
  initialize_buffer(size, dtype);

  // Handle value initialization based on target type
  std::visit(
      [&val, size](auto &buf) {
        using T = std::decay_t<decltype(buf[0])>;

        // Choose appropriate casting method based on type
        T value;

        if constexpr (std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, uint64_t>) {
          // For 64-bit integers, cast directly from Python to avoid precision
          // loss
          value = val.cast<T>();
        } else if constexpr (std::is_integral_v<T>) {
          // For other integer types
          value = static_cast<T>(val.cast<int>());
        } else if constexpr (std::is_same_v<T, bool>) {
          // Special handling for boolean
          value = val.cast<bool>();
        } else {
          // For floating point types
          value = static_cast<T>(val.cast<double>());
        }

        // Set all elements to the same value
        for (std::size_t i = 0; i < size; ++i) {
          buf[i] = value;
        }
      },
      buffer_);
}

Buffer::Buffer(const py::sequence &seq, std::string_view dtype_str) {
  DType dtype = dtype_from_string(dtype_str);
  const std::size_t n = seq.size();
  initialize_buffer(n, dtype);

  // Fill the buffer from the sequence
  std::visit(
      [&seq, n](auto &buf) {
        using T = std::decay_t<decltype(buf[0])>;
        for (std::size_t i = 0; i < n; ++i) {
          buf[i] = static_cast<T>(py::cast<double>(seq[i]));
        }
      },
      buffer_);
}

std::size_t Buffer::size() const {
  return std::visit([](const auto &buf) { return buf.size(); }, buffer_);
}

py::object Buffer::get_item(std::size_t i) const {
  return std::visit([i](const auto &buf) { return py::cast(buf[i]); }, buffer_);
}

void Buffer::set_item(std::size_t i, double val) {
  std::visit(
      [i, val](auto &buf) {
        using T = std::decay_t<decltype(buf[0])>;
        buf[i] = static_cast<T>(val);
      },
      buffer_);
}

std::string Buffer::get_dtype() const { return dtype_to_string(dtype_); }

DType Buffer::dtype() const { return dtype_; }

// --- Element-wise operations ---
Buffer Buffer::elementwise_op(const Buffer &other, EwOp op,
                              std::string_view out_dtype_str) const {
  if (size() != other.size()) {
    throw std::runtime_error("Size mismatch in elementwise operation");
  }

  // Use the output dtype if provided, otherwise use this buffer's dtype
  DType out_dtype =
      out_dtype_str.empty() ? dtype_ : dtype_from_string(out_dtype_str);

  const std::size_t n = size();
  Buffer result(n, dtype_to_string(out_dtype));

  std::visit(
      [&](const auto &l_buf) {
        std::visit(
            [&](const auto &r_buf) {
              std::visit(
                  [&](auto &o_buf) {
                    using T_left = std::decay_t<decltype(l_buf[0])>;
                    using T_right = std::decay_t<decltype(r_buf[0])>;
                    using T_out = std::decay_t<decltype(o_buf[0])>;

                    for (std::size_t i = 0; i < n; ++i) {
                      T_left lval = l_buf[i];
                      T_right rval = r_buf[i];

                      // Handle all operations with safe type conversions
                      // Convert to double for safe intermediate calculation
                      double lval_d = static_cast<double>(lval);
                      double rval_d = static_cast<double>(rval);
                      double result_d = 0.0;

                      switch (op) {
                      case EwOp::ADD:
                        result_d = lval_d + rval_d;
                        break;
                      case EwOp::SUB:
                        result_d = lval_d - rval_d;
                        break;
                      case EwOp::MUL:
                        result_d = lval_d * rval_d;
                        break;
                      case EwOp::DIV:
                        // Avoid division by zero
                        if (rval_d == 0.0) {
                          throw std::runtime_error("Division by zero");
                        }
                        result_d = lval_d / rval_d;
                        break;
                      }

                      o_buf[i] = static_cast<T_out>(result_d);
                    }
                  },
                  result.get_buffer());
            },
            other.get_buffer());
      },
      buffer_);

  return result;
}

Buffer Buffer::add(const Buffer &other, std::string_view out_dtype) const {
  return elementwise_op(other, EwOp::ADD, out_dtype);
}

Buffer Buffer::sub(const Buffer &other, std::string_view out_dtype) const {
  return elementwise_op(other, EwOp::SUB, out_dtype);
}

Buffer Buffer::mul(const Buffer &other, std::string_view out_dtype) const {
  return elementwise_op(other, EwOp::MUL, out_dtype);
}

Buffer Buffer::div(const Buffer &other, std::string_view out_dtype) const {
  return elementwise_op(other, EwOp::DIV, out_dtype);
}

// Standalone operation functions
Buffer add(const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
  return lhs.add(rhs, out_dtype);
}

Buffer sub(const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
  return lhs.sub(rhs, out_dtype);
}

Buffer mul(const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
  return lhs.mul(rhs, out_dtype);
}

Buffer div(const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
  return lhs.div(rhs, out_dtype);
}

// --- Explicit template instantiations ---
template class VecBuffer<bool>;
template class VecBuffer<int8_t>;
template class VecBuffer<uint8_t>;
template class VecBuffer<int16_t>;
template class VecBuffer<uint16_t>;
template class VecBuffer<int32_t>;
template class VecBuffer<uint32_t>;
template class VecBuffer<int64_t>;
template class VecBuffer<uint64_t>;
template class VecBuffer<Eigen::half>;
template class VecBuffer<float>;
template class VecBuffer<double>;

// --- Python module bindings ---
PYBIND11_MODULE(cpu_kernel, m) {
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<py::sequence, std::string_view>(), py::arg("sequence"),
           py::arg("dtype"))
      .def(py::init<std::size_t, std::string_view>(), py::arg("size"),
           py::arg("dtype"))
      .def(py::init<std::size_t, std::string_view, py::object>(),
           py::arg("size"), py::arg("dtype"), py::arg("value"))
      .def("__getitem__", &Buffer::get_item)
      .def("__setitem__", &Buffer::set_item)
      .def("__len__", &Buffer::size)
      .def("size", &Buffer::size)
      .def("dtype", &Buffer::get_dtype)
      .def("__add__", [](const Buffer &self,
                         const Buffer &other) { return add(self, other, ""); })
      .def("__sub__", [](const Buffer &self,
                         const Buffer &other) { return sub(self, other, ""); })
      .def("__mul__", [](const Buffer &self,
                         const Buffer &other) { return mul(self, other, ""); })
      .def("__truediv__", [](const Buffer &self, const Buffer &other) {
        return div(self, other, "");
      });

  m.def(
      "add",
      [](const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
        return add(lhs, rhs, out_dtype);
      },
      py::arg("lhs"), py::arg("rhs"), py::arg("out_dtype") = "");
  m.def(
      "sub",
      [](const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
        return sub(lhs, rhs, out_dtype);
      },
      py::arg("lhs"), py::arg("rhs"), py::arg("out_dtype") = "");
  m.def(
      "mul",
      [](const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
        return mul(lhs, rhs, out_dtype);
      },
      py::arg("lhs"), py::arg("rhs"), py::arg("out_dtype") = "");
  m.def(
      "div",
      [](const Buffer &lhs, const Buffer &rhs, std::string_view out_dtype) {
        return div(lhs, rhs, out_dtype);
      },
      py::arg("lhs"), py::arg("rhs"), py::arg("out_dtype") = "");
}
