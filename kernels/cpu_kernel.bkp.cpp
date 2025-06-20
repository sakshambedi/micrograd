// Copyright 2025 Saksham Bedi
#include "cpu_kernel.bkp.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

// ---------------------------- Eigen::half caster ----------------------------
namespace pybind11::detail {
template <> struct type_caster<Eigen::half> {
  Eigen::half value;
  static constexpr auto name = _<Eigen::half>();

  bool load(handle src, bool) {
    if (!src)
      return false;
    if (PyFloat_Check(src.ptr())) {
      value = static_cast<Eigen::half>(PyFloat_AsDouble(src.ptr()));
      return true;
    }
    return false;
  }
  static handle cast(Eigen::half src, return_value_policy, handle) {
    return PyFloat_FromDouble(static_cast<double>(src));
  }
  operator Eigen::half() { return value; }
};
} // namespace pybind11::detail

// ---------------------------- VecBuffer impl --------------------------------
template <typename T> VecBuffer<T>::VecBuffer(std::size_t n) : data_(n) {}
template <typename T>
VecBuffer<T>::VecBuffer(const T *src, std::size_t n) : data_(n) {
  std::copy(src, src + n, data_.data());
}
template <typename T> VecBuffer<T>::VecBuffer(const Array &a) : data_(a) {}

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
VecBuffer<T> &VecBuffer<T>::operator+=(const VecBuffer &rhs) {
  data_ += rhs.data_;
  return *this;
}
template <typename T>
VecBuffer<T> &VecBuffer<T>::operator-=(const VecBuffer &rhs) {
  data_ -= rhs.data_;
  return *this;
}

template <typename T>
VecBuffer<T> VecBuffer<T>::cwiseMul(const VecBuffer &rhs) const {
  VecBuffer<T> out(size());
  out.data_ = data_ * rhs.data_;
  return out;
}
template <typename T> T VecBuffer<T>::dot(const VecBuffer &rhs) const {
  return (data_ * rhs.data_).sum();
}

template <typename T>
Eigen::Ref<typename VecBuffer<T>::Array> VecBuffer<T>::ref() {
  return data_;
}
template <typename T>
Eigen::Ref<const typename VecBuffer<T>::Array> VecBuffer<T>::ref() const {
  return data_;
}

// -------------------------- dtype helpers -----------------------------------
DType dtype_from_string(std::string_view s) {
  static const std::unordered_map<std::string_view, DType> map = {
      {"bool", DType::BOOL},       {"?", DType::BOOL},
      {"int8", DType::INT8},       {"b", DType::INT8},
      {"uint8", DType::UINT8},     {"B", DType::UINT8},
      {"int16", DType::INT16},     {"h", DType::INT16},
      {"uint16", DType::UINT16},   {"H", DType::UINT16},
      {"int32", DType::INT32},     {"i", DType::INT32},
      {"uint32", DType::UINT32},   {"I", DType::UINT32},
      {"int64", DType::INT64},     {"q", DType::INT64},
      {"uint64", DType::UINT64},   {"Q", DType::UINT64},
      {"float16", DType::FLOAT16}, {"half", DType::FLOAT16},
      {"e", DType::FLOAT16},       {"float32", DType::FLOAT32},
      {"float", DType::FLOAT32},   {"f", DType::FLOAT32},
      {"float64", DType::FLOAT64}, {"double", DType::FLOAT64},
      {"d", DType::FLOAT64}};
  auto it = map.find(s);
  if (it == map.end())
    throw std::runtime_error("Unknown dtype: " + std::string(s));
  return it->second;
}
std::string dtype_to_string(DType t) {
  switch (t) {
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
  }
  throw std::runtime_error("Invalid dtype");
}

// -------------------- template binary_op (header-only) ----------------------
template <EwOp OP, typename L, typename R> constexpr auto binary_op(L l, R r) {
  using Common = std::common_type_t<L, R>;
  if constexpr (OP == EwOp::ADD)
    return static_cast<Common>(l) + static_cast<Common>(r);
  if constexpr (OP == EwOp::SUB)
    return static_cast<Common>(l) - static_cast<Common>(r);
  if constexpr (OP == EwOp::MUL)
    return static_cast<Common>(l) * static_cast<Common>(r);
  if constexpr (OP == EwOp::DIV) {
    if constexpr (std::is_integral_v<Common>)
      return static_cast<double>(l) / static_cast<double>(r);
    else
      return static_cast<Common>(l) / static_cast<Common>(r);
  }
}

// -------------------- Buffer internals ---------------------------------------
void Buffer::initialize_buffer(std::size_t n, DType dt) {
  dtype_ = dt;
  switch (dt) {
  case DType::BOOL:
    buffer_ = VecBuffer<bool>(n);
    break;
  case DType::INT8:
    buffer_ = VecBuffer<int8_t>(n);
    break;
  case DType::UINT8:
    buffer_ = VecBuffer<uint8_t>(n);
    break;
  case DType::INT16:
    buffer_ = VecBuffer<int16_t>(n);
    break;
  case DType::UINT16:
    buffer_ = VecBuffer<uint16_t>(n);
    break;
  case DType::INT32:
    buffer_ = VecBuffer<int32_t>(n);
    break;
  case DType::UINT32:
    buffer_ = VecBuffer<uint32_t>(n);
    break;
  case DType::INT64:
    buffer_ = VecBuffer<int64_t>(n);
    break;
  case DType::UINT64:
    buffer_ = VecBuffer<uint64_t>(n);
    break;
  case DType::FLOAT16:
    buffer_ = VecBuffer<Eigen::half>(n);
    break;
  case DType::FLOAT32:
    buffer_ = VecBuffer<float>(n);
    break;
  case DType::FLOAT64:
    buffer_ = VecBuffer<double>(n);
    break;
  }
}

// - ctors ---------------------------------------------------------------------
Buffer::Buffer(std::size_t n, std::string_view dt) {
  initialize_buffer(n, dtype_from_string(dt));
}

Buffer::Buffer(std::size_t n, std::string_view dt, const py::object &v) {
  initialize_buffer(n, dtype_from_string(dt));

  std::visit(
      [&](auto &buf) {
        using T = std::decay_t<decltype(buf[0])>;
        T value;
        if constexpr (std::is_same_v<T, bool>)
          value = v.cast<bool>();
        else if constexpr (std::is_integral_v<T>) {
          value = py::isinstance<py::float_>(v)
                      ? static_cast<T>(v.cast<double>())
                      : static_cast<T>(v.cast<int64_t>());
        } else { // floating point & Eigen::half
          value = static_cast<T>(v.cast<double>());
        }
        std::fill(buf.data(), buf.data() + buf.size(), value);
      },
      buffer_);
}

Buffer::Buffer(const py::sequence &seq, std::string_view dt) {
  const std::size_t n = seq.size();
  initialize_buffer(n, dtype_from_string(dt));

  std::visit(
      [&](auto &buf) {
        using T = std::decay_t<decltype(buf[0])>;
        for (std::size_t i = 0; i < n; ++i) {
          py::handle item = seq[i];
          if constexpr (std::is_same_v<T, bool>)
            buf[i] = item.cast<bool>();
          else if constexpr (std::is_integral_v<T>)
            buf[i] = static_cast<T>(py::isinstance<py::float_>(item)
                                        ? item.cast<double>()
                                        : item.cast<int64_t>());
          else
            buf[i] = static_cast<T>(item.cast<double>());
        }
      },
      buffer_);
}

// - trivial API ---------------------------------------------------------------
std::size_t Buffer::size() const {
  return std::visit([](auto &b) { return b.size(); }, buffer_);
}
py::object Buffer::get_item(std::size_t i) const {
  return std::visit([&](auto &b) { return py::cast(b[i]); }, buffer_);
}
void Buffer::set_item(std::size_t i, double v) {
  std::visit(
      [&](auto &b) {
        using T = std::decay_t<decltype(b[0])>;
        b[i] = static_cast<T>(v);
      },
      buffer_);
}
std::string Buffer::get_dtype() const { return dtype_to_string(dtype_); }

// - element-wise core ---------------------------------------------------------
// Buffer Buffer::elementwise_op(const Buffer &rhs, EwOp op,
//                               std::string_view out_dt) const {
//   if (size() != rhs.size())
//     throw std::runtime_error("shape mismatch");
//   const DType out_dtype = out_dt.empty() ? dtype_ :
//   dtype_from_string(out_dt);

//   Buffer out(size(), dtype_to_string(out_dtype));

//   auto work = [&](auto OP_CONST) {
//     std::visit(
//         [&](const auto &l_buf) {
//           std::visit(
//               [&](const auto &r_buf) {
//                 std::visit(
//                     [&](auto &o_buf) {
//                       for (std::size_t i = 0; i < size(); ++i) {
//                         auto tmp =
//                             binary_op<OP_CONST.value>(l_buf[i], r_buf[i]);
//                         o_buf[i] =
//                             static_cast<std::decay_t<decltype(o_buf[0])>>(tmp);
//                       }
//                     },
//                     out.get_buffer());
//               },
//               rhs.get_buffer());
//         },
//         this->get_buffer());
//   };

//   switch (op) {
//   case EwOp::ADD:
//     work(std::integral_constant<EwOp, EwOp::ADD>{});
//     break;
//   case EwOp::SUB:
//     work(std::integral_constant<EwOp, EwOp::SUB>{});
//     break;
//   case EwOp::MUL:
//     work(std::integral_constant<EwOp, EwOp::MUL>{});
//     break;
//   case EwOp::DIV:
//     work(std::integral_constant<EwOp, EwOp::DIV>{});
//     break;
//   }
//   return out;
// }

//---- wrappers --------------------------------------------------------
// Stub free-function 'add' to satisfy PYBIND11_MODULE linkage in this module.
Buffer add(const Buffer &lhs, const Buffer &rhs,
           std::string_view /*out_dtype*/) {
  // Return a new Buffer matching lhs (contents unspecified)
  return Buffer(lhs.size(), lhs.get_dtype());
}
// Buffer Buffer::add(const Buffer &o, std::string_view dt) const {
//   return elementwise_op(o, EwOp::ADD, dt);
// }
// Buffer Buffer::sub(const Buffer &o, std::string_view dt) const {
//   return elementwise_op(o, EwOp::SUB, dt);
// }
// Buffer Buffer::mul(const Buffer &o, std::string_view dt) const {
//   return elementwise_op(o, EwOp::MUL, dt);
// }
// Buffer Buffer::div(const Buffer &o, std::string_view dt) const {
//   return elementwise_op(o, EwOp::DIV, dt);
// }

// Buffer add(const Buffer &a, const Buffer &b, std::string_view dt) {
//   return a.add(b, dt);
// }
// Buffer sub(const Buffer &a, const Buffer &b, std::string_view dt) {
//   return a.sub(b, dt);
// }
// Buffer mul(const Buffer &a, const Buffer &b, std::string_view dt) {
//   return a.mul(b, dt);
// }
// Buffer div(const Buffer &a, const Buffer &b, std::string_view dt) {
//   return a.div(b, dt);
// }

// ---------------- explicit instantiations for VecBuffer ----------------
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

// -------------------------- PYBIND11 module ---------------------------------
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
      .def("get_dtype", &Buffer::get_dtype)
      .def(
          "__add__",
          [](const Buffer &a, const Buffer &b, const std::string &dt) {
            return add(a, b, dt);
          },
          py::arg("other"), py::arg("out_dtype") = "");

  m.def("add", &add, py::arg("lhs"), py::arg("rhs"), py::arg("out_dtype") = "");
}
