// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "cpu_kernel.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "Eigen/src/Core/Map.h"
#include "Eigen/src/Core/Ref.h"
#include "Eigen/src/Core/arch/Default/Half.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/detail/descr.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;

// pybind11 type_caster for Eigen::half
namespace pybind11::detail {

template <> struct type_caster<Eigen::half> {
protected:
  Eigen ::half value;

public:
  // static constexpr auto name = _("numpy.float16");
  template <
      typename T_,
      ::pybind11 ::detail ::enable_if_t<
          std ::is_same_v<Eigen ::half, ::pybind11 ::detail ::remove_cv_t<T_>>,
          int> = 0>
  static ::pybind11 ::handle cast(T_ *src,
                                  ::pybind11 ::return_value_policy policy,
                                  ::pybind11 ::handle parent) {
    if (!src)
      return ::pybind11 ::none().release();
    if (policy == ::pybind11 ::return_value_policy ::take_ownership) {
      auto h = cast(std ::move(*src), policy, parent);
      delete src;
      return h;
    }
    return cast(*src, policy, parent);
  }
  operator Eigen ::half *() { return &value; }
  operator Eigen ::half &() { return value; }
  operator Eigen ::half &&() && { return std ::move(value); }
  template <typename T_>
  using cast_op_type = ::pybind11 ::detail ::movable_cast_op_type<T_>;

  // Python → C++
  bool load(handle src, bool) {
    value = Eigen::half(src.cast<float>());
    return true;
  }

  // C++ → Python
  static handle cast(Eigen::half h, return_value_policy, handle) {
    return py::float_(static_cast<float>(h)).release();
  }
};

} // namespace pybind11::detail

void init_kernel_table() {

  reg<EwOp::ADD, bool, bool, bool>(DTypeEnum::BOOL, DTypeEnum::BOOL,
                                   DTypeEnum::BOOL);
  reg<EwOp::ADD, bool, std::int8_t, std::int8_t>(
      DTypeEnum::BOOL, DTypeEnum::INT8, DTypeEnum::INT8);
  reg<EwOp::ADD, std::int8_t, bool, std::int8_t>(
      DTypeEnum::INT8, DTypeEnum::BOOL, DTypeEnum::INT8);
  reg<EwOp::ADD, bool, std::int32_t, std::int32_t>(
      DTypeEnum::BOOL, DTypeEnum::INT32, DTypeEnum::INT32);
  reg<EwOp::ADD, std::int32_t, bool, std::int32_t>(
      DTypeEnum::INT32, DTypeEnum::BOOL, DTypeEnum::INT32);

  // Register ADD kernels for int8 types
  reg<EwOp::ADD, std::int8_t, std::int8_t, std::int8_t>(
      DTypeEnum::INT8, DTypeEnum::INT8, DTypeEnum::INT8);
  reg<EwOp::ADD, std::int8_t, std::uint8_t, std::int16_t>(
      DTypeEnum::INT8, DTypeEnum::UINT8, DTypeEnum::INT16);
  reg<EwOp::ADD, std::uint8_t, std::int8_t, std::int16_t>(
      DTypeEnum::UINT8, DTypeEnum::INT8, DTypeEnum::INT16);
  reg<EwOp::ADD, std::uint8_t, std::uint8_t, std::uint8_t>(
      DTypeEnum::UINT8, DTypeEnum::UINT8, DTypeEnum::UINT8);

  // Register ADD kernels for int16 types
  reg<EwOp::ADD, std::int16_t, std::int16_t, std::int16_t>(
      DTypeEnum::INT16, DTypeEnum::INT16, DTypeEnum::INT16);

  // Register ADD kernels for int32 types
  reg<EwOp::ADD, std::int32_t, std::int32_t, std::int32_t>(
      DTypeEnum::INT32, DTypeEnum::INT32, DTypeEnum::INT32);

  // Register float kernels which are commonly used
  reg<EwOp::ADD, float, float, float>(DTypeEnum::FLOAT32, DTypeEnum::FLOAT32,
                                      DTypeEnum::FLOAT32);
  reg<EwOp::ADD, double, double, double>(DTypeEnum::FLOAT64, DTypeEnum::FLOAT64,
                                         DTypeEnum::FLOAT64);
  reg<EwOp::ADD, float, double, double>(DTypeEnum::FLOAT32, DTypeEnum::FLOAT64,
                                        DTypeEnum::FLOAT64);
  reg<EwOp::ADD, double, float, double>(DTypeEnum::FLOAT64, DTypeEnum::FLOAT32,
                                        DTypeEnum::FLOAT64);

  // Register half-precision floating point operations
  reg<EwOp::ADD, Eigen::half, Eigen::half, Eigen::half>(
      DTypeEnum::FLOAT16, DTypeEnum::FLOAT16, DTypeEnum::FLOAT16);
  reg<EwOp::ADD, Eigen::half, float, float>(
      DTypeEnum::FLOAT16, DTypeEnum::FLOAT32, DTypeEnum::FLOAT32);
  reg<EwOp::ADD, float, Eigen::half, float>(
      DTypeEnum::FLOAT32, DTypeEnum::FLOAT16, DTypeEnum::FLOAT32);
  reg<EwOp::ADD, Eigen::half, double, double>(
      DTypeEnum::FLOAT16, DTypeEnum::FLOAT64, DTypeEnum::FLOAT64);
  reg<EwOp::ADD, double, Eigen::half, double>(
      DTypeEnum::FLOAT64, DTypeEnum::FLOAT16, DTypeEnum::FLOAT64);

  // Integer to float conversions
  reg<EwOp::ADD, std::int32_t, float, float>(
      DTypeEnum::INT32, DTypeEnum::FLOAT32, DTypeEnum::FLOAT32);
  reg<EwOp::ADD, float, std::int32_t, float>(
      DTypeEnum::FLOAT32, DTypeEnum::INT32, DTypeEnum::FLOAT32);

  // Bool type operations
  reg<EwOp::ADD, bool, bool, bool>(DTypeEnum::BOOL, DTypeEnum::BOOL,
                                   DTypeEnum::BOOL);
  reg<EwOp::ADD, bool, std::int32_t, std::int32_t>(
      DTypeEnum::BOOL, DTypeEnum::INT32, DTypeEnum::INT32);
  reg<EwOp::ADD, std::int32_t, bool, std::int32_t>(
      DTypeEnum::INT32, DTypeEnum::BOOL, DTypeEnum::INT32);
  reg<EwOp::ADD, bool, float, float>(DTypeEnum::BOOL, DTypeEnum::FLOAT32,
                                     DTypeEnum::FLOAT32);
  reg<EwOp::ADD, float, bool, float>(DTypeEnum::FLOAT32, DTypeEnum::BOOL,
                                     DTypeEnum::FLOAT32);

  // Additional integer conversions
  reg<EwOp::ADD, std::int16_t, std::int8_t, std::int16_t>(
      DTypeEnum::INT16, DTypeEnum::INT8, DTypeEnum::INT16);
  reg<EwOp::ADD, std::int8_t, std::int16_t, std::int16_t>(
      DTypeEnum::INT8, DTypeEnum::INT16, DTypeEnum::INT16);
  reg<EwOp::ADD, std::int32_t, std::int16_t, std::int32_t>(
      DTypeEnum::INT32, DTypeEnum::INT16, DTypeEnum::INT32);
  reg<EwOp::ADD, std::int16_t, std::int32_t, std::int32_t>(
      DTypeEnum::INT16, DTypeEnum::INT32, DTypeEnum::INT32);
  reg<EwOp::ADD, std::int64_t, std::int64_t, std::int64_t>(
      DTypeEnum::INT64, DTypeEnum::INT64, DTypeEnum::INT64);

  // No need to set initialized flag as we always want to initialize
}

// --- VecBuffer Implementation ---
template <typename T> VecBuffer<T>::VecBuffer(std::size_t n) : data_(n) {}

template <typename T>
VecBuffer<T>::VecBuffer(const T *src, std::size_t n)
    : data_(Eigen::Map<const typename VecBuffer<T>::Array>(src, n)) {}

template <typename T> T &VecBuffer<T>::operator[](std::size_t i) {
  return data_(i);
}

template <typename T> const T &VecBuffer<T>::operator[](std::size_t i) const {
  return data_(i);
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
  return VecBuffer(data_.cwiseProduct(rhs.data_));
}

template <typename T> T VecBuffer<T>::dot(const VecBuffer &rhs) const {
  return data_.matrix().dot(rhs.data_.matrix());
}

template <typename T>
Eigen::Ref<typename VecBuffer<T>::Array> VecBuffer<T>::ref() {
  return data_;
}

template <typename T>
Eigen::Ref<const typename VecBuffer<T>::Array> VecBuffer<T>::ref() const {
  return data_;
}

template <typename T>
VecBuffer<T>::VecBuffer(const typename VecBuffer<T>::Array &a) : data_(a) {}

// --- DTypeEnum Implementation ---
DTypeEnum get_dtype_enum(std::string_view dtype) {
  static const std::unordered_map<std::string_view, DTypeEnum> dtypeTable = {
      {"bool", DTypeEnum::BOOL},       {"?", DTypeEnum::BOOL},
      {"int8", DTypeEnum::INT8},       {"b", DTypeEnum::INT8},
      {"uint8", DTypeEnum::UINT8},     {"B", DTypeEnum::UINT8},
      {"int16", DTypeEnum::INT16},     {"h", DTypeEnum::INT16},
      {"uint16", DTypeEnum::UINT16},   {"H", DTypeEnum::UINT16},
      {"int32", DTypeEnum::INT32},     {"i", DTypeEnum::INT32},
      {"uint32", DTypeEnum::UINT32},   {"I", DTypeEnum::UINT32},
      {"int64", DTypeEnum::INT64},     {"q", DTypeEnum::INT64},
      {"uint64", DTypeEnum::UINT64},   {"Q", DTypeEnum::UINT64},
      {"float16", DTypeEnum::FLOAT16}, {"e", DTypeEnum::FLOAT16},
      {"float32", DTypeEnum::FLOAT32}, {"f", DTypeEnum::FLOAT32},
      {"float64", DTypeEnum::FLOAT64}, {"d", DTypeEnum::FLOAT64},
  };

  auto it = dtypeTable.find(dtype);
  return (it == dtypeTable.end()) ? DTypeEnum::UNKNOWN : it->second;
}

template <typename T>
static void fill_from_sequence(VecBuffer<T> &dst, PyObject *seq_fast,
                               std::size_t n) {
  T *out = dst.data();
  for (std::size_t i = 0; i < n; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq_fast, i); // borrowed ref
    py::handle h(item);

    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      // For integer types, handle float conversion explicitly
      if (PyFloat_Check(item)) {
        out[i] = static_cast<T>(PyFloat_AsDouble(item));
      } else {
        out[i] = py::cast<T>(h);
      }
    } else {
      out[i] = py::cast<T>(h);
    }
  }
}

void Buffer::initialize_buffer(std::size_t size, const std::string &dtype) {
  DTypeEnum dtype_enum = get_dtype_enum(dtype);
  auto it = factory_table.find(dtype_enum);
  if (it != factory_table.end()) {
    it->second(*this, size);
  } else {
    throw std::runtime_error("Unsupported dtype: " + dtype);
  }
}

void Buffer::initialize_buffer(std::size_t size, std::string_view dtype) {
  DTypeEnum dtype_enum = get_dtype_enum(dtype);
  auto it = factory_table.find(dtype_enum);
  if (it != factory_table.end()) {
    it->second(*this, size);
  } else {
    throw std::runtime_error("Unsupported dtype: " + std::string(dtype));
  }
}

Buffer::Buffer(std::size_t size, const std::string &dtype) {
  initialize_buffer(size, dtype);
}

Buffer::Buffer(std::size_t size, const std::string &dtype, py::object val) {
  initialize_buffer(size, dtype);
  std::visit(
      [val, size](auto &buf) {
        using T = typename std::decay_t<decltype(buf)>::Array::Scalar;
        T casted_val = py::cast<T>(val);
        for (std::size_t i = 0; i < size; ++i) {
          buf[i] = casted_val;
        }
      },
      buffer_);
}

Buffer::Buffer(py::sequence seq, std::string_view fmt) {
  PyObject *fast = PySequence_Fast(seq.ptr(), "expected list/tuple for Buffer");
  std::size_t n = PySequence_Fast_GET_SIZE(fast);

  initialize_buffer(n, fmt);

  std::visit(
      [&](auto &buf) {
        using BufT = std::decay_t<decltype(buf)>;
        using Scalar = typename BufT::Array::Scalar;
        fill_from_sequence(buf, fast, n);
      },
      buffer_);

  Py_DECREF(fast); // balance PySequence_Fast
}

const std::unordered_map<DTypeEnum, std::function<void(Buffer &, size_t)>>
    factory_table = {
        {DTypeEnum::BOOL,
         [](Buffer &self, size_t n) { self.set_buffer<bool>(n); }},
        {DTypeEnum::INT8,
         [](Buffer &self, size_t n) { self.set_buffer<std::int8_t>(n); }},
        {DTypeEnum::UINT8,
         [](Buffer &self, size_t n) { self.set_buffer<std::uint8_t>(n); }},
        {DTypeEnum::INT16,
         [](Buffer &self, size_t n) { self.set_buffer<std::int16_t>(n); }},
        {DTypeEnum::UINT16,
         [](Buffer &self, size_t n) { self.set_buffer<std::uint16_t>(n); }},
        {DTypeEnum::INT32,
         [](Buffer &self, size_t n) { self.set_buffer<std::int32_t>(n); }},
        {DTypeEnum::UINT32,
         [](Buffer &self, size_t n) { self.set_buffer<std::uint32_t>(n); }},
        {DTypeEnum::INT64,
         [](Buffer &self, size_t n) { self.set_buffer<std::int64_t>(n); }},
        {DTypeEnum::UINT64,
         [](Buffer &self, size_t n) { self.set_buffer<std::uint64_t>(n); }},
        {DTypeEnum::FLOAT16,
         [](Buffer &self, size_t n) { self.set_buffer<Eigen::half>(n); }},
        {DTypeEnum::FLOAT32,
         [](Buffer &self, size_t n) { self.set_buffer<float>(n); }},
        {DTypeEnum::FLOAT64,
         [](Buffer &self, size_t n) { self.set_buffer<double>(n); }},
};

std::size_t Buffer::size() const {
  return std::visit([](const auto &buf) { return buf.size(); }, buffer_);
}

py::object Buffer::get_item(std::size_t i) const {
  return std::visit([i](auto const &buf) { return py::cast(buf[i]); }, buffer_);
}

void Buffer::set_item(std::size_t i, double val) {
  std::visit(
      [i, val](auto &buf) {
        using T = typename std::decay_t<decltype(buf)>::Array::Scalar;
        buf[i] = static_cast<T>(val);
      },
      buffer_);
}

std::string Buffer::get_dtype() const {
  return std::visit(
      [](const auto &buf) -> std::string {
        using T = typename std::decay_t<decltype(buf)>::Array::Scalar;
        if constexpr (std::is_same_v<T, bool>)
          return "bool";
        if constexpr (std::is_same_v<T, std::int8_t>)
          return "int8";
        if constexpr (std::is_same_v<T, std::uint8_t>)
          return "uint8";
        if constexpr (std::is_same_v<T, std::int16_t>)
          return "int16";
        if constexpr (std::is_same_v<T, std::uint16_t>)
          return "uint16";
        if constexpr (std::is_same_v<T, std::int32_t>)
          return "int32";
        if constexpr (std::is_same_v<T, std::uint32_t>)
          return "uint32";
        if constexpr (std::is_same_v<T, std::int64_t>)
          return "int64";
        if constexpr (std::is_same_v<T, std::uint64_t>)
          return "uint64";
        if constexpr (std::is_same_v<T, float>)
          return "float32";
        if constexpr (std::is_same_v<T, double>)
          return "float64";
        if constexpr (std::is_same_v<T, Eigen::half>)
          return "float16";
        return "unknown";
      },
      buffer_);
}

// --- Kernel template --------------------------------------------------------
template <EwOp OP, class A, class B, class R>
static void ew_kernel(const void *a_raw, const void *b_raw, void *r_raw,
                      std::size_t n) {
  using ArrA = Eigen::Array<A, Eigen::Dynamic, 1>;
  using ArrB = Eigen::Array<B, Eigen::Dynamic, 1>;
  using ArrR = Eigen::Array<R, Eigen::Dynamic, 1>;

  Eigen::Map<const ArrA> lhs(reinterpret_cast<const A *>(a_raw), n);
  Eigen::Map<const ArrB> rhs(reinterpret_cast<const B *>(b_raw), n);
  Eigen::Map<ArrR> dst(reinterpret_cast<R *>(r_raw), n);

  auto lhsR = lhs.template cast<R>();
  auto rhsR = rhs.template cast<R>();

  if constexpr (OP == EwOp::ADD)
    dst = lhsR + rhsR;
  else if constexpr (OP == EwOp::MUL)
    dst = lhsR * rhsR;
  else if constexpr (OP == EwOp::DIV)
    dst = lhsR / rhsR;
  else if constexpr (OP == EwOp::SUB)
    dst = lhsR - rhsR;
}

template <EwOp OP, class A, class B, class R>
static void reg(DTypeEnum a, DTypeEnum b, DTypeEnum r) {
  TABLE[static_cast<std::size_t>(OP)][static_cast<std::size_t>(a)]
       [static_cast<std::size_t>(b)][static_cast<std::size_t>(r)] =
           &ew_kernel<OP, A, B, R>;
}

template <EwOp OP>
Buffer Buffer::ewise(const Buffer &other, const std::string &out_dtype) const {
  // Ensure the kernel table is initialized
  static bool initialized = false;
  if (!initialized) {
    init_kernel_table();
    initialized = true;
  }

  if (size() != other.size())
    throw std::runtime_error("size mismatch");

  if (get_dtype_enum(out_dtype) == DTypeEnum::UNKNOWN)
    throw std::runtime_error("invalid output dtype: " + out_dtype);

  const std::size_t n = size();
  const DTypeEnum lhs = get_dtype_enum(get_dtype());
  const DTypeEnum rhs = get_dtype_enum(other.get_dtype());
  const DTypeEnum out = get_dtype_enum(out_dtype);

  KernFn fn =
      TABLE[static_cast<std::size_t>(OP)][static_cast<std::size_t>(lhs)]
           [static_cast<std::size_t>(rhs)][static_cast<std::size_t>(out)];
  if (!fn)
    throw std::runtime_error("unsupported dtype combination: " + get_dtype() +
                             " + " + other.get_dtype() + " -> " + out_dtype);

  Buffer result(n, out_dtype);

  std::visit(
      [&](const auto &l) {
        std::visit(
            [&](const auto &r) {
              std::visit([&](auto &o) { fn(l.data(), r.data(), o.data(), n); },
                         result.buffer_);
            },
            other.buffer_);
      },
      buffer_);

  return result;
}

Buffer add(const Buffer &lhs, const Buffer &b, const std::string &dt) {
  // Ensure the kernel table is initialized
  static bool initialized = false;
  if (!initialized) {
    init_kernel_table();
    initialized = true;
  }

  if (dt.empty()) {
    // Determine appropriate output dtype if not specified
    std::string out_dtype;
    std::string lhs_dtype = lhs.get_dtype();
    std::string rhs_dtype = b.get_dtype();

    // If types are the same, use that type
    if (lhs_dtype == rhs_dtype) {
      out_dtype = lhs_dtype;
      return lhs.ewise<EwOp::ADD>(b, out_dtype);
    }
    // Otherwise use promotion rules
    else {
      // Try different output dtype combinations in order of preference

      // Float types take precedence over integer types
      if (lhs_dtype == "float64" || rhs_dtype == "float64") {
        try {
          return lhs.ewise<EwOp::ADD>(b, "float64");
        } catch (const std::runtime_error &) {
        }
      }

      if (lhs_dtype == "float32" || rhs_dtype == "float32") {
        try {
          return lhs.ewise<EwOp::ADD>(b, "float32");
        } catch (const std::runtime_error &) {
        }
      }

      if (lhs_dtype == "float16" || rhs_dtype == "float16") {
        try {
          return lhs.ewise<EwOp::ADD>(b, "float16");
        } catch (const std::runtime_error &) {
        }
      }

      // Integer type promotion
      if (lhs_dtype == "int64" || rhs_dtype == "int64" ||
          lhs_dtype == "uint64" || rhs_dtype == "uint64") {
        try {
          return lhs.ewise<EwOp::ADD>(b, "int64");
        } catch (const std::runtime_error &) {
        }
      }

      if (lhs_dtype == "int32" || rhs_dtype == "int32" ||
          lhs_dtype == "uint32" || rhs_dtype == "uint32") {
        try {
          return lhs.ewise<EwOp::ADD>(b, "int32");
        } catch (const std::runtime_error &) {
        }
      }

      if (lhs_dtype == "int16" || rhs_dtype == "int16" ||
          lhs_dtype == "uint16" || rhs_dtype == "uint16" ||
          (lhs_dtype == "int8" && rhs_dtype == "uint8") ||
          (lhs_dtype == "uint8" && rhs_dtype == "int8")) {
        try {
          return lhs.ewise<EwOp::ADD>(b, "int16");
        } catch (const std::runtime_error &) {
        }
      }

      // Last resort - try int8 for smaller integer types
      try {
        return lhs.ewise<EwOp::ADD>(b, "int8");
      } catch (const std::runtime_error &) {
        // If all automatic promotions failed, throw a more helpful error
        throw std::runtime_error("No suitable output dtype found for " +
                                 lhs_dtype + " + " + rhs_dtype);
      }
    }
  }

  return lhs.ewise<EwOp::ADD>(b, dt);
}

Buffer Buffer::mul(const Buffer &b, const std::string &dt) const {
  return this->ewise<EwOp::MUL>(b, dt);
}

Buffer Buffer::div(const Buffer &b, const std::string &dt) const {
  return this->ewise<EwOp::DIV>(b, dt);
}

// --- Pybind11 Module ---
template class VecBuffer<bool>;
template class VecBuffer<std::int8_t>;
template class VecBuffer<std::uint8_t>;
template class VecBuffer<std::int16_t>;
template class VecBuffer<std::uint16_t>;
template class VecBuffer<std::int32_t>;
template class VecBuffer<std::uint32_t>;
template class VecBuffer<std::int64_t>;
template class VecBuffer<std::uint64_t>;
template class VecBuffer<Eigen::half>;
template class VecBuffer<float>;
template class VecBuffer<double>;

PYBIND11_MODULE(cpu_kernel, m) {
  // Initialize kernel functions table - ensuring all needed kernels are
  // registered
  init_kernel_table();

  // Expose the variant-based Buffer class
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<py::sequence, std::string_view>(), py::arg("sequence"),
           py::arg("fmt"))
      .def(py::init<std::size_t, const std::string &>(), py::arg("size"),
           py::arg("dtype"))
      .def(py::init<std::size_t, const std::string &, py::object>(),
           py::arg("size"), py::arg("dtype"), py::arg("value"))
      .def("__getitem__", &Buffer::get_item)
      .def("__setitem__", &Buffer::set_item)
      .def("size", &Buffer::size)
      .def("get_dtype", &Buffer::get_dtype)
      .def(
          "__add__",
          [](const Buffer &self, const Buffer &other,
             const std::string &out_dtype) {
            return add(self, other, out_dtype);
          },
          py::arg("other"), py::arg("out_dtype") = "");

  m.def("add", &add, py::arg("lhs"), py::arg("rhs"), py::arg("out_dtype") = "",
        "Add two buffers together with SIMD optimization");
}
