#include "cpu_kernel.h"
#include "Eigen/src/Core/Map.h"
#include "Eigen/src/Core/Ref.h"
#include "Eigen/src/Core/arch/Default/Half.h"
#include "abstract.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/detail/descr.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pytypedefs.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace py = pybind11;

// pybind11 type_caster for Eigen::half
namespace pybind11::detail {

template <> struct type_caster<Eigen::half> {
public:
protected:
  Eigen ::half value;

public:
  static constexpr auto name = _("numpy.float16");
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
    static py::object np = py::module_::import("numpy");
    static py::object f16 = np.attr("float16");
    return f16(static_cast<float>(h)).release();
  }
};

} // namespace pybind11::detail

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
    // pybind11::cast is safe & fast inside C++ (holds GIL)
    // out[i] = py::cast<T>(item);
    out[i] = py::cast<T>(py::handle(item));
  }
}

Buffer::Buffer(const std::string &dtype, std::size_t size) {
  DTypeEnum dtype_enum = get_dtype_enum(dtype);
  auto it = factory_table.find(dtype_enum);
  if (it != factory_table.end()) {
    it->second(*this, size);
  } else {
    throw std::runtime_error("Unsupported dtype: " + dtype);
  }
}

Buffer::Buffer(py::sequence seq, std::string_view fmt) {
  DTypeEnum dt = get_dtype_enum(fmt);
  PyObject *fast = PySequence_Fast(seq.ptr(), "expected list/tuple for Buffer");
  std::size_t n = PySequence_Fast_GET_SIZE(fast);

  // Use existing factory_table to allocate the right VecBuffer<T>
  auto it = factory_table.find(dt);
  if (it == factory_table.end())
    throw std::runtime_error("Unsupported dtype for sequence ctor");

  // Allocate empty buffer of size n
  it->second(*this, n);

  // Dispatch fill loop via std::visit
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
  // Expose the variant-based Buffer class
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<py::sequence, std::string_view>(), py::arg("sequence"),
           py::arg("fmt"))
      .def(py::init<const std::string &, std::size_t>(), py::arg("dtype"),
           py::arg("size"))
      .def("__getitem__", &Buffer::get_item)
      .def("__setitem__", &Buffer::set_item)
      .def("size", &Buffer::size)
      .def("get_dtype", &Buffer::get_dtype);

  //   Keep the old individual classes for backwards compatibility
  py::class_<VecBuffer<float>>(m, "VecBufferFloat")
      .def(py::init<std::size_t>())
      .def("__getitem__",
           [](const VecBuffer<float> &v, std::size_t i) { return v[i]; })
      .def("__setitem__",
           [](VecBuffer<float> &v, std::size_t i, float val) { v[i] = val; })
      .def("size", &VecBuffer<float>::size)
      .def("dot", &VecBuffer<float>::dot)
      .def("cwiseMul", &VecBuffer<float>::cwiseMul)
      .def("__iadd__", &VecBuffer<float>::operator+=, py::is_operator())
      .def("__isub__", &VecBuffer<float>::operator-=, py::is_operator());
}
