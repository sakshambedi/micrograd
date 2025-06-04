#include "cpu_kernel.h"
namespace py = pybind11;

// pybind11 type_caster for Eigen::half
namespace pybind11 {
namespace detail {

template <> struct type_caster<Eigen::half> {
public:
  PYBIND11_TYPE_CASTER(Eigen::half, _("numpy.float16"));

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

} // namespace detail
} // namespace pybind11

// --- VecBuffer Implementation ---
template <typename T>
VecBuffer<T>::VecBuffer(std::size_t n) : data_(n) {}

template <typename T>
VecBuffer<T>::VecBuffer(const T* src, std::size_t n)
    : data_(Eigen::Map<const typename VecBuffer<T>::Array>(src, n)) {}

template <typename T>
T& VecBuffer<T>::operator[](std::size_t i) { return data_(i); }

template <typename T>
const T& VecBuffer<T>::operator[](std::size_t i) const { return data_(i); }

template <typename T>
std::size_t VecBuffer<T>::size() const { return data_.size(); }

template <typename T>
T* VecBuffer<T>::data() { return data_.data(); }

template <typename T>
const T* VecBuffer<T>::data() const { return data_.data(); }

template <typename T>
VecBuffer<T>& VecBuffer<T>::operator+=(const VecBuffer& rhs) {
    data_ += rhs.data_;
    return *this;
}

template <typename T>
VecBuffer<T>& VecBuffer<T>::operator-=(const VecBuffer& rhs) {
    data_ -= rhs.data_;
    return *this;
}

template <typename T>
VecBuffer<T> VecBuffer<T>::cwiseMul(const VecBuffer& rhs) const {
    return VecBuffer(data_.cwiseProduct(rhs.data_));
}

template <typename T>
T VecBuffer<T>::dot(const VecBuffer& rhs) const {
    return data_.matrix().dot(rhs.data_.matrix());
}

template <typename T>
Eigen::Ref<typename VecBuffer<T>::Array> VecBuffer<T>::ref() { return data_; }

template <typename T>
Eigen::Ref<const typename VecBuffer<T>::Array> VecBuffer<T>::ref() const { return data_; }

template <typename T>
VecBuffer<T>::VecBuffer(const typename VecBuffer<T>::Array& a) : data_(a) {}

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

// --- Buffer Implementation ---
Buffer::Buffer(const std::string &dtype, std::size_t size) {
  if (dtype == "bool" || dtype == "?") {
    buffer_ = VecBuffer<bool>(size);
  } else if (dtype == "int8" || dtype == "b") {
    buffer_ = VecBuffer<std::int8_t>(size);
  } else if (dtype == "uint8" || dtype == "B") {
    buffer_ = VecBuffer<std::uint8_t>(size);
  } else if (dtype == "int16" || dtype == "h") {
    buffer_ = VecBuffer<std::int16_t>(size);
  } else if (dtype == "uint16" || dtype == "H") {
    buffer_ = VecBuffer<std::uint16_t>(size);
  } else if (dtype == "int32" || dtype == "i") {
    buffer_ = VecBuffer<std::int32_t>(size);
  } else if (dtype == "uint32" || dtype == "I") {
    buffer_ = VecBuffer<std::uint32_t>(size);
  } else if (dtype == "int64" || dtype == "q") {
    buffer_ = VecBuffer<std::int64_t>(size);
  } else if (dtype == "uint64" || dtype == "Q") {
    buffer_ = VecBuffer<std::uint64_t>(size);
  } else if (dtype == "float16" || dtype == "e") {
    buffer_ = VecBuffer<Eigen::half>(size);
  } else if (dtype == "float32" || dtype == "f") {
    buffer_ = VecBuffer<float>(size);
  } else if (dtype == "float64" || dtype == "d") {
    buffer_ = VecBuffer<double>(size);
  } else {
    throw std::runtime_error("Unsupported dtype: " + dtype);
  }
}

std::size_t Buffer::size() const {
  return std::visit([](const auto &buf) { return buf.size(); }, buffer_);
}

py::object Buffer::get_item(std::size_t i) const {
  return std::visit(
      [i](auto const &buf) {
        return py::cast(buf[i]);
      },
      buffer_);
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
PYBIND11_MODULE(cpu_kernel, m) {
  // Expose the variant-based Buffer class
  py::class_<Buffer>(m, "Buffer")
      .def(py::init<const std::string &, std::size_t>())
      .def("__getitem__", &Buffer::get_item)
      .def("__setitem__", &Buffer::set_item)
      .def("size", &Buffer::size)
      .def("get_dtype", &Buffer::get_dtype);

  // Keep the old individual classes for backwards compatibility
  // py::class_<VecBuffer<float>>(m, "VecBufferFloat")
  //     .def(py::init<std::size_t>())
  //     .def("__getitem__",
  //          [](const VecBuffer<float> &v, std::size_t i) { return v[i]; })
  //     .def("__setitem__",
  //          [](VecBuffer<float> &v, std::size_t i, float val) { v[i] = val; })
  //     .def("size", &VecBuffer<float>::size)
  //     .def("dot", &VecBuffer<float>::dot)
  //     .def("cwiseMul", &VecBuffer<float>::cwiseMul)
  //     .def("__iadd__", &VecBuffer<float>::operator+=, py::is_operator())
  //     .def("__isub__", &VecBuffer<float>::operator-=, py::is_operator());
}
