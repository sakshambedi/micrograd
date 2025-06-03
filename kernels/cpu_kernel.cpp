// #pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

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

template <typename T> class VecBuffer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // keeps std::new aligned
      using Array = Eigen::Array<T, Eigen::Dynamic, 1>;

  explicit VecBuffer(std::size_t n = 0) : data_(n) {}
  VecBuffer(const T *src, std::size_t n)
      : data_(Eigen::Map<const Array>(src, n)) {}

  T &operator[](std::size_t i) { return data_(i); }
  const T &operator[](std::size_t i) const { return data_(i); }
  std::size_t size() const { return data_.size(); }
  T *data() { return data_.data(); }
  const T *data() const { return data_.data(); }

  VecBuffer &operator+=(const VecBuffer &rhs) {
    data_ += rhs.data_;
    return *this;
  }

  VecBuffer &operator-=(const VecBuffer &rhs) {
    data_ -= rhs.data_;
    return *this;
  }

  VecBuffer cwiseMul(const VecBuffer &rhs) const {
    return VecBuffer(data_.cwiseProduct(rhs.data_));
  }

  T dot(const VecBuffer &rhs) const {
    return data_.matrix().dot(rhs.data_.matrix());
  }

  Eigen::Ref<Array> ref() { return data_; }
  Eigen::Ref<const Array> ref() const { return data_; }

private:
  Array data_;
  explicit VecBuffer(const Array &a) : data_(a) {}
};

using BufferVariant =
    std::variant<VecBuffer<bool>, VecBuffer<std::int8_t>,
                 VecBuffer<std::uint8_t>, VecBuffer<int16_t>,
                 VecBuffer<uint16_t>, VecBuffer<int32_t>, VecBuffer<uint32_t>,
                 VecBuffer<int64_t>, VecBuffer<uint64_t>, VecBuffer<float>,
                 VecBuffer<double>, VecBuffer<Eigen::half>>;

class Buffer {
public:
  explicit Buffer(const std::string &dtype, std::size_t size) {
    // if (dtype == "void") {
    //   buffer_ = VecBuffer<void>(0); // .. aah doesnt work rn
    // }
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
      // Eigen::half is a 2‐byte floating‐point
      buffer_ = VecBuffer<Eigen::half>(size);
    } else if (dtype == "float32" || dtype == "f") {
      buffer_ = VecBuffer<float>(size);
    } else if (dtype == "float64" || dtype == "d") {
      buffer_ = VecBuffer<double>(size);
    } else {
      throw std::runtime_error("Unsupported dtype: " + dtype);
    }
  }

  std::size_t size() const {
    return std::visit([](const auto &buf) { return buf.size(); }, buffer_);
  }

  // Generic getter using std::visit + py::cast()
  py::object get_item(std::size_t i) const {
    return std::visit(
        [i](auto const &buf) {
          // for T=float,double,int… → returns PyFloat/PyInt
          // for T=Eigen::half    → invokes your type_caster → np.float16
          return py::cast(buf[i]);
        },
        buffer_);
  }

  // Generic setter using std::visit
  void set_item(std::size_t i, double val) {
    std::visit(
        [i, val](auto &buf) {
          using T = typename std::decay_t<decltype(buf)>::Array::Scalar;
          buf[i] = static_cast<T>(val);
        },
        buffer_);
  }

  std::string get_dtype() const {
    return std::visit(
        [](const auto &buf) -> std::string {
          using T = typename std::decay_t<decltype(buf)>::Array::Scalar;
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

private:
  BufferVariant buffer_;
};

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
