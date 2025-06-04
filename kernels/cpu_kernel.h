#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cstddef>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace py = pybind11;

// --- VecBuffer ---
template <typename T>
class VecBuffer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Array = Eigen::Array<T, Eigen::Dynamic, 1>;

    explicit VecBuffer(std::size_t n = 0);
    VecBuffer(const T* src, std::size_t n);

    T& operator[](std::size_t i);
    const T& operator[](std::size_t i) const;
    std::size_t size() const;
    T* data();
    const T* data() const;

    VecBuffer& operator+=(const VecBuffer& rhs);
    VecBuffer& operator-=(const VecBuffer& rhs);
    VecBuffer cwiseMul(const VecBuffer& rhs) const;
    T dot(const VecBuffer& rhs) const;

    Eigen::Ref<Array> ref();
    Eigen::Ref<const Array> ref() const;

private:
    Array data_;
    explicit VecBuffer(const Array& a);
};

// --- DTypeEnum ---
enum class DTypeEnum {
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64,
    FLOAT16, FLOAT32, FLOAT64, UNKNOWN
};

DTypeEnum get_dtype_enum(std::string_view dtype);

// --- Buffer ---
class Buffer {
public:
    explicit Buffer(const std::string& dtype, std::size_t size);

    std::size_t size() const;
    py::object get_item(std::size_t i) const;
    void set_item(std::size_t i, double val);
    std::string get_dtype() const;

private:
    using BufferVariant =
        std::variant<
            VecBuffer<bool>, VecBuffer<std::int8_t>, VecBuffer<std::uint8_t>,
            VecBuffer<int16_t>, VecBuffer<uint16_t>, VecBuffer<int32_t>, VecBuffer<uint32_t>,
            VecBuffer<int64_t>, VecBuffer<uint64_t>, VecBuffer<float>, VecBuffer<double>,
            VecBuffer<Eigen::half>
        >;
    BufferVariant buffer_;
};