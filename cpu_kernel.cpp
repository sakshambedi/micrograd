#include "vecbuffer.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <variant>
#include <unordered_map>
#include <string>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// dtype enum and helpers
enum class DType : uint8_t {
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    NUM_TYPES
};

static const std::unordered_map<std::string, DType> str_to_dtype = {
    {"int32", DType::INT32},
    {"int64", DType::INT64},
    {"float32", DType::FLOAT32},
    {"float64", DType::FLOAT64},
};

static inline DType dtype_from_string(const std::string& s) {
    auto it = str_to_dtype.find(s);
    if (it == str_to_dtype.end()) throw std::runtime_error("unknown dtype");
    return it->second;
}

static inline std::string dtype_to_string(DType t) {
    for (auto& kv : str_to_dtype) if (kv.second == t) return kv.first;
    return "";
}

// -----------------------------------------------------------------------------
// promotion table following simple rank order
static constexpr std::array<int, static_cast<int>(DType::NUM_TYPES)> rank = {
    0, // int32
    1, // int64
    2, // float32
    3  // float64
};
static inline DType promote(DType a, DType b) {
    return (rank[(int)a] > rank[(int)b]) ? a : b;
}

// -----------------------------------------------------------------------------
using BufferVariant = std::variant<
    VecBuffer<int32_t>,
    VecBuffer<int64_t>,
    VecBuffer<float>,
    VecBuffer<double>>;

struct Buffer {
    Buffer(std::size_t n, const std::string& dtype) {
        init(n, dtype_from_string(dtype));
    }

    Buffer(const py::buffer& view, const std::string& dtype) {
        auto info = view.request();
        init(info.size, dtype_from_string(dtype));
        std::visit([&](auto& buf){
            using T = std::decay_t<decltype(buf[0])>;
            const T* src = static_cast<const T*>(info.ptr);
            std::copy(src, src + info.size, buf.data());
        }, data_);
    }

    std::size_t size() const {
        return std::visit([](auto& b){ return b.size(); }, data_);
    }

    const BufferVariant& raw() const { return data_; }
    BufferVariant& raw() { return data_; }

    py::dict array_interface() const {
        py::dict d;
        std::visit([&](auto& b){
            using T = std::decay_t<decltype(b[0])>;
            d["shape"] = py::make_tuple(b.size());
            if constexpr(std::is_same_v<T,int32_t>) d["typestr"] = "<i4";
            else if constexpr(std::is_same_v<T,int64_t>) d["typestr"] = "<i8";
            else if constexpr(std::is_same_v<T,float>) d["typestr"] = "<f4";
            else if constexpr(std::is_same_v<T,double>) d["typestr"] = "<f8";
            d["data"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(b.data()), false);
        }, data_);
        d["version"] = 3;
        return d;
    }

    std::string dtype() const { return dtype_to_string(dtype_); }

    // elementwise add
    Buffer add(const Buffer& rhs) const {
        if (size() != rhs.size()) throw std::runtime_error("size mismatch");
        DType out_t = promote(dtype_, rhs.dtype_);
        Buffer out(size(), dtype_to_string(out_t));
        auto dispatch = [&](auto& lbuf){
            using L = std::decay_t<decltype(lbuf[0])>;
            std::visit([&](auto& rbuf){
                using R = std::decay_t<decltype(rbuf[0])>;
                std::visit([&](auto& obuf){
                    using O = std::decay_t<decltype(obuf[0])>;
                    binary_kernel<O, AddOp>(
                        (const O*)lbuf.data(),
                        (const O*)rbuf.data(),
                        obuf.data(),
                        size());
                }, out.data_);
            }, rhs.data_);
        };
        std::visit(dispatch, data_);
        return out;
    }

private:
    void init(std::size_t n, DType t) {
        dtype_ = t;
        switch(t) {
            case DType::INT32: data_ = VecBuffer<int32_t>(n); break;
            case DType::INT64: data_ = VecBuffer<int64_t>(n); break;
            case DType::FLOAT32: data_ = VecBuffer<float>(n); break;
            case DType::FLOAT64: data_ = VecBuffer<double>(n); break;
            default: throw std::runtime_error("bad dtype");
        }
    }

    BufferVariant data_;
    DType dtype_;
};

// -----------------------------------------------------------------------------
PYBIND11_MODULE(cpu_kernel, m) {
    py::class_<Buffer>(m, "Buffer")
        .def(py::init<std::size_t, const std::string&>())
        .def(py::init<py::buffer, const std::string&>())
        .def("size", &Buffer::size)
        .def("get_dtype", &Buffer::dtype)
        .def_property_readonly("__array_interface__", &Buffer::array_interface)
        .def("add", &Buffer::add);
}
