// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "operations.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace simd_ops {

struct AddOp {
  template <typename T> static constexpr T apply_scalar(T a, T b) noexcept {
    return a + b;
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
    return a + b;
  }

  static constexpr const char *name() noexcept { return "add"; }
};

struct SubOp {
  template <typename T> static constexpr T apply_scalar(T a, T b) noexcept {
    return a - b;
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
    return a - b;
  }

  static constexpr const char *name() noexcept { return "sub"; }
};

struct MulOp {
  template <typename T> static constexpr T apply_scalar(T a, T b) noexcept {
    return a * b;
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
    return a * b;
  }

  static constexpr const char *name() noexcept { return "mul"; }
};

struct DivOp {
  template <typename T> static constexpr T apply_scalar(T a, T b) noexcept {
    // Handle special case: Inf/0 = NaN in IEEE 754
    if (std::isinf(a) && b == T(0)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    return a / b;
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
    using T = typename Batch::value_type;

    // Create masks for special case: Inf/0 = NaN
    // if constexpr (std::is_floating_point_v<T>) {
    //   // Check for infinity in a and zero in b
    //   auto inf_mask = xsimd::isinf(a);
    //   auto zero_mask = (b == Batch(T(0)));
    //   auto special_mask = inf_mask && zero_mask;

    //   // If we have any special cases
    //   if (xsimd::any(special_mask)) {
    //     // Regular division
    //     auto result = a / b;
    //     // Replace Inf/0 cases with NaN
    //     auto nan_value = Batch(std::numeric_limits<T>::quiet_NaN());
    //     return xsimd::select(special_mask, nan_value, result);
    //   }
    // }

    // Normal case
    return a / b;
  }

  static constexpr const char *name() noexcept { return "div"; }
};

struct PowOp {
  template <typename T> static constexpr T apply_scalar(T a, T b) noexcept {
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, half>) {
      if constexpr (std::is_same_v<T, half>) {
        // Convert half to float for computation, then back to half
        auto a_f = static_cast<float>(a);
        auto b_f = static_cast<float>(b);
        return static_cast<T>(std::pow(a_f, b_f));
      } else {
        return std::pow(a, b);
      }
    } else {
      // For integer types, use simple iterative approach for positive integer
      // exponents
      if (b < T(0)) {
        return T(0); // Integer division would be 0 for most cases
      }
      if (b == T(0)) {
        return T(1);
      }
      T result = T(1);
      T base = a;
      T exp = b;
      while (exp > T(0)) {
        if (exp % T(2) == T(1)) {
          result *= base;
        }
        base *= base;
        exp /= T(2);
      }
      return result;
    }
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
    using T = typename Batch::value_type;
    if constexpr (std::is_floating_point_v<T> && !std::is_same_v<T, half>) {
      return xsimd::pow(a, b);
    } else {
      alignas(Batch::arch_type::alignment()) T a_arr[Batch::size];
      alignas(Batch::arch_type::alignment()) T b_arr[Batch::size];
      alignas(Batch::arch_type::alignment()) T result_arr[Batch::size];

      a.store_aligned(a_arr);
      b.store_aligned(b_arr);

      for (std::size_t i = 0; i < Batch::size; ++i) {
        result_arr[i] = apply_scalar(a_arr[i], b_arr[i]);
      }

      return Batch::load_aligned(result_arr);
    }
  }

  static constexpr const char *name() noexcept { return "pow"; }
};

struct NegOp {
  template <typename T> static constexpr T apply_scalar(T a) noexcept {
    return -a;
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a) noexcept {
    return -a;
  }

  static constexpr const char *name() noexcept { return "neg"; }
};

// Core SIMD binary operation kernel - fully optimized for performance
template <typename T, typename Op, bool Aligned>
void binary_kernel_impl(const T *__restrict__ lhs, const T *__restrict__ rhs,
                        T *__restrict__ out, std::size_t n) noexcept {
  if constexpr (std::is_same_v<T, half>) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = Op::apply_scalar(lhs[i], rhs[i]);
    }
  } else {
    using batch_type = xsimd::batch<T>;
    constexpr std::size_t batch_size = batch_type::size;
    constexpr std::size_t unroll = 4;

    std::size_t i = 0;
    const std::size_t simd_end = batch_size > 0 ? n - (n % batch_size) : 0;

    auto load = [&](const T *ptr) {
      if constexpr (Aligned)
        return batch_type::load_aligned(ptr);
      else
        return batch_type::load_unaligned(ptr);
    };

    auto store = [&](T *ptr, const batch_type &val) {
      if constexpr (Aligned)
        val.store_aligned(ptr);
      else
        val.store_unaligned(ptr);
    };

    const std::size_t unroll_step = unroll * batch_size;
    for (; i + unroll_step <= simd_end; i += unroll_step) {
      auto a0 = load(lhs + i);
      auto b0 = load(rhs + i);
      auto a1 = load(lhs + i + batch_size);
      auto b1 = load(rhs + i + batch_size);
      auto a2 = load(lhs + i + 2 * batch_size);
      auto b2 = load(rhs + i + 2 * batch_size);
      auto a3 = load(lhs + i + 3 * batch_size);
      auto b3 = load(rhs + i + 3 * batch_size);

      auto r0 = Op::apply_simd(a0, b0);
      auto r1 = Op::apply_simd(a1, b1);
      auto r2 = Op::apply_simd(a2, b2);
      auto r3 = Op::apply_simd(a3, b3);

      store(out + i, r0);
      store(out + i + batch_size, r1);
      store(out + i + 2 * batch_size, r2);
      store(out + i + 3 * batch_size, r3);
    }

    for (; i < simd_end; i += batch_size) {
      auto a_batch = load(lhs + i);
      auto b_batch = load(rhs + i);
      auto result = Op::apply_simd(a_batch, b_batch);
      store(out + i, result);
    }

    for (; i < n; ++i) {
      out[i] = Op::apply_scalar(lhs[i], rhs[i]);
    }
  }
}

// General binary kernel that chooses the best strategy based on alignment
template <typename T, typename Op>
void binary_kernel(const T *__restrict__ lhs, const T *__restrict__ rhs,
                   T *__restrict__ out, std::size_t n) noexcept {
  // Special case for half precision - always use scalar operations
  if constexpr (std::is_same_v<T, half>) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = Op::apply_scalar(lhs[i], rhs[i]);
    }
  } else {
    // For small arrays, use scalar operations directly
    constexpr std::size_t simd_threshold = xsimd::batch<T>::size * 4;
    if (n < simd_threshold) {
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = Op::apply_scalar(lhs[i], rhs[i]);
      }
      return;
    }

    bool aligned =
        is_aligned<T>(lhs) && is_aligned<T>(rhs) && is_aligned<T>(out);
    if (aligned) {
      binary_kernel_impl<T, Op, true>(lhs, rhs, out, n);
    } else {
      binary_kernel_impl<T, Op, false>(lhs, rhs, out, n);
    }
  }
}

// Convenience functions for common operations
template <typename T>
void add(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept {
  binary_kernel<T, AddOp>(lhs, rhs, out, n);
}

template <typename T>
void subtract(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept {
  binary_kernel<T, SubOp>(lhs, rhs, out, n);
}

template <typename T>
void multiply(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept {
  binary_kernel<T, MulOp>(lhs, rhs, out, n);
}

template <typename T>
void divide(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept {
  binary_kernel<T, DivOp>(lhs, rhs, out, n);
}

template <typename T>
void power(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept {
  binary_kernel<T, PowOp>(lhs, rhs, out, n);
}

// Unary operation kernel for operations like negation
template <typename T, typename Op>
void unary_kernel(const T *__restrict__ in, T *__restrict__ out,
                  std::size_t n) noexcept {
  // Special handling for half precision which doesn't support SIMD
  if constexpr (std::is_same_v<T, half>) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = Op::apply_scalar(in[i]);
    }
  } else {
    using batch_type = xsimd::batch<T>;
    constexpr std::size_t batch_size = batch_type::size;

    // For small arrays, use scalar operations directly
    constexpr std::size_t simd_threshold = batch_size * 4;
    if (n < simd_threshold) {
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = Op::apply_scalar(in[i]);
      }
      return;
    }

    std::size_t i = 0;
    const std::size_t simd_end = batch_size > 0 ? n - (n % batch_size) : 0;

    if (is_aligned<T>(in) && is_aligned<T>(out)) {

      for (; i < simd_end; i += batch_size) {
        auto a_batch = batch_type::load_aligned(&in[i]);
        auto result = Op::apply_simd(a_batch);
        result.store_aligned(&out[i]);
      }
    } else {

      for (; i < simd_end; i += batch_size) {
        auto a_batch = batch_type::load_unaligned(&in[i]);
        auto result = Op::apply_simd(a_batch);
        result.store_unaligned(&out[i]);
      }
    }

    for (; i < n; ++i) {
      out[i] = Op::apply_scalar(in[i]);
    }
  }
}

template <typename T> void negate(const T *in, T *out, std::size_t n) noexcept {
  unary_kernel<T, NegOp>(in, out, n);
}

Buffer buffer_add(const Buffer &a, const Buffer &b,
                  const std::string &result_dtype) {

  Buffer result(a.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        auto &a_buf = std::get<VecBuffer<T>>(a.raw());
        auto &b_buf = std::get<VecBuffer<T>>(b.raw());
        add(a_buf.data(), b_buf.data(), out_buf.data(), a.size());
      },
      result.raw());

  return result;
}

Buffer buffer_subtract(const Buffer &a, const Buffer &b,
                       const std::string &result_dtype) {

  Buffer result(a.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        auto &a_buf = std::get<VecBuffer<T>>(a.raw());
        auto &b_buf = std::get<VecBuffer<T>>(b.raw());
        subtract(a_buf.data(), b_buf.data(), out_buf.data(), a.size());
      },
      result.raw());

  return result;
}

Buffer buffer_multiply(const Buffer &a, const Buffer &b,
                       const std::string &result_dtype) {

  Buffer result(a.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        auto &a_buf = std::get<VecBuffer<T>>(a.raw());
        auto &b_buf = std::get<VecBuffer<T>>(b.raw());
        multiply(a_buf.data(), b_buf.data(), out_buf.data(), a.size());
      },
      result.raw());

  return result;
}

Buffer buffer_divide(const Buffer &a, const Buffer &b,
                     const std::string &result_dtype) {

  Buffer result(a.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        auto &a_buf = std::get<VecBuffer<T>>(a.raw());
        auto &b_buf = std::get<VecBuffer<T>>(b.raw());
        divide(a_buf.data(), b_buf.data(), out_buf.data(), a.size());
      },
      result.raw());

  return result;
}

Buffer buffer_power(const Buffer &a, const Buffer &b,
                    const std::string &result_dtype) {

  Buffer result(a.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        auto &a_buf = std::get<VecBuffer<T>>(a.raw());
        auto &b_buf = std::get<VecBuffer<T>>(b.raw());
        power(a_buf.data(), b_buf.data(), out_buf.data(), a.size());
      },
      result.raw());

  return result;
}

Buffer buffer_negate(const Buffer &a, const std::string &result_dtype) {

  Buffer result(a.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        auto &a_buf = std::get<VecBuffer<T>>(a.raw());
        negate(a_buf.data(), out_buf.data(), a.size());
      },
      result.raw());

  return result;
}

Buffer binary_op(const Buffer &a, const Buffer &b, BinaryOpType op,
                 const std::string &result_dtype) {
  if (a.size() != b.size())
    throw std::runtime_error("Buffers must have the same size");

  Buffer a_cast = a.cast(result_dtype);
  Buffer b_cast = b.cast(result_dtype);

  switch (op) {
  case BinaryOpType::ADD:
    return buffer_add(a_cast, b_cast, result_dtype);
  case BinaryOpType::SUB:
    return buffer_subtract(a_cast, b_cast, result_dtype);
  case BinaryOpType::MUL:
    return buffer_multiply(a_cast, b_cast, result_dtype);
  case BinaryOpType::DIV:
    return buffer_divide(a_cast, b_cast, result_dtype);
  case BinaryOpType::POW:
    return buffer_power(a_cast, b_cast, result_dtype);
  default:
    throw std::runtime_error("Unknown binary operation");
  }
}

Buffer unary_op(const Buffer &a, UnaryOpType op,
                const std::string &result_dtype) {
  Buffer a_cast = a.cast(result_dtype);

  switch (op) {
  case UnaryOpType::NEG:
    return buffer_negate(a_cast, result_dtype);
  default:
    throw std::runtime_error("Unknown unary operation");
  }
}

// Define a macro to instantiate a function template for a single type
#define INSTANTIATE_FOR_TYPE(func, type)                                       \
  template void func<type>(const type *, const type *, type *,                 \
                           std::size_t) noexcept;

#define INSTANTIATE_FOR_ALL_NUMERIC_TYPES(func)                                \
  INSTANTIATE_FOR_TYPE(func, float)                                            \
  INSTANTIATE_FOR_TYPE(func, double)                                           \
  INSTANTIATE_FOR_TYPE(func, int8_t)                                           \
  INSTANTIATE_FOR_TYPE(func, uint8_t)                                          \
  INSTANTIATE_FOR_TYPE(func, int16_t)                                          \
  INSTANTIATE_FOR_TYPE(func, uint16_t)                                         \
  INSTANTIATE_FOR_TYPE(func, int32_t)                                          \
  INSTANTIATE_FOR_TYPE(func, uint32_t)                                         \
  INSTANTIATE_FOR_TYPE(func, int64_t)                                          \
  INSTANTIATE_FOR_TYPE(func, uint64_t)                                         \
  INSTANTIATE_FOR_TYPE(func, half)

INSTANTIATE_FOR_ALL_NUMERIC_TYPES(add)
INSTANTIATE_FOR_ALL_NUMERIC_TYPES(subtract)
INSTANTIATE_FOR_ALL_NUMERIC_TYPES(multiply)
INSTANTIATE_FOR_ALL_NUMERIC_TYPES(divide)
INSTANTIATE_FOR_ALL_NUMERIC_TYPES(power)

// Define a macro to instantiate unary function templates
#define INSTANTIATE_UNARY_FOR_TYPE(func, type)                                 \
  template void func<type>(const type *, type *, std::size_t) noexcept;

#define INSTANTIATE_UNARY_FOR_ALL_NUMERIC_TYPES(func)                          \
  INSTANTIATE_UNARY_FOR_TYPE(func, float)                                      \
  INSTANTIATE_UNARY_FOR_TYPE(func, double)                                     \
  INSTANTIATE_UNARY_FOR_TYPE(func, int8_t)                                     \
  INSTANTIATE_UNARY_FOR_TYPE(func, uint8_t)                                    \
  INSTANTIATE_UNARY_FOR_TYPE(func, int16_t)                                    \
  INSTANTIATE_UNARY_FOR_TYPE(func, uint16_t)                                   \
  INSTANTIATE_UNARY_FOR_TYPE(func, int32_t)                                    \
  INSTANTIATE_UNARY_FOR_TYPE(func, uint32_t)                                   \
  INSTANTIATE_UNARY_FOR_TYPE(func, int64_t)                                    \
  INSTANTIATE_UNARY_FOR_TYPE(func, uint64_t)                                   \
  INSTANTIATE_UNARY_FOR_TYPE(func, half)

INSTANTIATE_UNARY_FOR_ALL_NUMERIC_TYPES(negate)

#define INSTANTIATE_BINARY_KERNEL(type, op)                                    \
  template void binary_kernel<type, op>(const type *, const type *, type *,    \
                                        std::size_t) noexcept;

INSTANTIATE_BINARY_KERNEL(float, AddOp)
INSTANTIATE_BINARY_KERNEL(float, SubOp)
INSTANTIATE_BINARY_KERNEL(float, MulOp)
INSTANTIATE_BINARY_KERNEL(float, DivOp)
INSTANTIATE_BINARY_KERNEL(float, PowOp)
INSTANTIATE_BINARY_KERNEL(double, AddOp)
INSTANTIATE_BINARY_KERNEL(double, SubOp)
INSTANTIATE_BINARY_KERNEL(double, MulOp)
INSTANTIATE_BINARY_KERNEL(double, DivOp)
INSTANTIATE_BINARY_KERNEL(double, PowOp)
INSTANTIATE_BINARY_KERNEL(int32_t, AddOp)
INSTANTIATE_BINARY_KERNEL(int32_t, SubOp)
INSTANTIATE_BINARY_KERNEL(int32_t, MulOp)
INSTANTIATE_BINARY_KERNEL(int32_t, DivOp)
INSTANTIATE_BINARY_KERNEL(int32_t, PowOp)
INSTANTIATE_BINARY_KERNEL(int64_t, AddOp)
INSTANTIATE_BINARY_KERNEL(int64_t, SubOp)
INSTANTIATE_BINARY_KERNEL(int64_t, MulOp)
INSTANTIATE_BINARY_KERNEL(int64_t, DivOp)
INSTANTIATE_BINARY_KERNEL(int64_t, PowOp)

#define INSTANTIATE_UNARY_KERNEL(type, op)                                     \
  template void unary_kernel<type, op>(const type *, type *,                   \
                                       std::size_t) noexcept;

INSTANTIATE_UNARY_KERNEL(float, NegOp)
INSTANTIATE_UNARY_KERNEL(double, NegOp)
INSTANTIATE_UNARY_KERNEL(int32_t, NegOp)
INSTANTIATE_UNARY_KERNEL(int64_t, NegOp)

} // namespace simd_ops

PYBIND11_MODULE(operations, m) {
  py::enum_<simd_ops::BinaryOpType>(m, "BinaryOpType")
      .value("ADD", simd_ops::BinaryOpType::ADD)
      .value("SUB", simd_ops::BinaryOpType::SUB)
      .value("MUL", simd_ops::BinaryOpType::MUL)
      .value("DIV", simd_ops::BinaryOpType::DIV)
      .value("POW", simd_ops::BinaryOpType::POW);
  py::enum_<simd_ops::UnaryOpType>(m, "UnaryOpType")
      .value("NEG", simd_ops::UnaryOpType::NEG)
      .value("POW", simd_ops::UnaryOpType::POW);

  m.def("binary_op", &simd_ops::binary_op,
        "Generic binary operation on two buffers", py::arg("a"), py::arg("b"),
        py::arg("op_name"), py::arg("result_dtype"));

  m.def("unary_op", &simd_ops::unary_op, "Generic unary operation on a buffer",
        py::arg("a"), py::arg("op_name"), py::arg("result_dtype"));

  m.doc() = "High-performance SIMD operations for tensors";
}
