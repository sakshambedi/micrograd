// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "operations.h"
#include <algorithm>
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

// Core SIMD binary operation kernel - fully optimized for performance
template <typename T, typename Op>
void binary_kernel_aligned(const T *__restrict__ lhs, const T *__restrict__ rhs,
                           T *__restrict__ out, std::size_t n) noexcept {
  // half precision doesn't have SIMD support yet
  if constexpr (std::is_same_v<T, half>) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = Op::apply_scalar(lhs[i], rhs[i]);
    }
  } else {
    using batch_type = xsimd::batch<T>;
    constexpr std::size_t batch_size = batch_type::size;

    std::size_t i = 0;
    const std::size_t simd_end = batch_size > 0 ? n - (n % batch_size) : 0;

    // Main SIMD loop - assumes aligned memory
    for (; i < simd_end; i += batch_size) {
      auto a_batch = batch_type::load_aligned(&lhs[i]);
      auto b_batch = batch_type::load_aligned(&rhs[i]);
      auto result = Op::apply_simd(a_batch, b_batch);
      result.store_aligned(&out[i]);
    }

    // Handle remaining elements with scalar operations
    for (; i < n; ++i) {
      out[i] = Op::apply_scalar(lhs[i], rhs[i]);
    }
  }
}

// SIMD binary operation kernel with unaligned memory handling
template <typename T, typename Op>
void binary_kernel_unaligned(const T *__restrict__ lhs,
                             const T *__restrict__ rhs, T *__restrict__ out,
                             std::size_t n) noexcept {
  // Special handling for half precision which doesn't support SIMD
  if constexpr (std::is_same_v<T, half>) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = Op::apply_scalar(lhs[i], rhs[i]);
    }
  } else {
    using batch_type = xsimd::batch<T>;
    constexpr std::size_t batch_size = batch_type::size;

    // Process elements in SIMD-sized chunks using unaligned loads/stores
    std::size_t i = 0;
    const std::size_t simd_end = batch_size > 0 ? n - (n % batch_size) : 0;

    // Main SIMD loop with unaligned memory access
    for (; i < simd_end; i += batch_size) {
      auto a_batch = batch_type::load_unaligned(&lhs[i]);
      auto b_batch = batch_type::load_unaligned(&rhs[i]);
      auto result = Op::apply_simd(a_batch, b_batch);
      result.store_unaligned(&out[i]);
    }

    // Handle remaining elements with scalar operations
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

    // Check if all pointers are properly aligned
    if (is_aligned<T>(lhs) && is_aligned<T>(rhs) && is_aligned<T>(out)) {
      binary_kernel_aligned<T, Op>(lhs, rhs, out, n);
    } else {
      binary_kernel_unaligned<T, Op>(lhs, rhs, out, n);
    }
  }
}

// Implementation of buffer_add
Buffer buffer_add(const Buffer &a, const Buffer &b,
                  const std::string &result_dtype) {
  if (a.size() != b.size())
    throw std::runtime_error("Buffers must have the same size");

  Buffer a_cast = a.cast(result_dtype);
  Buffer b_cast = b.cast(result_dtype);
  Buffer result(a_cast.size(), result_dtype);

  std::visit(
      [&](auto &out_buf) {
        using T = std::decay_t<decltype(out_buf[0])>;

        // Always use the add function
        auto &a_buf = std::get<VecBuffer<T>>(a_cast.raw());
        auto &b_buf = std::get<VecBuffer<T>>(b_cast.raw());
        add(a_buf.data(), b_buf.data(), out_buf.data(), a_cast.size());
      },
      result.raw());

  return result;
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

#define INSTANTIATE_BINARY_KERNEL(type, op)                                    \
  template void binary_kernel<type, op>(const type *, const type *, type *,    \
                                        std::size_t) noexcept;

INSTANTIATE_BINARY_KERNEL(float, AddOp)
INSTANTIATE_BINARY_KERNEL(float, SubOp)
INSTANTIATE_BINARY_KERNEL(float, MulOp)
INSTANTIATE_BINARY_KERNEL(float, DivOp)
INSTANTIATE_BINARY_KERNEL(double, AddOp)
INSTANTIATE_BINARY_KERNEL(double, SubOp)
INSTANTIATE_BINARY_KERNEL(double, MulOp)
INSTANTIATE_BINARY_KERNEL(double, DivOp)
INSTANTIATE_BINARY_KERNEL(int32_t, AddOp)
INSTANTIATE_BINARY_KERNEL(int32_t, SubOp)
INSTANTIATE_BINARY_KERNEL(int32_t, MulOp)
INSTANTIATE_BINARY_KERNEL(int32_t, DivOp)
INSTANTIATE_BINARY_KERNEL(int64_t, AddOp)
INSTANTIATE_BINARY_KERNEL(int64_t, SubOp)
INSTANTIATE_BINARY_KERNEL(int64_t, MulOp)
INSTANTIATE_BINARY_KERNEL(int64_t, DivOp)

} // namespace simd_ops
PYBIND11_MODULE(operations, m) {
  m.def("buffer_add", &simd_ops::buffer_add,
        "Element-wise addition of two buffers", py::arg("a"), py::arg("b"),
        py::arg("result_dtype"));

  // Create a function to handle broadcasting for add
  // m.def(
  //     "add",
  //     [](const Buffer &a, const Buffer &b, const std::vector<int> &a_shape,
  //        const std::vector<int> &b_shape, const std::vector<int> &out_shape,
  //        const std::string &result_dtype) {
  //       // For now, we're just implementing buffer_add
  //       // In a complete implementation, you would handle broadcasting here
  //       return simd_ops::buffer_add(a, b, result_dtype);
  //     },
  //     "Element-wise addition with broadcasting support");

  m.doc() = "High-performance SIMD operations for tensors";
}
