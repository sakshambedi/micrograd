// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "operations.h"
#include <algorithm>
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
    return a / b;
  }

  template <typename Batch>
  static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
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

// Explicit template instantiations for all supported types

// Add operations
template void add<float>(const float *, const float *, float *,
                         std::size_t) noexcept;
template void add<double>(const double *, const double *, double *,
                          std::size_t) noexcept;
template void add<int8_t>(const int8_t *, const int8_t *, int8_t *,
                          std::size_t) noexcept;
template void add<uint8_t>(const uint8_t *, const uint8_t *, uint8_t *,
                           std::size_t) noexcept;
template void add<int16_t>(const int16_t *, const int16_t *, int16_t *,
                           std::size_t) noexcept;
template void add<uint16_t>(const uint16_t *, const uint16_t *, uint16_t *,
                            std::size_t) noexcept;
template void add<int32_t>(const int32_t *, const int32_t *, int32_t *,
                           std::size_t) noexcept;
template void add<uint32_t>(const uint32_t *, const uint32_t *, uint32_t *,
                            std::size_t) noexcept;
template void add<int64_t>(const int64_t *, const int64_t *, int64_t *,
                           std::size_t) noexcept;
template void add<uint64_t>(const uint64_t *, const uint64_t *, uint64_t *,
                            std::size_t) noexcept;
template void add<half>(const half *, const half *, half *,
                        std::size_t) noexcept;

// Subtract operations
template void subtract<float>(const float *, const float *, float *,
                              std::size_t) noexcept;
template void subtract<double>(const double *, const double *, double *,
                               std::size_t) noexcept;
template void subtract<int8_t>(const int8_t *, const int8_t *, int8_t *,
                               std::size_t) noexcept;
template void subtract<uint8_t>(const uint8_t *, const uint8_t *, uint8_t *,
                                std::size_t) noexcept;
template void subtract<int16_t>(const int16_t *, const int16_t *, int16_t *,
                                std::size_t) noexcept;
template void subtract<uint16_t>(const uint16_t *, const uint16_t *, uint16_t *,
                                 std::size_t) noexcept;
template void subtract<int32_t>(const int32_t *, const int32_t *, int32_t *,
                                std::size_t) noexcept;
template void subtract<uint32_t>(const uint32_t *, const uint32_t *, uint32_t *,
                                 std::size_t) noexcept;
template void subtract<int64_t>(const int64_t *, const int64_t *, int64_t *,
                                std::size_t) noexcept;
template void subtract<uint64_t>(const uint64_t *, const uint64_t *, uint64_t *,
                                 std::size_t) noexcept;
template void subtract<half>(const half *, const half *, half *,
                             std::size_t) noexcept;

// Multiply operations
template void multiply<float>(const float *, const float *, float *,
                              std::size_t) noexcept;
template void multiply<double>(const double *, const double *, double *,
                               std::size_t) noexcept;
template void multiply<int8_t>(const int8_t *, const int8_t *, int8_t *,
                               std::size_t) noexcept;
template void multiply<uint8_t>(const uint8_t *, const uint8_t *, uint8_t *,
                                std::size_t) noexcept;
template void multiply<int16_t>(const int16_t *, const int16_t *, int16_t *,
                                std::size_t) noexcept;
template void multiply<uint16_t>(const uint16_t *, const uint16_t *, uint16_t *,
                                 std::size_t) noexcept;
template void multiply<int32_t>(const int32_t *, const int32_t *, int32_t *,
                                std::size_t) noexcept;
template void multiply<uint32_t>(const uint32_t *, const uint32_t *, uint32_t *,
                                 std::size_t) noexcept;
template void multiply<int64_t>(const int64_t *, const int64_t *, int64_t *,
                                std::size_t) noexcept;
template void multiply<uint64_t>(const uint64_t *, const uint64_t *, uint64_t *,
                                 std::size_t) noexcept;
template void multiply<half>(const half *, const half *, half *,
                             std::size_t) noexcept;

// Divide operations
template void divide<float>(const float *, const float *, float *,
                            std::size_t) noexcept;
template void divide<double>(const double *, const double *, double *,
                             std::size_t) noexcept;
template void divide<int8_t>(const int8_t *, const int8_t *, int8_t *,
                             std::size_t) noexcept;
template void divide<uint8_t>(const uint8_t *, const uint8_t *, uint8_t *,
                              std::size_t) noexcept;
template void divide<int16_t>(const int16_t *, const int16_t *, int16_t *,
                              std::size_t) noexcept;
template void divide<uint16_t>(const uint16_t *, const uint16_t *, uint16_t *,
                               std::size_t) noexcept;
template void divide<int32_t>(const int32_t *, const int32_t *, int32_t *,
                              std::size_t) noexcept;
template void divide<uint32_t>(const uint32_t *, const uint32_t *, uint32_t *,
                               std::size_t) noexcept;
template void divide<int64_t>(const int64_t *, const int64_t *, int64_t *,
                              std::size_t) noexcept;
template void divide<uint64_t>(const uint64_t *, const uint64_t *, uint64_t *,
                               std::size_t) noexcept;
template void divide<half>(const half *, const half *, half *,
                           std::size_t) noexcept;

// Binary kernel explicit instantiations for important types
template void binary_kernel<float, AddOp>(const float *, const float *, float *,
                                          std::size_t) noexcept;
template void binary_kernel<float, SubOp>(const float *, const float *, float *,
                                          std::size_t) noexcept;
template void binary_kernel<float, MulOp>(const float *, const float *, float *,
                                          std::size_t) noexcept;
template void binary_kernel<float, DivOp>(const float *, const float *, float *,
                                          std::size_t) noexcept;

template void binary_kernel<double, AddOp>(const double *, const double *,
                                           double *, std::size_t) noexcept;
template void binary_kernel<double, SubOp>(const double *, const double *,
                                           double *, std::size_t) noexcept;
template void binary_kernel<double, MulOp>(const double *, const double *,
                                           double *, std::size_t) noexcept;
template void binary_kernel<double, DivOp>(const double *, const double *,
                                           double *, std::size_t) noexcept;

template void binary_kernel<int32_t, AddOp>(const int32_t *, const int32_t *,
                                            int32_t *, std::size_t) noexcept;
template void binary_kernel<int32_t, SubOp>(const int32_t *, const int32_t *,
                                            int32_t *, std::size_t) noexcept;
template void binary_kernel<int32_t, MulOp>(const int32_t *, const int32_t *,
                                            int32_t *, std::size_t) noexcept;
template void binary_kernel<int32_t, DivOp>(const int32_t *, const int32_t *,
                                            int32_t *, std::size_t) noexcept;

template void binary_kernel<int64_t, AddOp>(const int64_t *, const int64_t *,
                                            int64_t *, std::size_t) noexcept;
template void binary_kernel<int64_t, SubOp>(const int64_t *, const int64_t *,
                                            int64_t *, std::size_t) noexcept;
template void binary_kernel<int64_t, MulOp>(const int64_t *, const int64_t *,
                                            int64_t *, std::size_t) noexcept;
template void binary_kernel<int64_t, DivOp>(const int64_t *, const int64_t *,
                                            int64_t *, std::size_t) noexcept;

} // namespace simd_ops
