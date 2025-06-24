// Copyright 2025 Saksham Bedi

#ifndef KERNELS_OPERATIONS_H_
#define KERNELS_OPERATIONS_H_

#include "cpu_kernel.h"
#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <xsimd/xsimd.hpp>

using half = Eigen::half;

namespace simd_ops {

enum class BinaryOpType : std::uint8_t { ADD, SUB, MUL, DIV };

Buffer binary_op(const Buffer &a, const Buffer &b, BinaryOpType op,
                 const std::string &result_dtype);

// Public interface for buffer operations
// Type trait to check if a type supports SIMD operations
template <typename T> struct is_simd_supported : std::false_type {};

template <> struct is_simd_supported<float> : std::true_type {};
template <> struct is_simd_supported<double> : std::true_type {};
template <> struct is_simd_supported<int8_t> : std::true_type {};
template <> struct is_simd_supported<uint8_t> : std::true_type {};
template <> struct is_simd_supported<int16_t> : std::true_type {};
template <> struct is_simd_supported<uint16_t> : std::true_type {};
template <> struct is_simd_supported<int32_t> : std::true_type {};
template <> struct is_simd_supported<uint32_t> : std::true_type {};
template <> struct is_simd_supported<int64_t> : std::true_type {};
template <> struct is_simd_supported<uint64_t> : std::true_type {};
template <>
struct is_simd_supported<half> : std::false_type {
}; // half doesn't support SIMD

template <typename T>
constexpr bool is_simd_supported_v = is_simd_supported<T>::value;

// Function declarations
Buffer buffer_add(const Buffer &a, const Buffer &b,
                  const std::string &result_dtype);

// Convenience functions for common operations on raw pointers
template <typename T>
void add(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept;

template <typename T>
void subtract(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept;

template <typename T>
void multiply(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept;

template <typename T>
void divide(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept;

// Performance information utilities
template <typename T> constexpr std::size_t simd_width() {
  return xsimd::batch<T>::size;
}

template <typename T> constexpr std::size_t simd_bytes() {
  return xsimd::batch<T>::size * sizeof(T);
}

// Memory alignment utilities
template <typename T> constexpr std::size_t simd_alignment() {
  return xsimd::batch<T>::size * sizeof(T);
}

template <typename T> inline bool is_aligned(const void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % simd_alignment<T>() == 0;
}

template <typename T> inline std::size_t align_offset(const void *ptr) {
  const auto alignment = simd_alignment<T>();
  const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  const auto offset = alignment - (addr % alignment);
  return (offset == alignment) ? 0 : offset / sizeof(T);
}

} // namespace simd_ops

#endif // KERNELS_OPERATIONS_H_"
