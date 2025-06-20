// Copyright 2025 Saksham Bedi
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <xsimd/xsimd.hpp>

using half = Eigen::half;

template <typename T>
using AlignedVec = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T> class VecBuffer {
public:
  explicit VecBuffer(std::size_t n = 0)
      : data_(std::max(size_t(1), n)), size_(n) {}
  VecBuffer(const T *src, std::size_t n)
      : data_(std::max(size_t(1), n)), size_(n) {
    if (n > 0) {
      std::copy(src, src + n, data_.data());
    }
  }

  [[nodiscard]] std::size_t size() const { return size_; }
  T *data() { return data_.data(); }
  const T *data() const { return data_.data(); }

  T &operator[](std::size_t i) { return data_[i]; }
  const T &operator[](std::size_t i) const { return data_[i]; }

private:
  AlignedVec<T> data_;
  std::size_t size_;
};

// SIMD binary kernels
struct AddOp {
  template <class B> static B apply(const B &a, const B &b) { return a + b; }
  template <class T> static T scalar(T a, T b) { return a + b; }
};
struct SubOp {
  template <class B> static B apply(const B &a, const B &b) { return a - b; }
  template <class T> static T scalar(T a, T b) { return a - b; }
};
struct MulOp {
  template <class B> static B apply(const B &a, const B &b) { return a * b; }
  template <class T> static T scalar(T a, T b) { return a * b; }
};
struct DivOp {
  template <class B> static B apply(const B &a, const B &b) { return a / b; }
  template <class T> static T scalar(T a, T b) { return a / b; }
};

template <typename T, class Op>
void binary_kernel(const T *lhs, const T *rhs, T *out, std::size_t n) {
  using batch = xsimd::batch<T>;
  constexpr std::size_t stride = batch::size;
  std::size_t i = 0;
  for (; i + stride <= n; i += stride) {
    batch l = batch::load_aligned(lhs + i);
    batch r = batch::load_aligned(rhs + i);
    batch o = Op::apply(l, r);
    o.store_aligned(out + i);
  }
  for (; i < n; ++i) {
    out[i] = Op::scalar(lhs[i], rhs[i]);
  }
}

// Specialized non-SIMD implementation for Eigen::half,
// keeps fp16 supports on cpu
template <class Op>
void binary_kernel(const Eigen::half *lhs, const Eigen::half *rhs,
                   Eigen::half *out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = Op::scalar(lhs[i], rhs[i]);
  }
}

extern template void binary_kernel<float, AddOp>(const float *, const float *,
                                                 float *, std::size_t);
extern template void binary_kernel<float, SubOp>(const float *, const float *,
                                                 float *, std::size_t);
extern template void binary_kernel<float, MulOp>(const float *, const float *,
                                                 float *, std::size_t);
extern template void binary_kernel<float, DivOp>(const float *, const float *,
                                                 float *, std::size_t);

extern template void binary_kernel<double, AddOp>(const double *,
                                                  const double *, double *,
                                                  std::size_t);
extern template void binary_kernel<double, SubOp>(const double *,
                                                  const double *, double *,
                                                  std::size_t);
extern template void binary_kernel<double, MulOp>(const double *,
                                                  const double *, double *,
                                                  std::size_t);
extern template void binary_kernel<double, DivOp>(const double *,
                                                  const double *, double *,
                                                  std::size_t);

extern template void binary_kernel<Eigen::half, AddOp>(const Eigen::half *,
                                                       const Eigen::half *,
                                                       Eigen::half *,
                                                       std::size_t);
extern template void binary_kernel<Eigen::half, SubOp>(const Eigen::half *,
                                                       const Eigen::half *,
                                                       Eigen::half *,
                                                       std::size_t);
extern template void binary_kernel<Eigen::half, MulOp>(const Eigen::half *,
                                                       const Eigen::half *,
                                                       Eigen::half *,
                                                       std::size_t);
extern template void binary_kernel<Eigen::half, DivOp>(const Eigen::half *,
                                                       const Eigen::half *,
                                                       Eigen::half *,
                                                       std::size_t);

template <typename T>
inline std::vector<std::size_t>
_make_stride(const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> stride(shape.size());
  std::size_t acc = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    stride[i] = acc;
    acc *= shape[i];
  }
  return stride;
}

template <typename T, class Op>
void binary_kernel_broadcast(const T *lhs, const std::vector<std::size_t> &lhs_shape,
                             const T *rhs, const std::vector<std::size_t> &rhs_shape,
                             T *out, const std::vector<std::size_t> &out_shape) {
  auto lhs_stride = _make_stride<T>(lhs_shape);
  auto rhs_stride = _make_stride<T>(rhs_shape);
  auto out_stride = _make_stride<T>(out_shape);

  std::size_t dims = out_shape.size();
  std::size_t total = 1;
  for (auto s : out_shape)
    total *= s;

  for (std::size_t idx = 0; idx < total; ++idx) {
    std::size_t tmp = idx;
    std::size_t loff = 0, roff = 0;
    for (std::size_t d = 0; d < dims; ++d) {
      std::size_t coord = tmp / out_stride[d];
      tmp %= out_stride[d];

      if (d >= dims - lhs_shape.size()) {
        std::size_t lcoord = lhs_shape[d - (dims - lhs_shape.size())] == 1
                                ? 0
                                : coord;
        loff += lcoord * lhs_stride[d - (dims - lhs_shape.size())];
      }
      if (d >= dims - rhs_shape.size()) {
        std::size_t rcoord = rhs_shape[d - (dims - rhs_shape.size())] == 1
                                ? 0
                                : coord;
        roff += rcoord * rhs_stride[d - (dims - rhs_shape.size())];
      }
    }
    out[idx] = Op::scalar(lhs[loff], rhs[roff]);
  }
}

extern template void binary_kernel_broadcast<float, AddOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
extern template void binary_kernel_broadcast<float, SubOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
extern template void binary_kernel_broadcast<float, MulOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
extern template void binary_kernel_broadcast<float, DivOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
