// Copyright 2025 Saksham Bedi

#include "vecbuffer.h"
#include <Eigen/Core>
#include <vector>

// Specialized non-SIMD implementation for Eigen::half
// TODO(saksy): needs to go the operations.cpp
template <class Op>
void binary_kernel(const Eigen::half *lhs, const Eigen::half *rhs,
                   Eigen::half *out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = Op::scalar(lhs[i], rhs[i]);
  }
}

template <typename T>
std::vector<std::size_t> _make_stride(const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> stride(shape.size());
  std::size_t acc = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    stride[i] = acc;
    acc *= shape[i];
  }
  return stride;
}

// Implementation of binary_kernel_broadcast
template <typename T, class Op>
void binary_kernel_broadcast(const T *lhs,
                             const std::vector<std::size_t> &lhs_shape,
                             const T *rhs,
                             const std::vector<std::size_t> &rhs_shape, T *out,
                             const std::vector<std::size_t> &out_shape) {
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
        std::size_t lcoord =
            lhs_shape[d - (dims - lhs_shape.size())] == 1 ? 0 : coord;
        loff += lcoord * lhs_stride[d - (dims - lhs_shape.size())];
      }
      if (d >= dims - rhs_shape.size()) {
        std::size_t rcoord =
            rhs_shape[d - (dims - rhs_shape.size())] == 1 ? 0 : coord;
        roff += rcoord * rhs_stride[d - (dims - rhs_shape.size())];
      }
    }
    out[idx] = Op::scalar(lhs[loff], rhs[roff]);
  }
}
