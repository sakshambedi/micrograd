// Copyright 2025 Saksham Bedi

#include "vecbuffer.h"
#include <Eigen/Core>
#include <vector>

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

// Specialized non-SIMD implementation for Eigen::half
template <class Op>
void binary_kernel(const Eigen::half *lhs, const Eigen::half *rhs,
                   Eigen::half *out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = Op::scalar(lhs[i], rhs[i]);
  }
}

// Implementation of _make_stride helper function
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

// Explicit template instantiations for common types

#define INST(T)                                                                \
  template void binary_kernel<T, AddOp>(const T *, const T *, T *,             \
                                        std::size_t);                          \
  template void binary_kernel<T, SubOp>(const T *, const T *, T *,             \
                                        std::size_t);                          \
  template void binary_kernel<T, MulOp>(const T *, const T *, T *,             \
                                        std::size_t);                          \
  template void binary_kernel<T, DivOp>(const T *, const T *, T *, std::size_t);

INST(float)
INST(double)
INST(int32_t)
INST(int64_t)
INST(uint32_t)
INST(uint64_t)

// Specialized instantiations for Eigen::half using the non-SIMD version
template void binary_kernel<AddOp>(const Eigen::half *, const Eigen::half *,
                                   Eigen::half *, std::size_t);
template void binary_kernel<SubOp>(const Eigen::half *, const Eigen::half *,
                                   Eigen::half *, std::size_t);
template void binary_kernel<MulOp>(const Eigen::half *, const Eigen::half *,
                                   Eigen::half *, std::size_t);
template void binary_kernel<DivOp>(const Eigen::half *, const Eigen::half *,
                                   Eigen::half *, std::size_t);

// Explicit instantiations for _make_stride
template std::vector<std::size_t>
_make_stride<float>(const std::vector<std::size_t> &);
template std::vector<std::size_t>
_make_stride<double>(const std::vector<std::size_t> &);
template std::vector<std::size_t>
_make_stride<Eigen::half>(const std::vector<std::size_t> &);

// broadcast variants for float types
template void binary_kernel_broadcast<float, AddOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
template void binary_kernel_broadcast<float, SubOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
template void binary_kernel_broadcast<float, MulOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
template void binary_kernel_broadcast<float, DivOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);

#undef INST
