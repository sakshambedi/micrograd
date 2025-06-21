// Copyright 2025 Saksham Bedi
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <limits>
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

  // Constructor from initializer list for ease of use in tests
  VecBuffer(std::initializer_list<T> init)
      : data_(std::max(size_t(1), init.size())), size_(init.size()) {
    if (init.size() > 0) {
      std::copy(init.begin(), init.end(), data_.data());
    }
  }

  [[nodiscard]] std::size_t size() const { return size_; }
  T *data() { return data_.data(); }
  const T *data() const { return data_.data(); }

  T &operator[](std::size_t i) { return data_[i]; }
  const T &operator[](std::size_t i) const { return data_[i]; }

  // Cast buffer from one type to another (static method)
  template <typename OutT> static VecBuffer<OutT> cast(const VecBuffer<T> &in) {
    std::size_t n = in.size();
    VecBuffer<OutT> out(n);

    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> a(in.data(), n);
    Eigen::Map<Eigen::Array<OutT, Eigen::Dynamic, 1>> b(out.data(), n);

    b = a.template cast<OutT>();

    return out;
  }

  // Non-static member cast method for convenience
  template <typename OutT> VecBuffer<OutT> cast() const {
    return VecBuffer<T>::template cast<OutT>(*this);
  }

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
void binary_kernel(const T *lhs, const T *rhs, T *out, std::size_t n);

// Specialized non-SIMD implementation for Eigen::half,
// keeps fp16 supports on cpu
template <class Op>
void binary_kernel(const Eigen::half *lhs, const Eigen::half *rhs,
                   Eigen::half *out, std::size_t n);

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
std::vector<std::size_t> _make_stride(const std::vector<std::size_t> &shape);

template <typename T, class Op>
void binary_kernel_broadcast(const T *lhs,
                             const std::vector<std::size_t> &lhs_shape,
                             const T *rhs,
                             const std::vector<std::size_t> &rhs_shape, T *out,
                             const std::vector<std::size_t> &out_shape);

extern template void binary_kernel_broadcast<float, AddOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
extern template void binary_kernel_broadcast<float, SubOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
extern template void binary_kernel_broadcast<float, MulOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
extern template void binary_kernel_broadcast<float, DivOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *,
    const std::vector<std::size_t> &);
