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
