// Copyright 2025 Saksham Bedi

#include "vecbuffer.h"
#include <Eigen/Core>

// explicit template instantiations for common types

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

// broadcast variants for float types
template void binary_kernel_broadcast<float, AddOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
template void binary_kernel_broadcast<float, SubOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
template void binary_kernel_broadcast<float, MulOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);
template void binary_kernel_broadcast<float, DivOp>(
    const float *, const std::vector<std::size_t> &, const float *,
    const std::vector<std::size_t> &, float *, const std::vector<std::size_t> &);

#undef INST
