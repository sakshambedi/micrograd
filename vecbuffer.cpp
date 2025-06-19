#include "vecbuffer.h"

// explicit template instantiations for common types

#define INST(T) \
    template void binary_kernel<T, AddOp>(const T*, const T*, T*, std::size_t); \
    template void binary_kernel<T, SubOp>(const T*, const T*, T*, std::size_t); \
    template void binary_kernel<T, MulOp>(const T*, const T*, T*, std::size_t); \
    template void binary_kernel<T, DivOp>(const T*, const T*, T*, std::size_t);

INST(float)
INST(double)
INST(int32_t)
INST(int64_t)
INST(uint32_t)
INST(uint64_t)

#undef INST
