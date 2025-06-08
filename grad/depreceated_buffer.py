# import array
# import threading
# from collections import defaultdict
# from collections.abc import Iterable
# from typing import Any, Optional

# from grad.device import Device
# from grad.kernels import cpu_kernel

# try:
#     import psutil

#     HAS_PSUTIL = True
# except ImportError:
#     HAS_PSUTIL = False

# from grad.dtype import DType, DTypeLike, dtypes, to_dtype
# from grad.utils.constants import ARRAY_E_SUPPORTED
# from grad.utils.fp16 import float16_to_uint16


# class BufferPool:
#     """Thread-safe buffer pool with dynamic memory management, similar to PyTorch's allocator"""

#     def __init__(self, max_pool_size: Optional[int] = None, memory_fraction: float = 0.8) -> None:
#         # pools[fmt][bucket_size] -> list[array.array]
#         self._pools: dict[str, dict[int, list[array.array]]] = defaultdict(
#             lambda: defaultdict(list)
#         )
#         self._lock = threading.RLock()
#         self._max_pool_size = max_pool_size  # None means unlimited
#         self._memory_fraction = memory_fraction  # Maximum fraction of system memory to use
#         self._total_cached_bytes = 0
#         self._allocation_stats = {"hits": 0, "misses": 0, "releases": 0}

#     def get_buffer(self, fmt: str, size: int) -> array.array:
#         """Fetch a buffer of at least size elements."""
#         if size == 0:
#             return array.array(fmt)

#         bucket = self._next_pow2(size)
#         with self._lock:
#             pool = self._pools[fmt][bucket]
#             if pool:
#                 buf = pool.pop()
#                 self._allocation_stats["hits"] += 1
#                 # Update cached memory tracking
#                 self._total_cached_bytes -= self._get_buffer_size_bytes(buf, fmt)
#                 # Clear and resize buffer to requested size
#                 buf[:] = array.array(fmt, [0] * size)
#                 return buf

#         # Cache miss - create new buffer
#         self._allocation_stats["misses"] += 1
#         return array.array(fmt, [0] * bucket)

#     def release_buffer(self, buf: array.array, fmt: str) -> None:
#         """Return a buffer to the pool for future reuse."""
#         if not buf:
#             return

#         bucket = len(buf)
#         buffer_size_bytes = self._get_buffer_size_bytes(buf, fmt)

#         with self._lock:
#             # Check memory pressure before adding to pool
#             if self._should_cache_buffer(buffer_size_bytes):
#                 pool = self._pools[fmt][bucket]

#                 # Apply pool size limit if specified
#                 if self._max_pool_size is None or len(pool) < self._max_pool_size:
#                     pool.append(buf)
#                     self._total_cached_bytes += buffer_size_bytes
#                     self._allocation_stats["releases"] += 1

#     def clear_cache(self) -> None:
#         """Clear all cached buffers, similar to torch.cuda.empty_cache()"""
#         with self._lock:
#             self._pools.clear()
#             self._total_cached_bytes = 0

#     def get_memory_stats(self) -> dict[str, Any]:
#         """Get memory usage and allocation statistics"""
#         with self._lock:
#             total_buffers = sum(
#                 len(pool) for format_pools in self._pools.values() for pool in format_pools.values()
#             )

#             h_rate = (hits := self._allocation_stats["hits"]) / max(
#                 1,
#                 hits + (misses := self._allocation_stats["misses"]),  # noqa : W503
#             )
#             return {
#                 "total_cached_bytes": self._total_cached_bytes,
#                 "total_cached_buffers": total_buffers,
#                 "allocation_hits": hits,
#                 "allocation_misses": misses,
#                 "releases": self._allocation_stats["releases"],
#                 "hit_rate": h_rate,
#             }

#     def _should_cache_buffer(self, buffer_size_bytes: int) -> bool:
#         """Determine if we should cache this buffer based on memory pressure"""
#         if not HAS_PSUTIL:
#             max_cache_bytes = 1024 * 1024 * 1024  # 1GB
#             return (self._total_cached_bytes + buffer_size_bytes) < max_cache_bytes

#         # Check if adding this buffer would exceed memory limits
#         available_memory = psutil.virtual_memory().available
#         total_memory = psutil.virtual_memory().total
#         max_cache_memory = total_memory * self._memory_fraction

#         return (self._total_cached_bytes + buffer_size_bytes) < min(
#             max_cache_memory, available_memory
#         )

#     def _get_buffer_size_bytes(self, buf: array.array, fmt: str) -> int:
#         """Calculate the memory footprint of a buffer"""
#         return len(buf) * buf.itemsize

#     @staticmethod
#     def _next_pow2(n: int) -> int:
#         return 1 if n <= 1 else 1 << (n - 1).bit_length()


# _buffer_pool = BufferPool()  # singleton instance


# class Bufferv1:
#     __slots__ = ("dtype", "_storage", "_fmt")

#     def __init__(self, dtype: DTypeLike, iterable: Iterable[Any]):
#         self.dtype: DType = to_dtype(dtype)

#         self._storage: memoryview = self._make_buffer(dtype, iterable)
#         # self._storage = cpu_kernel.VecBufferFloat(len(iterable))
#         # for i, v in enumerate(iterable):
#         #     self._storage[i] = float(v)

#     def to(self, device: Device): ...  # noqa: E704

#     @property
#     def fmt(self):
#         return self._fmt

#     def __del__(self):
#         """Return buffer to pool when Buffer is garbage collected"""
#         if hasattr(self, "_storage") and hasattr(self, "_fmt"):
#             try:
#                 (
#                     self._release_buffer(underlying_array, self._fmt)
#                     if isinstance(underlying_array := self._storage.obj, array.array)
#                     else None
#                 )
#             except (AttributeError, TypeError):
#                 ...

#     @classmethod
#     def _get_buffer(cls, fmt: str, size: int) -> array.array:
#         """Borrow from the global, thread-safe pool."""
#         return _buffer_pool.get_buffer(fmt, size)

#     @classmethod
#     def _release_buffer(cls, buf, fmt: str):
#         """Return a buffer to the pool"""
#         _buffer_pool.release_buffer(buf, fmt)

#     @classmethod
#     def clear_cache(cls):
#         """Clear the global buffer cache"""
#         _buffer_pool.clear_cache()

#     @classmethod
#     def get_memory_stats(cls):
#         """Get global buffer pool memory statistics"""
#         return _buffer_pool.get_memory_stats()

#     @classmethod
#     def configure_pool(cls, max_pool_size: Optional[int] = None, memory_fraction: float = 0.8):
#         """Configure the global buffer pool settings"""
#         global _buffer_pool
#         _buffer_pool = BufferPool(max_pool_size=max_pool_size, memory_fraction=memory_fraction)

#     def allocate_buffer(self):
#         """Allocates an empty buffer of the correct size and type."""
#         storage_fmt = dtypes._storage_format(self.dtype)
#         return array.array(storage_fmt)

#     @staticmethod
#     def _make_buffer(type_: DTypeLike, iterable: Iterable[Any]) -> memoryview:
#         dtype = to_dtype(type_)
#         fmt = dtypes._storage_format(dtype)

#         data = list(iterable)
#         arr = _buffer_pool.get_buffer(fmt, len(data))

#         if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
#             values = [float16_to_uint16(float(v)) for v in data]
#             arr[: len(data)] = array.array(fmt, values)
#         elif dtype.fmt == "?":
#             values = [1 if bool(v) else 0 for v in data]
#             arr[: len(data)] = array.array(fmt, values)
#         else:
#             arr[: len(data)] = array.array(fmt, data)
#         return memoryview(arr)[: len(data)]

#     def clone(self) -> "Bufferv1":
#         """Create a copy of this buffer"""
#         return Bufferv1(self.dtype, self.to_list())

#     def resize(self, new_size: int) -> "Bufferv1":
#         """Create a new buffer with different size, preserving existing data where possible"""
#         current_data = self.to_list()
#         if new_size > len(current_data):
#             current_data.extend([0] * (new_size - len(current_data)))
#         elif new_size < len(current_data):
#             current_data = current_data[:new_size]  # Truncate

#         new_buffer = Bufferv1(self.dtype, current_data)  # Create new buffer w resized data
#         return new_buffer

#     def to_list(self) -> list[Any]:
#         return [dtypes._from_storage(item, self.dtype) for item in self._storage]

#     @classmethod
#     def _filled(
#         cls: type["Bufferv1"], dtype: DTypeLike, num_elem: int, val: int | float
#     ) -> "Bufferv1":
#         out_dtype = to_dtype(dtype)
#         valiter = (val for _ in range(num_elem)) if num_elem != 0 else []
#         buff: "Bufferv1" = cls.__new__(cls)
#         buff.dtype = out_dtype
#         buff._fmt = dtypes._storage_format(out_dtype)
#         buff._storage = cls._make_buffer(out_dtype, valiter)
#         return buff

#     def __getitem__(self, idx: int):
#         return self._storage[idx]

#     def __setitem__(self, idx: int, value: Any) -> None:
#         storage_value = dtypes._to_storage(value, self.dtype)
#         self._storage[idx] = storage_value

#     def __len__(self) -> int:
#         return len(self._storage)

#     def size_bytes(self) -> int:
#         """Return the size of this buffer in bytes"""
#         return len(self._storage) * self._storage.itemsize if self._storage else 0


# # Convenience functions for global buffer management
# def clear_buffer_cache():
#     """Clear the global buffer cache"""
#     _buffer_pool.clear_cache()


# def get_buffer_memory_stats():
#     """Get global buffer pool memory statistics"""
#     return _buffer_pool.get_memory_stats()


# def configure_buffer_pool(max_pool_size: Optional[int] = None, memory_fraction: float = 0.8):
#     """Configure the global buffer pool settings"""
#     global _buffer_pool
#     _buffer_pool = BufferPool(max_pool_size=max_pool_size, memory_fraction=memory_fraction)
