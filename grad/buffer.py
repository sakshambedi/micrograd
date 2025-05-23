import array
import threading
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from grad.dtype import DType, DTypeLike, dtypes, to_dtype
from grad.utils.fp16 import float16_to_uint16

ARRAY_E_SUPPORTED = "e" in array.typecodes


class BufferPool:
    """Thread-safe buffer pool with size bucketing for better memory reuse"""

    def __init__(self) -> None:
        # pools[fmt][bucket_size] -> list[array.array]
        self._pools: dict[str, dict[int, list[array.array]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._lock = threading.RLock()
        self._max_pool_size = 128

    def get_buffer(self, fmt: str, size: int) -> array.array:
        """Fetch a buffer of at least size elements."""
        if size == 0:
            return array.array(fmt)

        bucket = self._next_pow2(size)
        with self._lock:
            pool = self._pools[fmt][bucket]
            if pool:
                buf = pool.pop()
                buf[:size] = array.array(fmt, [0] * size)
                return buf

        return array.array(fmt, [0] * bucket)

    def release_buffer(self, buf: array.array, fmt: str) -> None:
        """Return a buffer to the pool for future reuse."""
        if not buf:
            return
        bucket = len(buf)
        with self._lock:
            pool = self._pools[fmt][bucket]
            if len(pool) < self._max_pool_size:
                pool.append(buf)

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()


_buffer_pool = BufferPool()  # singleton instance


class Buffer:
    __slots__ = ("dtype", "_storage")
    _buffer_pool = {}

    def __init__(self, dtype: DTypeLike, iterable: Iterable[Any]):
        self.dtype: DType = to_dtype(dtype)
        self._storage: memoryview = self._make_buffer(dtype, iterable)

    def to(self, device): ...  # noqa: E704

    @classmethod
    def _get_buffer(cls, fmt: str, size: int) -> array.array:
        """Borrow from the global, thread-safe pool."""
        return _buffer_pool.get_buffer(fmt, size)

    @classmethod
    def _release_buffer(cls, buf, fmt: str):
        """Return a buffer to the pool"""
        _buffer_pool.release_buffer(buf, fmt)

    def allocate_buffer(self):
        """Allocates an empty buffer of the correct size and type."""
        storage_fmt = dtypes._storage_format(self.dtype)
        return array.array(storage_fmt)

    @staticmethod
    def _make_buffer(type_: DTypeLike, iterable: Iterable[Any]) -> memoryview:
        dtype = to_dtype(type_)
        fmt = dtypes._storage_format(dtype)

        data = list(iterable)
        arr = _buffer_pool.get_buffer(fmt, len(data))

        if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            values = [float16_to_uint16(float(v)) for v in data]
            arr[: len(data)] = array.array(fmt, values)
        elif dtype.fmt == "?":
            values = [1 if bool(v) else 0 for v in data]
            arr[: len(data)] = array.array(fmt, values)
        else:
            arr[: len(data)] = array.array(fmt, data)

        return memoryview(arr)

    def to_list(self) -> list[Any]:
        return [dtypes._from_storage(item, self.dtype) for item in self._storage]

    @classmethod
    def _filled(cls, dtype: DTypeLike, num_elem: int, val: int | float) -> "Buffer":
        out_dtype = to_dtype(dtype)
        valiter = (val for _ in range(num_elem)) if num_elem != 0 else []
        buff = cls.__new__(cls)
        buff.dtype = out_dtype
        buff._storage = Buffer._make_buffer(out_dtype, valiter)
        return buff

    def __getitem__(self, idx: int):
        return self._storage[idx]

    # @staticmethod
    # def _broadcast(t: Tensor, shape, fmt) -> Tensor:
    #     if t.shape == shape:
    #         return t

    #     # Broadcasting a scalar tensor to the target shape.
    #     if t.shape == ():
    #         scalar_val = _from_storage(t._buffer[0], t.dtype)
    #         storage_fmt = _storage_format(t.dtype)
    #         numel = _prod(shape)

    #         arr = array.array(storage_fmt, [_to_storage(scalar_val, fmt)] * numel)
    #         new_tensor = Tensor.__new__(Tensor)
    #         new_tensor.dtype = fmt
    #         new_tensor.shape = shape
    #         new_tensor.device = t.device
    #         new_tensor.requires_grad = t.requires_grad
    #         new_tensor.grad = None
    #         new_tensor._buffer = memoryview(arr)
    #         return new_tensor
    #     raise ValueError(f"Cannot broadcast tensor with shape {t.shape} to {shape}")
