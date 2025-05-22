import array
from collections.abc import Iterable
from typing import Any

from grad.dtype import DType, DTypeLike, to_dtype
from grad.utils.fp16 import float16_to_uint16, uint16_to_float16

ARRAY_E_SUPPORTED = "e" in array.typecodes


class Buffer:
    __slots__ = ("dtype", "_storage")

    def __init__(self, dtype: DTypeLike, iterable: Iterable[Any]):
        self.dtype: DType = to_dtype(dtype)
        self._storage: memoryview = self._make_buffer(dtype, iterable)

    def to(self, device): ...  # noqa: E704

    def allocate_buffer(self):
        """Allocates an empty buffer of the correct size and type."""
        storage_fmt = self._storage_format(self.dtype)
        return array.array(storage_fmt)

    @staticmethod
    def _make_buffer(type, iterable: Iterable[Any]) -> memoryview:
        dtype = to_dtype(type)
        storage_fmt = Buffer._storage_format(dtype)

        if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            arr = array.array(
                storage_fmt, [float16_to_uint16(float(val)) for val in iterable]
            )  # fp16 -> uint16
        elif dtype.fmt == "?":
            arr = array.array(storage_fmt, [1 if bool(val) else 0 for val in iterable])
        else:
            arr = array.array(storage_fmt, iterable)
        return memoryview(arr)

    @staticmethod
    def _storage_format(dtype: DType) -> str:
        """Return the actual format code used to store data in the buffer."""
        if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            return "H"  # store fp16 as uint16
        elif dtype.fmt == "?":
            return "b"  # store bools as signed char
        elif dtype.fmt is None:
            raise TypeError(f"Unsupported dtype {dtype.name} (no format string)")
        return dtype.fmt

    @staticmethod
    def _to_storage(val: Any, dtype: DType) -> Any:
        """Convert val (python scalar) to the representation expected by storage.
        Keeps numeric types fast for the common cases, only struct‑packs for fp16.
        """
        if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            # Use manual conversion when struct 'e' is not supported
            return float16_to_uint16(float(val))
        elif dtype.fmt == "?":
            return 1 if bool(val) else 0
        return val

    @staticmethod
    def _from_storage(stored: Any, dtype: DType) -> Any:
        """Inverse of _to_storage, read a python value from raw buffer item."""
        if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            return uint16_to_float16(stored)
        elif dtype.fmt == "?":
            return bool(stored)
        return stored

    def to_list(self) -> list:
        return [Buffer._from_storage(item, self.dtype) for item in self._storage]

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
