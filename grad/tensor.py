from __future__ import annotations

import array
import struct  # Import the struct module

from grad.dtype import DType, DTypeLike, dtypes, to_dtype
from grad.ops import UOp
from grad.utils.fp16 import formatted_fp16_buffer

__slots__ = ("lazydata", "requires_grad", "grad")  # reserved slots

ARRAY_E_SUPPORTED = "e" in array.typecodes


class Tensor:
    """A `Tensor` is a multi-dimensional matrix containing elements of a single data type, backed by a memoryview."""

    def __init__(
        self,
        data: list | tuple | UOp | None,
        device: str | None = "cpu",
        dtype: DTypeLike = dtypes.float32,
        requires_grad: bool | None = None,
    ):
        self.dtype = to_dtype(dtype)
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None

        if isinstance(data, (list, tuple)):
            # infer shape and flatten data
            self.shape = self._infer_shape(data)
            flat = self._flatten(data)

            dtype_obj = self.dtype

            if dtype_obj.fmt is None:
                raise TypeError(f"Unsupported dtype {dtype_obj.name} for Tensor data")

            if dtype_obj.fmt == "e":
                try:
                    # Use little-endian format '<' for packing
                    # We store the raw bytes and create a memoryview of bytes.
                    # The interpretation will be handled in _buffer_to_nested.
                    raw = bytearray(struct.pack(f"<{len(flat)}{dtype_obj.fmt}", *flat))
                    self._raw = raw  # Store the bytearray
                    self._buffer = memoryview(raw).cast("H")
                except struct.error as e:
                    # Catch potential errors during struct packing
                    raise TypeError(f"Could not pack data with format '{dtype_obj.fmt}': {e}")
            else:
                try:
                    # array.array expects a list of numbers. It handles basic type conversions.
                    arr = array.array(dtype_obj.fmt, flat)
                    self._raw = arr
                    self._buffer = memoryview(arr)
                except TypeError as e:
                    # Catch potential errors during array.array creation (e.g., invalid data for format)
                    raise TypeError(
                        f"Could not create array.array with format '{dtype_obj.fmt}': {e}"
                    )

        else:
            raise TypeError(f"Unsupported data type {type(data)} for Tensor initialization")

    @staticmethod
    def _flatten(x):
        """Recursively flatten nested lists/tuples."""
        if isinstance(x, (list, tuple)):
            return [y for sub in x for y in Tensor._flatten(sub)]
        return [x]

    @staticmethod
    def _infer_shape(x):
        """Recursively walk nested lists or tuples to figure out the shape"""
        if isinstance(x, (list, tuple)):
            return (len(x),) + Tensor._infer_shape(x[0]) if x else (0,)
        return ()

    @staticmethod
    def _nest(flat: list, dims: list[int]):
        """Reconstruct nested lists from flat data according to dims."""
        # print(f"dims : {dims}")
        if not dims:
            return flat.pop(0)
        return [Tensor._nest(flat, dims[1:]) for _ in range(dims[0])]

    def _buffer_to_nested(self):
        """Convert flat buffer (memoryview) to nested lists according to shape."""

        if self.dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            flat_data = formatted_fp16_buffer(self._buffer)
        else:
            flat_data = list(self._buffer)

        return Tensor._nest(flat_data, list(self.shape))

    @classmethod
    def ones(cls, shape, dtype: DType = dtypes.fp32, device="cpu", requires_grad=None):
        """Create a tensor filled with ones"""
        size = 1
        for dim in shape:
            size *= dim
        dtype_obj = to_dtype(dtype)

        if dtype_obj.fmt is None:
            raise TypeError(f"Unsupported dtype {dtype_obj.name} for Tensor data")

        flat_data = [1] * size

        t = cls.__new__(cls)
        t.dtype = dtype_obj
        t.device = device
        t.requires_grad = requires_grad
        t.grad = None
        t.shape = tuple(shape)

        try:
            if dtype_obj.fmt == "e" and not ARRAY_E_SUPPORTED:
                raw = bytearray(struct.pack(f"<{size}{dtype_obj.fmt}", *flat_data))
                t._raw = raw
                t._buffer = memoryview(raw).cast("H")  # unsigned short, 16‑bit
            elif dtype_obj.fmt != "e":
                raw = bytearray(struct.pack(f"<{size}{dtype_obj.fmt}", *flat_data))
                t._raw = raw
                t._buffer = memoryview(raw).cast(dtype_obj.fmt)
            else:
                arr = array.array(dtype_obj.fmt, flat_data)
                t._raw = arr
                t._buffer = memoryview(arr)
            return t
        except struct.error as e:
            raise TypeError(f"Could not pack data with format '{dtype_obj.fmt}' for ones: {e}")
        except TypeError as e:
            raise TypeError(
                f"Could not create array.array with format '{dtype_obj.fmt}' for ones: {e}"
            )

    def __repr__(self):
        # _buffer_to_nested will now handle the correct interpretation for fp16 fallback
        nested = self._buffer_to_nested()
        return (
            f"Tensor(shape={self.shape}, data={nested}, "
            f"device={self.device}, dtype={self.dtype.name}, "
            f"requires_grad={self.requires_grad})"
        )

    @classmethod
    def zeros(cls, shape, dtype: DType = dtypes.fp32, device="cpu", requires_grad=None):
        """Create a tensor filled with zeros"""
        size = 1
        for dim in shape:
            size *= dim
        dtype_obj = to_dtype(dtype)

        if dtype_obj.fmt is None:
            raise TypeError(f"Unsupported dtype {dtype_obj.name} for Tensor data")

        flat_data = [0] * size

        t = cls.__new__(cls)
        t.dtype = dtype_obj
        t.device = device
        t.requires_grad = requires_grad
        t.grad = None
        t.shape = tuple(shape)

        try:
            if dtype_obj.fmt == "e" and not ARRAY_E_SUPPORTED:
                raw = bytearray(size * 2)
                t._raw = raw
                t._buffer = memoryview(raw).cast("H")  # unsigned short, 16‑bit
            elif dtype_obj.fmt != "e":
                raw = bytearray(struct.pack(f"<{size}{dtype_obj.fmt}", *flat_data))
                t._raw = raw
                t._buffer = memoryview(raw).cast(dtype_obj.fmt)
            else:
                arr = array.array(dtype_obj.fmt, flat_data)
                t._raw = arr
                t._buffer = memoryview(arr)
            return t
        except struct.error as e:
            raise TypeError(f"Could not pack data with format '{dtype_obj.fmt}' for ones: {e}")
        except TypeError as e:
            raise TypeError(
                f"Could not create array.array with format '{dtype_obj.fmt}' for ones: {e}"
            )
