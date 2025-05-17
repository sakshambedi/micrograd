from __future__ import annotations

import array
import operator
from collections.abc import Generator, Iterable, Sequence

# Local Caching at Module Level 3-4x speedup in tight loops (microbenchmark)
from math import prod as _prod
from typing import Any

from grad.dtype import DType, DTypeLike, dtypes, to_dtype
from grad.utils.fp16 import float16_to_uint16, formatted_fp16_buffer, uint16_to_float16

ARRAY_E_SUPPORTED = "e" in array.typecodes


def _storage_format(dtype: DType) -> str:
    """Return the actual format code used to store data in the buffer."""
    if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
        return "H"  # store fp16 as uint16
    if dtype.fmt == "?":
        return "b"  # store bools as signed char
    if dtype.fmt is None:
        raise TypeError(f"Unsupported dtype {dtype.name} (no format string)")
    return dtype.fmt


def _to_storage(val: Any, dtype: DType) -> Any:
    """Convert val (python scalar) to the representation expected by storage.
    Keeps numeric types fast for the common cases, only struct‑packs for fp16.
    """
    if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
        # Use manual conversion when struct 'e' is not supported
        return float16_to_uint16(float(val))
    if dtype.fmt == "?":
        return 1 if bool(val) else 0
    return val


def _from_storage(stored: Any, dtype: DType) -> Any:
    """Inverse of _to_storage, read a python value from raw buffer item."""
    if dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
        # Use manual conversion when struct \'e\' is not supported
        return uint16_to_float16(stored)
    if dtype.fmt == "?":
        return bool(stored)
    return stored


class Tensor:
    """Tiny, NumPy‑like dense tensor backed by a contiguous buffer."""

    __slots__ = ("dtype", "shape", "device", "requires_grad", "grad", "_buffer")

    def __init__(
        self,
        data: Iterable | int | float | None = None,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: bool | None = None,
    ) -> None:
        self.dtype: DType = to_dtype(dtype)
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None

        if data is None:
            self.shape = ()
            self._buffer = self._make_buffer([_to_storage(0, self.dtype)])
        elif isinstance(data, (list, tuple)):
            self.shape = self._infer_shape(data)
            self._buffer = self._list_like_to_buffer(data)
        else:
            self.shape = ()
            self._buffer = self._make_buffer([_to_storage(data, self.dtype)])

    @classmethod
    def zeros(cls, shape: Sequence[int], **kw) -> Tensor:
        return cls._filled(shape, 0, **kw)

    @classmethod
    def ones(cls, shape: Sequence[int], **kw) -> Tensor:
        return cls._filled(shape, 1, **kw)

    @staticmethod
    def _infer_shape(seq: Sequence) -> tuple[int, ...]:
        if not isinstance(seq, (list, tuple)):
            return ()
        if not seq:
            return (0,)
        inner = Tensor._infer_shape(seq[0])
        if any(Tensor._infer_shape(s) != inner for s in seq[1:]):
            raise IndexError("Inconsistent tensor shape")
        return (len(seq),) + inner

    @staticmethod
    def _flatten_gen(x: Any) -> Generator:
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from Tensor._flatten_gen(item)
        else:
            yield x

    def _list_like_to_buffer(self, nested: Sequence) -> memoryview:
        """Stream‑flatten nested into an array.array"""
        storage_fmt = _storage_format(self.dtype)
        arr = array.array(storage_fmt)
        arr.extend(_to_storage(v, self.dtype) for v in self._flatten_gen(nested))
        return memoryview(arr)

    def _make_buffer(self, iterable: Iterable[Any]) -> memoryview:
        storage_fmt = _storage_format(self.dtype)
        arr = array.array(storage_fmt, iterable)
        return memoryview(arr)

    @staticmethod
    def _nest(flat: list[Any], dims: list[int]) -> Any:
        if not dims:
            return flat.pop(0)
        return [Tensor._nest(flat, dims[1:]) for _ in range(dims[0])]

    def _to_nested(self) -> Any:
        if _prod(self.shape) == 0:
            flat: list[Any] = []
        elif self.dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
            flat = formatted_fp16_buffer(self._buffer)
        else:
            flat = list(self._buffer)
        return self._nest(flat, list(self.shape))

    @staticmethod
    def _broadcast(t: Tensor, shape, fmt) -> Tensor:
        if t.shape == shape:
            return t

        # Broadcasting a scalar tensor to the target shape.
        if t.shape == ():
            scalar_val = _from_storage(t._buffer[0], t.dtype)
            storage_fmt = _storage_format(t.dtype)
            numel = _prod(shape)

            arr = array.array(storage_fmt, [_to_storage(scalar_val, fmt)] * numel)
            new_tensor = Tensor.__new__(Tensor)
            new_tensor.dtype = fmt
            new_tensor.shape = shape
            new_tensor.device = t.device
            new_tensor.requires_grad = t.requires_grad
            new_tensor.grad = None
            new_tensor._buffer = memoryview(arr)
            return new_tensor

        raise ValueError(f"Cannot broadcast tensor with shape {t.shape} to {shape}")

    def _binary_op(self, other: Tensor | int | float | bool | list | tuple, op):
        if not isinstance(other, Tensor):
            if isinstance(other, (list, tuple)):
                other = Tensor(other, dtype=self.dtype, device=self.device)
            elif isinstance(other, (float, int, bool)):
                other = Tensor([other] * _prod(self.shape), dtype=self.dtype, device=self.device)
                other.shape = self.shape

        _upcaste_dtype = dtypes._upcast(self.dtype, other.dtype)
        if self.shape != other.shape:
            if self.shape == ():
                self = Tensor._broadcast(self, other.shape, _upcaste_dtype)
            elif other.shape == ():
                other = Tensor._broadcast(other, self.shape, _upcaste_dtype)
            else:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        out = Tensor.__new__(Tensor)
        out.dtype = _upcaste_dtype
        out.shape = self.shape
        out.device = self.device
        out.requires_grad = (
            (self.requires_grad or False) or (other.requires_grad or False)
        ) or None
        out.grad = None

        storage_fmt = _storage_format(out.dtype)
        buf = array.array(storage_fmt)
        append = buf.append  # local fast name

        for a, b in zip(self._buffer, other._buffer):
            aval = _from_storage(a, self.dtype)
            bval = _from_storage(b, other.dtype)

            # Simplified division handling relies on Python's float behavior for inf/nan
            append(_to_storage(op(aval, bval), out.dtype))

        out._buffer = memoryview(buf)
        return out

    __add__ = lambda self, other: self._binary_op(other, operator.add)
    __mul__ = lambda self, other: self._binary_op(other, operator.mul)
    __rmul__ = lambda self, other: self._binary_op(other, operator.mul)
    __sub__ = lambda self, other: self._binary_op(other, operator.sub)
    __rsub__ = lambda self, other: self._binary_op(other, operator.sub)
    __truediv__ = lambda self, other: self._binary_op(other, operator.truediv)
    __rtruediv__ = lambda self, other: self._binary_op(other, lambda a, b: operator.truediv(b, a))
    __floordiv__ = lambda self, other: self._binary_op(other, operator.floordiv)

    @classmethod
    def _filled(
        cls,
        shape: Sequence[int],
        value: Any,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: bool | None = None,
    ) -> Tensor:
        inst = cls.__new__(cls)
        inst.dtype = to_dtype(dtype)
        inst.shape = tuple(shape)
        inst.device = device
        inst.requires_grad = requires_grad
        inst.grad = None

        storage_val = _to_storage(value, inst.dtype)
        numel = _prod(shape)
        inst._buffer = inst._make_buffer([storage_val] * numel)
        return inst

    def to_numpy(self):
        import numpy as np

        return np.array(self._to_nested())

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype='{self.dtype.name}', "
            f"device='{self.device}', requires_grad={self.requires_grad}, "
            f"data={self._to_nested()})"
        )

    def __str__(self) -> str:
        nested = self._to_nested()
        return str(nested if self.shape else nested)

    @staticmethod
    def _contiguous_index(shape: list, index: list[int]) -> int:
        """
        Compute the flat index for a tensor of given shape at multi‐dimensional index.
        """
        if len(shape) != len(index):
            raise IndexError(f"Dimensionality mismatch: shape={shape}, index={index}")

        flat_index = 0
        stride = 1

        for dim, idx in zip(reversed(shape), reversed(index)):
            if idx < 0 or idx >= dim:
                raise IndexError(f"Index out of bounds: {idx} not in [0, {dim})")
            flat_index += idx * stride
            stride *= dim

        return flat_index

    def __getitem__(self, idx: list[int] | tuple[int, ...] | int) -> Any:
        if self.shape == ():
            raise IndexError(f"Cannot index a constant Tensor: ({self.__repr__()})")

        if not isinstance(idx, (list, tuple, int)):
            raise IndexError(f"Invalid indexing argument for the tensor with shape({self.shape})")

        if isinstance(idx, int):
            if len(self.shape) == 1:
                # Normalize negative indices
                if idx < 0:
                    idx += self.shape[0]
                if idx < 0 or idx >= self.shape[0]:
                    raise IndexError(
                        f"Trying to access an element outside the bounds of the Tensor with length : {self.shape[0]}"
                    )
                return _from_storage(self._buffer[idx], self.dtype)
            else:
                raise NotImplementedError(
                    "Integer indexing of multi-dimensional tensors not implemented"
                )

        elif isinstance(idx, (list, tuple)):
            t_shape = self.shape
            if len(idx) == len(t_shape):
                return _from_storage(
                    self._buffer[Tensor._contiguous_index(list(self.shape), list(idx))],
                    self.dtype,
                )
            else:
                raise IndexError(
                    f"Dimensionality mismatch: tensor has {len(t_shape)} dimensions but {len(idx)} indices were provided"
                )
