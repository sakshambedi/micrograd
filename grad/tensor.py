from __future__ import annotations

import array
from collections.abc import Generator, Iterable, Sequence
from math import prod as _prod
from typing import Any

from grad.buffer import Buffer
from grad.dtype import DType, DTypeLike, dtypes
from grad.utils.fp16 import formatted_fp16_buffer

ARRAY_E_SUPPORTED = "e" in array.typecodes


class Tensor:
    """Tiny, NumPy‑like dense tensor backed by a contiguous buffer."""

    __slots__ = ("shape", "device", "requires_grad", "grad", "storage", "_ctx", "_prev", "_stride")

    def __init__(
        self,
        data: Iterable | int | float | None = None,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: bool | None = None,
    ) -> None:
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self.storage: Buffer | None = None
        self._ctx = None
        self._prev = None

        if data is None:
            self.shape: tuple[int, ...] = ()
            self.storage = Buffer(dtype, [0])
        elif isinstance(data, (list, tuple)):
            self.shape: tuple[int, ...] = self._infer_shape(data)
            self.storage = Buffer(dtype, self._flatten_gen(data))
        else:
            self.shape: tuple[int, ...] = ()
            self.storage = Buffer(dtype, [data])

        self._stride: tuple[int, ...] = self._calc_stride()

    @classmethod
    def zeros(cls, shape: Sequence[int], **kw) -> Tensor:
        return cls._filled(shape, 0, **kw)

    @classmethod
    def ones(cls, shape: Sequence[int], **kw) -> Tensor:
        return cls._filled(shape, 1, **kw)

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
        inst.storage = Buffer._filled(dtype, _prod(shape), value)
        inst.shape = tuple(shape)
        inst.device = device
        inst.requires_grad = requires_grad
        inst.grad = None
        return inst

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        return self._stride[dim] if dim else self._stride

    @property
    def dtype(self) -> DType:
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage.dtype

    @property
    def buffer(self) -> memoryview:
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage._storage

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
        stack = [x]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                for item in reversed(current):
                    stack.append(item)
            else:
                yield current

    def __repr__(self) -> str:
        nested = self._to_nested()
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype.name},"
            f"device={self.device}, requires_grad={self.requires_grad}, "
            f"data={nested})"
        )

    def __str__(self) -> str:
        nested = self._to_nested()
        return str(nested)

    @staticmethod
    def _nest(flat: list[Any], dims: list[int]) -> Any:
        if not dims:
            return flat.pop(0)
        return [Tensor._nest(flat, dims[1:]) for _ in range(dims[0])]

    def _to_nested(self) -> Any:
        if _prod(self.shape) == 0:
            flat: list[Any] = []
        else:
            if self.storage is None:
                raise AttributeError("Tensor with data is not initialized yet!")
            if self.dtype.fmt == "e" and ARRAY_E_SUPPORTED:
                flat = formatted_fp16_buffer(self.storage._storage)  # convert FP16 -> Python float
            else:
                flat = self.storage.to_list()
        return self._nest(flat, list(self.shape))

    def _calc_stride(self) -> tuple[int, ...]:
        """
        O(len(shape)) time, O(1) extra memory.
        pytorch stride : https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html
        """
        if not self.shape:
            return ()

        acc = 1
        stride = []
        for dim in reversed(self.shape):
            stride.append(acc)
            acc *= dim
        stride.reverse()
        return tuple()

    def buffer_id(self) -> int:
        """Returns the memory address of the underlying storage.
        Read more : https://www.w3schools.com/python/ref_func_id.asp"""
        if self.storage is None:
            return 0
        return id(self.storage._storage)

    def view(self, *shape: int) -> Tensor:
        shape_tuple = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        new_size, old_size = _prod(shape_tuple), _prod(self.shape)

        if new_size != old_size:
            raise ValueError(
                f"Cannot view tensor of shape {self.shape} with {old_size} elements as shape {shape_tuple} with {new_size} elements"
            )

        result = Tensor.__new__(Tensor)
        result.shape = tuple(shape_tuple)
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result._ctx = None
        result._prev = None
        result.storage = self.storage
        return result
