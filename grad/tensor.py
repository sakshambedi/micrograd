from __future__ import annotations

import array
from collections.abc import Generator, Iterable, Sequence
from math import prod as _prod
from typing import Any, Optional, TypeVar, Union

from grad.buffer import Buffer
from grad.dtype import DType, DTypeLike, dtypes
from grad.utils.fp16 import formatted_fp16_buffer
from grad.utils.misc import tensor_stride

ARRAY_E_SUPPORTED = "e" in array.typecodes
T = TypeVar("T", bound="Tensor")


class Tensor:
    """Tiny, NumPy‑like dense tensor backed by a contiguous buffer."""

    __slots__ = ("shape", "device", "requires_grad", "grad", "storage", "_ctx", "_prev", "_stride")

    def __init__(
        self,
        data: Union[Iterable, int, float, None] = None,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: Optional[bool] = None,
    ) -> None:
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self.storage: Optional[Buffer] = None
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

        self._stride: tuple[int, ...] = tensor_stride(self.shape)

    # ---- Factory Methods ----

    @classmethod
    def zeros(cls, shape: Sequence[int], **kw) -> Tensor:
        """Create a tensor filled with zeros."""
        return cls._filled(shape, 0, **kw)

    @classmethod
    def ones(cls, shape: Sequence[int], **kw) -> Tensor:
        """Create a tensor filled with ones."""
        return cls._filled(shape, 1, **kw)

    @classmethod
    def full(cls, shape: Sequence[int], fill_value: Any, **kw) -> Tensor:
        """Create a tensor filled with the specified value."""
        return cls._filled(shape, fill_value, **kw)

    @classmethod
    def _filled(
        cls,
        shape: Sequence[int],
        value: Any,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: Optional[bool] = None,
    ) -> Tensor:
        """Internal method for creating tensors filled with a value."""
        inst = cls.__new__(cls)
        inst.storage = Buffer._filled(dtype, _prod(shape), value)
        inst.shape = tuple(shape)
        inst._stride = tensor_stride(inst.shape)
        inst.device = device
        inst.requires_grad = requires_grad
        inst.grad = None
        inst._ctx = None
        inst._prev = None
        return inst

    # ---- Shape Manipulation Methods ----

    def view(self, *shape: int) -> Tensor:
        """Return a tensor with the same data but a different shape."""
        shape_tuple = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        new_size, old_size = _prod(shape_tuple), _prod(self.shape)

        if new_size != old_size:
            raise ValueError(
                f"Cannot view tensor of shape {self.shape} with {old_size} elements as shape {shape_tuple} with {new_size} elements"
            )

        return self._create_view(shape_tuple)

    def reshape(self, *shape: int) -> Tensor:
        """Alias for view method."""
        return self.view(*shape)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Swap dimensions dim0 and dim1 of the tensor."""
        if dim0 == dim1:
            return self

        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

        new_stride = list(self._stride)
        new_stride[dim0], new_stride[dim1] = new_stride[dim1], new_stride[dim0]

        return self._create_view(tuple(new_shape), stride=tuple(new_stride))

    @staticmethod
    def T(ten: Tensor) -> Tensor:
        """Transpose the tensor"""
        if len(ten.shape) <= 1:
            return ten
        elif len(ten.shape) == 2:
            return ten.transpose(0, 1)
        raise BufferError(
            f"Input tensor with shape({ten.shape}) has len: ({len(ten.shape)})>= 2 for transpose not supported"
        )

    def permute(self): ...  # NOQA: E704

    def _create_view(
        self, shape: tuple[int, ...], *, stride: tuple[int, ...] | None = None
    ) -> Tensor:
        """Create a new tensor that shares storage with self but has a different shape."""
        result = Tensor.__new__(Tensor)
        result.shape = shape
        result._stride = tensor_stride(shape) if stride is None else stride
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result._ctx = None
        result._prev = None
        result.storage = self.storage
        return result

    # ---- Properties and Accessors ----

    @property
    def dtype(self) -> DType:
        """Return the data type of the tensor."""
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage.dtype

    @property
    def buffer(self) -> memoryview:
        """Return a memoryview of the underlying storage."""
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage._storage

    def buffer_id(self) -> int:
        """Returns the memory address of the underlying storage."""
        if self.storage is None:
            return 0
        return id(self.storage._storage)

    def stride(self, dim: Optional[int] = None) -> Union[tuple[int, ...], int]:
        """Return the stride of the tensor. If dim is specified, return the stride for that dimension."""
        return self._stride if dim is None else self._stride[dim % len(self.shape)]

    # ---- Data Access Methods ----

    def __getitem__(self, index):
        """Access tensor data by index."""
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape):
            raise IndexError("Wrong number of indices")
        if self.storage is not None:
            offset = sum(i * s for i, s in zip(index, self._stride))

            return self.storage[offset]
        raise AttributeError("Tensor with a storage has not been initialized yet!")

    # ---- Internal Helper Methods ----

    @staticmethod
    def _infer_shape(seq: Sequence) -> tuple[int, ...]:
        """Infer the shape of a nested sequence."""
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
        """Flatten a nested sequence into a generator."""
        stack = [x]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                for item in reversed(current):
                    stack.append(item)
            else:
                yield current

    # ---- String Representation Methods ----

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        nested = self._to_nested()
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype.name},"
            f"device={self.device}, requires_grad={self.requires_grad}, "
            f"data={nested})"
        )

    def __str__(self) -> str:
        """Return a string representation of the tensor data."""
        nested = self._to_nested()
        return str(nested)

    def _to_nested(self) -> Any:
        """Convert the flat buffer to a nested list structure matching the tensor's shape."""
        if _prod(self.shape) == 0:
            flat: list[Any] = []
        else:
            if self.storage is None:
                raise AttributeError("Tensor with data is not initialized yet!")
            if self.dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
                flat = formatted_fp16_buffer(self.storage._storage)  # convert FP16 -> Python float
            else:
                flat = self.storage.to_list()
        return self._nest(flat, list(self.shape))

    @staticmethod
    def _nest(flat: list[Any], dims: list[int]) -> Any:
        """Recursively nest a flat list according to the provided dimensions."""
        if not dims:
            return flat.pop(0)
        return [Tensor._nest(flat, dims[1:]) for _ in range(dims[0])]
