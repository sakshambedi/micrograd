from __future__ import annotations

import array
from collections.abc import Generator, Iterable, Sequence
from math import prod as _prod
from typing import Any, Optional, overload

from grad.autograd.function import Function
from grad.buffer import Buffer
from grad.dtype import DType, DTypeLike, dtypes
from grad.utils.fp16 import formatted_fp16_buffer, uint16_to_float16
from grad.utils.misc import _nd_indices, tensor_stride

ARRAY_E_SUPPORTED = "e" in array.typecodes


class Tensor:
    """Tiny, PyTorch‑like dense tensor backed by a contiguous buffer."""

    __slots__ = (
        "shape",
        "device",
        "requires_grad",
        "grad",
        "storage",
        "grad_fn",
        "_stride",
        "_contiguous",
        "base_offset",
    )

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
        self.grad: Optional[Tensor] = None
        self.storage: Optional[Buffer] = None
        self.grad_fn: Optional[Function] = None
        self._contiguous: bool = True
        self.base_offset: int = 0

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

    @classmethod
    def zeros(cls, shape: Sequence[int], **kw) -> Tensor:
        """Create a tensor filled with zeros."""
        return cls._filled(shape, 0, **kw)

    @classmethod
    def ones(cls, shape: Sequence[int], **kw) -> Tensor:
        """Create a tensor filled with ones."""
        return cls._filled(shape, 1, **kw)

    @classmethod
    def arange(cls, range: int, **kw) -> Tensor:
        """Pytroch arange like function."""
        ...

    @classmethod
    def randn(cls, range: int, **kw) -> Tensor: ...

    @classmethod
    def full(cls, shape: Sequence[int], fill_value: Any, **kw) -> Tensor:
        """Create a tensor filled with the specified value."""
        return cls._filled(shape, fill_value, **kw)

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

    @staticmethod
    def permute(ten: Tensor, *idx: int) -> Tensor:
        """Permute the tensor. Read more: https://docs.pytorch.org/docs/stable/generated/torch.permute.html"""
        idx_tup: tuple[int, ...] = idx[0] if len(idx) == 1 and isinstance(idx[0], tuple) else idx
        if len(idx) != len(ten.shape):
            raise ValueError(
                f"Number of permutation indices ({len(idx)}) must match tensor dimensions ({len(ten.shape)})"
            )
        if len(set(idx)) != len(idx):
            raise ValueError(f"Permutation indices contain duplicates: {idx}")
        if sorted(idx) != list(range(len(ten.shape))):
            raise ValueError(
                f"Invalid permutation indices: {idx}. Must be a permutation of {list(range(len(ten.shape)))}"
            )

        # apply permutation
        shape_n = [ten.shape[d] for d in idx_tup]
        stride_n = [ten.stride(d) for d in idx_tup]
        return ten._create_view(tuple(shape_n), stride=tuple(stride_n))

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

    @overload
    def stride(self) -> tuple[int, ...]: ...  # noqa : E704

    @overload
    def stride(self, dim: int) -> int: ...  # noqa : E704

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        """Return the stride of the tensor. If dim is specified, return the stride for that dimension."""
        return self._stride if dim is None else self._stride[dim % len(self.shape)]

    @staticmethod
    def matmul(t1: Tensor, t2: Tensor, /, dtype: dtypes | None = None): ...  # noqa : E704

    # ---- Default override fuctions ----

    def __add__(self, other):
        from grad.autograd.ops import Add

        return Add.apply(self, other)

    def __sub__(self, other):
        from grad.autograd.ops import Sub

        return Sub.apply(self, other)

    def __mul__(self, other):
        from grad.autograd.ops import Mul

        return Mul.apply(self, other)

    def __truediv__(self, other):
        from grad.autograd.ops import Div

        return Div.apply(self, other)

    def __pow__(self, other):
        from grad.autograd.ops import Pow

        return Pow.apply(self, other)

    def __neg__(self):
        from grad.autograd.ops import Neg

        return Neg.apply(self)

    def _offset(self, index):
        return self.base_offset + sum(i * s for i, s in zip(index, self._stride))

    def __getitem__(self, index):
        """Access tensor data by index."""
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape):
            raise IndexError("Wrong number of indices")
        if self.storage is not None:
            offsetval = self._offset(index)
            val = self.storage[offsetval]
            if self.dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
                val = uint16_to_float16(val)

            return val

        raise AttributeError("Tensor with a storage has not been initialized yet!")

    def to_numpy(self):
        import numpy as np

        size = int(np.prod(self.shape))
        arr = np.array(self.buffer[:size], dtype=self.dtype.fmt)
        return arr.reshape(self.shape)

    def __setitem__(self, idx, value):
        """Standard function for setting values by indexing"""
        if not isinstance(idx, tuple):
            idx = (idx,)  # incase of 1d tensor
        if len(idx) != len(self.shape):
            raise IndexError(
                f"Indexing a tensor with incorrect dimension. Tensor with shape: ({self.shape})"
            )
        if self.storage is None:
            raise AttributeError("Tensor with a storage has not been initialized yet!")

        offsetval = self._offset(index=idx)
        self.storage[offsetval] = value

    def __repr__(self) -> str:
        """Return a string representation of the tensor"""
        nested = self._to_nested()
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype.name},"
            f"device={self.device}, requires_grad={self.requires_grad}, "
            f"data={nested}), contiguous={self._contiguous}"
        )

    def __str__(self) -> str:
        """Return a string representation of the tensor data."""
        nested = self._to_nested()
        return str(nested)

    # ---- Internal Helper Methods ----
    @classmethod
    def _filled(
        cls: type[Tensor],
        shape: Sequence[int],
        value: Any,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: Optional[bool] = None,
    ) -> Tensor:
        """Internal method for creating tensors filled with a value."""
        inst: Tensor = cls.__new__(cls)
        inst.storage = Buffer._filled(dtype, _prod(shape), value)
        inst.shape = tuple(shape)
        inst._stride = tensor_stride(inst.shape)
        inst.device = device
        inst.requires_grad = requires_grad
        inst.grad = None
        inst.grad_fn = None
        inst._contiguous = True
        inst.base_offset = 0
        return inst

    def _create_view(
        self,
        shape: tuple[int, ...],
        *,
        stride: tuple[int, ...] | None = None,
        base_offset: int | None = None,
    ) -> Tensor:
        """Create a new tensor that shares storage with self but has a different shape."""
        result = Tensor.__new__(Tensor)
        result.shape = shape
        result._stride = tensor_stride(shape) if stride is None else stride
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad = None
        result.grad_fn = None
        result.storage = self.storage
        result._contiguous = False
        result.base_offset = 0 if base_offset is None else base_offset
        return result

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

    @staticmethod
    def _contiguous_tensor(t: Tensor) -> Tensor:
        """Make the cheap view/permuate to a buffer on device"""
        if t._contiguous:
            return t

        out = Tensor.zeros(t.shape, dtype=t.dtype, device=t.device, requires_grad=t.requires_grad)
        for idx in _nd_indices(t.shape):
            out[idx] = t[idx]

        return out

    def _to_nested(self) -> Any:
        """Convert the flat buffer to a nested list structure matching the tensor's shape."""
        if _prod(self.shape) == 0:
            if not self.shape:
                return [] if self.storage and _prod(self.shape) == 0 else None

            return [self._nest([], list(self.shape[1:])) for _ in range(self.shape[0])]

        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")

        if self._contiguous:
            if self.dtype.fmt == "e" and not ARRAY_E_SUPPORTED:
                flat = formatted_fp16_buffer(self.storage._storage)  # FP16 -> py float
            else:
                flat = self.storage.to_list()
            return self._nest(flat, list(self.shape))

        flat_ordered_data = []
        for idx in _nd_indices(self.shape):
            flat_ordered_data.append(self.__getitem__(idx))

        return self._nest(flat_ordered_data, list(self.shape))

    @staticmethod
    def _nest(flat: list[Any], dims: list[int]) -> Any:
        """Recursively nest a flat list according to the provided dimensions."""
        if not dims:
            return flat.pop(0) if flat else None
        return [Tensor._nest(flat, dims[1:]) for _ in range(dims[0])]
