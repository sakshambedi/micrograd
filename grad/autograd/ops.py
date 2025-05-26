from __future__ import annotations

from math import prod
from typing import Any, Callable

from grad.autograd.function import Function
from grad.dtype import dtypes
from grad.tensor import Tensor
from grad.utils.misc import _nd_indices


def _elementwise_op(a: Tensor, b: Tensor, op: Callable[[Any, Any], Any]) -> Tensor:
    """Generic element-wise operation implementation."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    rtype = dtypes._upcast(a.dtype, b.dtype)
    result = Tensor.zeros(a.shape, dtype=rtype, device=a.device)

    if a._contiguous and b._contiguous:
        assert a.storage is not None and b.storage is not None and result.storage is not None
        a_memview = a.storage._storage
        b_memview = b.storage._storage
        result_memview = result.storage._storage

        for i in range(prod(a.shape)):
            val_a_stored = a_memview[i]
            val_b_stored = b_memview[i]

            py_val_a = dtypes._from_storage(val_a_stored, a.dtype)
            py_val_b = dtypes._from_storage(val_b_stored, b.dtype)

            result_py = op(py_val_a, py_val_b)

            result_memview[i] = dtypes._to_storage(result_py, rtype)

    else:
        for idx in _nd_indices(a.shape):
            result[idx] = op(a[idx], b[idx])

    return result


def _unary_op(a: Tensor, op: Callable[[Any], Any]) -> Tensor:
    """Generic unary operation implementation."""

    result = Tensor.zeros(a.shape, dtype=a.dtype, device=a.device)

    if a._contiguous:
        # Fast path: direct memory access for contiguous tensors
        assert a.storage is not None and result.storage is not None
        a_memview = a.storage._storage
        result_memview = result.storage._storage

        for i in range(prod(a.shape)):
            val_stored = a_memview[i]
            py_val = dtypes._from_storage(val_stored, a.dtype)
            result_py = op(py_val)
            result_memview[i] = dtypes._to_storage(result_py, a.dtype)
    else:
        for idx in _nd_indices(a.shape):
            result[idx] = op(a[idx])

    return result


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        return _elementwise_op(
            a,
            b,
            lambda x, y: x + y,
        )

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return _elementwise_op(
            a,
            b,
            lambda x, y: x - y,
        )

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Mul(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        return _elementwise_op(a, b, lambda x, y: x * y)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Div(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise division of two tensors."""
        return _elementwise_op(a, b, lambda x, y: x / y if y != 0 else float("inf"))

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Pow(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise power operation."""
        return _elementwise_op(
            a,
            b,
            lambda x, y: x**y,
        )

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


# class Neg(Function):
#     @staticmethod
#     def forward(ctx: Function, a: Tensor) -> Tensor:
#         """Element-wise negation."""
#         return _unary_op(a, lambda x: -x)

#     @staticmethod
#     def backward(ctx: Function, *grad_outputs: Any) -> Any:
#         pass
