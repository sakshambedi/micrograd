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
        a_memview, b_memview = a.storage._storage, b.storage._storage
        result_memview = result.storage._storage

        for i in range(prod(a.shape)):
            # py_val_a = dtypes._from_storage(a_memview[i], a.dtype)
            # py_val_b = dtypes._from_storage(b_memview[i], b.dtype)
            result_memview[i] = op(a_memview[i], b_memview[i])

    else:
        for idx in _nd_indices(a.shape):
            result[idx] = op(a[idx], b[idx])

    return result


def _unary_op(a: Tensor, op: Callable[[Any], Any]) -> Tensor:
    """Generic unary operation implementation."""

    result = Tensor.zeros(a.shape, dtype=a.dtype, device=a.device)

    if a._contiguous:
        assert a.storage is not None and result.storage is not None
        a_memview = a.storage._storage
        result_memview = result.storage._storage

        for i in range(prod(a.shape)):
            val_stored = a_memview[i]
            result_memview[i] = op(val_stored)  # dtypes._to_storage(result_py, a.dtype)
    else:
        for idx in _nd_indices(a.shape):
            result[idx] = op(a[idx])

    return result


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        ctx.save_for_backward(a, b)
        from grad.kernels import cpu_kernel  # type: ignore

        return cpu_kernel.add(a.storage, b._storage, "float32")
        # _elementwise_op(a, b, lambda x, y: x + y)

    @staticmethod
    def backward(ctx: Function, *grad_output: Any) -> Any:
        # For addition, L = a + b ;  dL/da = grad_output, dL/db = grad_output
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return _elementwise_op(a, b, lambda x, y: x - y)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # subtraction, dL/da =
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
        return _elementwise_op(a, b, lambda x, y: x**y)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Neg(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor) -> Tensor:
        """Element-wise negation."""
        ctx.save_for_backward(a)
        return _unary_op(a, lambda x: -x)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass
