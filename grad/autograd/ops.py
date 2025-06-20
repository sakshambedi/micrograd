from __future__ import annotations

from typing import Any

from grad.autograd.function import Function
from grad.buffer import Buffer
from grad.dtype import dtypes
from grad.kernels import cpu_kernel  # type: ignore
from grad.tensor import Tensor
from grad.utils.misc import tensor_stride


# from grad.utils.misc import _nd_indices
def _elementwise_forward(
    ctx, a: Tensor, b: Tensor, kernel_func: Any, op_name: str | None = "elementwise"
):
    ctx.save_for_backward(a, b)

    if a.storage is None or b.storage is None:
        raise ValueError(f"Cannot perform {op_name} on tensors with no storage")

    rdtype = dtypes._upcast(a.dtype, b.dtype)
    cpp_result_buffer = kernel_func(
        a.storage._storage,
        b.storage._storage,
        rdtype.name,
    )

    result = Tensor.__new__(Tensor)
    result.shape = a.shape  # TODO: add broadcasting support
    result._stride = tensor_stride(result.shape)
    result.device = (a.device or b.device) or "cpu"
    result._contiguous = True
    result.base_offset = 0
    result.storage = Buffer._from_cpp_buffer(cpp_result_buffer, rdtype)
    result.grad, result.grad_fn, result.requires_grad = None, None, None

    return result


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        return _elementwise_forward(ctx, a, b, cpu_kernel.add, "Addition")

    @staticmethod
    def backward(ctx: Function, *grad_output: Any) -> Any:
        # For addition, L = a + b ;  dL/da = grad_output, dL/db = grad_output
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return _elementwise_forward(ctx, a, b, cpu_kernel.sub, "Addition")

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # subtraction, dL/da =
        pass


class Mul(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        # return _elementwise_op(a, b, lambda x, y: x * y)
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Div(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise division of two tensors."""
        # return _elementwise_op(a, b, lambda x, y: x / y if y != 0 else float("inf"))
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Pow(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise power operation."""
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Neg(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor) -> Tensor:
        """Element-wise negation."""
        # ctx.save_for_backward(a)
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass
