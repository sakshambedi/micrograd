from __future__ import annotations

from typing import Any

from grad.autograd.function import Function
from grad.buffer import Buffer
from grad.dtype import dtypes
from grad.tensor import Tensor
from grad.utils.misc import tensor_stride

# from grad.utils.misc import _nd_indices


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        ctx.save_for_backward(a, b)

        from grad.kernels import cpu_kernel  # type: ignore

        if a.storage is None or b.storage is None:
            raise ValueError("Cannot perform addition on tensors with no storage")

        # Previously this created a zero-filled tensor and then overwrote the
        # buffer with the kernel result. That unnecessary allocation has been
        # removed to reduce overhead.
        rdtype = dtypes._upcast(a.dtype, b.dtype)
        cpp_result_buffer = cpu_kernel.add(
            a.storage._storage,
            b.storage._storage,
            rdtype.name,
        )

        result = Tensor.__new__(Tensor)
        result.shape = a.shape
        result._stride = tensor_stride(result.shape)
        result.device = (a.device or b.device) or "cpu"
        result._contiguous = True
        result.base_offset = 0
        result.storage = Buffer._from_cpp_buffer(cpp_result_buffer, rdtype)
        result.grad = None
        result.grad_fn = None
        result.requires_grad = None

        return result

    @staticmethod
    def backward(ctx: Function, *grad_output: Any) -> Any:
        # For addition, L = a + b ;  dL/da = grad_output, dL/db = grad_output
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        # return _elementwise_op(a, b, lambda x, y: x - y)
        ...

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
