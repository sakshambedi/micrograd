from __future__ import annotations

from typing import Any

from grad.autograd import operations  # type:ignore
from grad.autograd.function import Function
from grad.buffer import Buffer
from grad.dtype import dtypes
from grad.tensor import Tensor
from grad.utils.misc import broadcast_shape, tensor_stride


def _elementwise_forward(ctx, a: Tensor, b: Tensor, op_type: operations.BinaryOpType):
    if a.storage is None or b.storage is None:
        raise ValueError(f"Cannot perform {op_type} on tensors with no storage")

    ctx.save_for_backward(a, b)
    rdtype = dtypes._upcast(a.dtype, b.dtype)
    out_shape = broadcast_shape(a.shape, b.shape)
    # cpp_result_buffer = operations.buffer_add(a.storage._storage, b.storage._storage, rdtype.name)
    cpp_result_buffer = operations.binary_op(
        a.storage._storage, b.storage._storage, op_type, rdtype.name
    )
    result = Tensor.__new__(Tensor)
    result.shape = tuple(out_shape)
    result._stride = tensor_stride(result.shape)
    result.device = (a.device or b.device) or "cpu"
    result._contiguous = a._contiguous and b._contiguous
    result.base_offset = 0
    result.storage = Buffer._from_cpp_buffer(cpp_result_buffer, rdtype)
    result.grad, result.grad_fn, result.requires_grad = None, None, None

    return result


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        return _elementwise_forward(ctx, a, b, operations.BinaryOpType.ADD)

    @staticmethod
    def backward(ctx: Function, *grad_output: Any) -> Any:
        # For addition, L = a + b ;  dL/da = grad_output, dL/db = grad_output
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return _elementwise_forward(ctx, a, b, operations.BinaryOpType.SUB)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # For subtraction, L = a - b; dL/da = grad_output, dL/db = -grad_output
        grad_output = grad_outputs[0]
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        return _elementwise_forward(ctx, a, b, operations.BinaryOpType.MUL)
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Div(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise division of two tensors."""
        return _elementwise_forward(ctx, a, b, operations.BinaryOpType.DIV)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        pass


class Pow(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise power operation."""
        import numpy as np

        ctx.save_for_backward(a, b)

        # Convert tensors to numpy arrays for power operation
        a_np = a.to_numpy()

        # If b is a tensor, use its numpy representation, otherwise use the scalar
        if isinstance(b, Tensor):
            b_np = b.to_numpy()
        else:
            b_np = b

        # Perform the power operation
        result_np = np.power(a_np, b_np)

        # Create a new tensor with the result
        result = Tensor(result_np.tolist(), dtype=a.dtype)
        result.requires_grad = a.requires_grad
        return result

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # Power rule: if y = x^n, then dy/dx = n*x^(n-1)
        # Not implementing the full gradient for now
        pass


class Neg(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor) -> Tensor:
        """Element-wise negation."""
        if a.storage is None:
            raise ValueError("Cannot perform neg on tensors with no storage")
        ctx.save_for_backward(a)
        rdtype = a.dtype
        cpp_result_buffer = operations.unary_op(
            a.storage._storage, operations.UnaryOpType.NEG, rdtype.name
        )
        result = Tensor.__new__(Tensor)
        result.shape = a.shape
        result._stride = a._stride
        result.device = a.device
        result._contiguous = a._contiguous
        result.base_offset = a.base_offset
        result.storage = Buffer._from_cpp_buffer(cpp_result_buffer, rdtype)
        result.grad, result.grad_fn, result.requires_grad = None, None, None

        return result

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # For negation: L = -a, dL/da = -grad_output
        grad_output = grad_outputs[0]
        return (-grad_output,)
