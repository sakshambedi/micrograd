from __future__ import annotations

import threading
from typing import Any

# from types import TracebackType

__all__ = ["Function", "grad_enabled"]

autograd_state = threading.local()
setattr(autograd_state, "enabled", True)


def grad_enabled() -> bool:
    """Return True if autograd recording is currently enabled."""
    return getattr(autograd_state, "enabled", True)


def _tensor_cls():
    from grad.tensor import Tensor

    return Tensor


class Function:
    _version: int

    def __init__(self):
        self._saved_tensors: tuple[Any, ...] = ()
        self.needs_input_grad = []
        self.next_functions = []
        self._version = 0

    @staticmethod
    def forward(ctx: Function, a: Any, b: Any) -> Any:
        """Compute forward pass. Must be overridden."""
        raise NotImplementedError("Forward must be implemented in child classes")

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        """Compute gradients of inputs w.r.t. grad_outputs. Must be overridden."""
        raise NotImplementedError("Backward must be implemented in child classes")

    @classmethod
    def apply(cls, *inputs: Any, **kwargs: Any):
        ctx = cls()
        output = cls.forward(ctx, *inputs, **kwargs)
        return output
