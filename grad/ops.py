from __future__ import annotations

import operator
from functools import lru_cache


@lru_cache(maxsize=1)
def get_tensor_class():
    from grad.tensor import Tensor

    return Tensor


class Ops:
    @staticmethod
    def _to_tensor(x):
        from grad.tensor import Tensor

        if not isinstance(x, Tensor):
            return Tensor(x)
        return x

    @staticmethod
    def add(a, b):
        a = Ops._to_tensor(a)
        b = Ops._to_tensor(b)
        return BinaryOps.add(a, b)

    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError("Forward must be implemented in child classes")

    @staticmethod
    def backward(*args, **kwargs):
        raise NotImplementedError("Backward must be implemented in child classes")

    @staticmethod
    def apply():
        pass


class BinaryOps(Ops):
    import operator

    @staticmethod
    def _elementwise_op(a, b, op_func):
        Tensor = get_tensor_class()
        result_shape = a.shape

        result = Tensor.zeros(result_shape, dtype=a.dtype)
        if a.shape == b.shape and a._contiguous and b._contiguous:
            # Use itertools to apply operation directly on flattened buffers
            for i, (x, y) in enumerate(zip(a.storage, b.storage)):
                result.storage[i] = op_func(x, y)

        return result

    @staticmethod
    def add(a, b):
        return BinaryOps._elementwise_op(a, b, operator.add)

    @staticmethod
    def sub(a, b):
        return BinaryOps._elementwise_op(a, b, operator.add)


class UnaryOps(Ops):
    pass
