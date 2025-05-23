from __future__ import annotations


class Ops:
    @staticmethod
    def _to_tensor(x) -> "Tensor":  # type: ignore # noqa : F821
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
    @staticmethod
    def add(a, b):
        print("a : ", a)
        print("b : ", b)


class UnaryOps(Ops):
    pass
