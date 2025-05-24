from __future__ import annotations

# import threading
# from types import TracebackType


class Function:
    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError("Forward must be implemented in child classes")

    @staticmethod
    def backward(*args, **kwargs):
        raise NotImplementedError("Backward must be implemented in child classes")

    @staticmethod
    def apply():
        pass
