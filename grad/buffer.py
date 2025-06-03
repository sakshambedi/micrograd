from collections.abc import Iterable
from typing import Any

from grad.device import Device
from grad.dtype import DType, DTypeLike, to_dtype
from grad.kernels import cpu_kernel


class Buffer:
    __slots__ = ("dtype", "_storage", "_fmt")

    def __init__(self, dtype: DTypeLike, iterable: list[Any] | tuple[Any]):
        self.dtype: DType = to_dtype(dtype)
        self._storage = cpu_kernel.Buffer(self.dtype.fmt, len(iterable))
        for i, v in enumerate(iterable):
            self._storage[i] = v

    def to(self, device: Device): ...  # noqa: E704

    @property
    def fmt(self):
        return self.dtype.fmt

    def __len__(self):
        return self._storage.size()

    def __repr__(self) -> str:
        rstr = "["
        for idx in range(tlen := self.__len__()):
            rstr += str(self._storage[idx]) + (", " if idx != tlen - 1 else "]")
        return rstr

    def to_list(self) -> list[Any]:
        return [val for val in self._storage]

    def __del__(self): ...

    @staticmethod
    def _make_buffer(type_: DTypeLike, iterable: Iterable[Any]) -> memoryview: ...

    def clone(self) -> "Buffer":
        """Create a copy of this buffer"""
        ...

    def resize(self, new_size: int) -> "Buffer":
        """Create a new buffer with different size, preserving existing data where possible"""
        ...

    @classmethod
    def _filled(
        cls: type["Buffer"], dtype: DTypeLike, num_elem: int, val: int | float
    ) -> "Buffer": ...

    def __getitem__(self, idx: int):
        return self._storage[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._storage[idx] = value

    def size_bytes(self) -> int:
        """Return the size of this buffer in bytes"""
        return len(self._storage) * self._storage.itemsize if self._storage else 0
