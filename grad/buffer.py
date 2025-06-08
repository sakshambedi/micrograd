from collections.abc import Iterable
from typing import Any

from grad.device import Device
from grad.dtype import DType, DTypeLike, to_dtype
from grad.kernels import cpu_kernel  # type: ignore


class Buffer:
    __slots__ = ("dtype", "_storage")

    def __init__(self, dtype: DTypeLike, iterable: list[Any], *, copy: bool = True):
        self.dtype: DType = to_dtype(dtype)
        print(f"buffer: {self.dtype} , {self.dtype.fmt}")
        self._storage = cpu_kernel.Buffer(iterable, self.dtype.fmt)

    def to(self, device: Device): ...  # noqa: E704

    def __len__(self):
        return self._storage.size()

    def __repr__(self) -> str:
        return str(self.to_list())

    def iterstorage(self) -> Iterable[Any]:
        for i in range(self.__len__()):
            yield self._storage[i]

    def to_list(self) -> list[Any]:
        return [val for val in self.iterstorage()]

    def __del__(self): ...  # noqa : E704

    @staticmethod
    def _make_buffer(type_: DTypeLike, iterable: Iterable[Any]) -> memoryview: ...  # noqa : E704

    def clone(self) -> "Buffer":
        """Create a copy of this buffer"""
        ...

    def resize(self, new_size: int) -> "Buffer":
        """Create a new buffer with different size, preserving existing data where possible"""
        ...

    @classmethod
    def _filled(cls, dtype: DTypeLike, num_elem: int, val: int | float) -> "Buffer":
        buff = cls.__new__(cls)
        buff.dtype = out_dtype = to_dtype(dtype)
        buff._storage = cpu_kernel.Buffer(num_elem, out_dtype.fmt, val)
        return buff

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Buffer index {idx} out of range")
        return self._storage[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._storage[idx] = value

    def size_bytes(self) -> int:
        """Return the size of this buffer in bytes"""
        ...
