from collections.abc import Iterable
from typing import Any

from grad.device import Device
from grad.dtype import DType, DTypeLike, to_dtype
from grad.kernels import cpu_kernel  # type: ignore


class Buffer:
    __slots__ = ("dtype", "_storage", "_fmt", "numelem")

    def __init__(self, dtype: DTypeLike, iterable: list[Any], *, copy: bool = True):
        self.dtype: DType = to_dtype(dtype)
        self._storage = cpu_kernel.Buffer(iterable, self.dtype.fmt)

    # --- Shared storage helpers -------------------------------------------------
    def share(self) -> "Buffer":
        """Return a new :class:`Buffer` instance that shares the underlying
        storage with ``self``.  The returned object is a lightweight wrapper
        around the same C++ buffer so mutations on either instance will be
        visible to the other."""

        new = self.__class__.__new__(self.__class__)
        new.dtype = self.dtype
        new._storage = self._storage
        return new

    def shares_storage_with(self, other: "Buffer") -> bool:
        """Return ``True`` if ``self`` and ``other`` refer to the same
        underlying storage."""

        return self._storage is getattr(other, "_storage", None)

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

    def __del__(self): ...

    @staticmethod
    def _make_buffer(type_: DTypeLike, iterable: Iterable[Any]) -> memoryview: ...

    def clone(self) -> "Buffer":
        """Create a copy of this buffer"""
        copied = self.to_list()
        return Buffer(self.dtype, copied)

    def resize(self, new_size: int) -> "Buffer":
        """Create a new buffer with different size, preserving existing data where possible"""
        current = self.to_list()
        if new_size > len(current):
            current.extend([0] * (new_size - len(current)))
        else:
            current = current[:new_size]
        return Buffer(self.dtype, current)

    @classmethod
    def _filled(
        cls: type["Buffer"], dtype: DTypeLike, num_elem: int, val: int | float
    ) -> "Buffer":
        inst: "Buffer" = cls.__new__(cls)
        inst.dtype = to_dtype(dtype)
        inst._storage = cpu_kernel.Buffer([val for _ in range(num_elem)], inst.dtype.fmt)
        return inst

    def __getitem__(self, idx: int):
        return self._storage[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._storage[idx] = value

    def size_bytes(self) -> int:
        """Return the size of this buffer in bytes"""
        return len(self._storage) * self._storage.itemsize if self._storage else 0
