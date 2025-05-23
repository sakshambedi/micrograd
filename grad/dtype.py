from __future__ import annotations

import array
from dataclasses import dataclass
from typing import Any, Final, Literal, Union

from grad.utils.fp16 import float16_to_uint16, uint16_to_float16

_ARRAY_E_SUPPORTED = "e" in array.typecodes

FmtStr = Literal["?", "b", "B", "h", "H", "i", "I", "q", "Q", "e", "f", "d"]


class DTypeMetaClass(type):
    dcache: dict[tuple, DType] = {}

    def __call__(self, *args, **kwds):
        if ret := DTypeMetaClass.dcache.get(args, None):
            return ret

        DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
        return ret


@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
    priority: int
    itemsize: int
    name: str
    fmt: FmtStr | None
    count: int
    _scalar: DType | None

    @staticmethod
    def new(priority: int, itemsize: int, name: str, fmt: FmtStr | None):
        return DType(priority, itemsize, name, fmt, 1, None)


class dtypes:
    void: Final[DType] = DType.new(-1, 0, "void", None)
    bool: Final[DType] = DType.new(0, 1, "bool", "?")
    int8: Final[DType] = DType.new(1, 1, "signed char", "b")
    uint8: Final[DType] = DType.new(2, 1, "unsigned char", "B")
    int16: Final[DType] = DType.new(3, 2, "short", "h")
    uint16: Final[DType] = DType.new(4, 2, "unsigned short", "H")
    int32: Final[DType] = DType.new(5, 4, "int", "i")
    uint32: Final[DType] = DType.new(6, 4, "unsigned int", "I")
    int64: Final[DType] = DType.new(7, 8, "long", "q")
    uint64: Final[DType] = DType.new(8, 8, "unsigned long", "Q")

    # Floating point
    float16: Final[DType] = DType.new(9, 2, "float16", "e")
    float32: Final[DType] = DType.new(10, 4, "float32", "f")
    float64: Final[DType] = DType.new(11, 8, "double", "d")

    # alias
    fp16: Final[DType] = float16
    fp32: Final[DType] = float32
    double: Final[DType] = float64

    @staticmethod
    def _upcast(t1: DType, t2: DType) -> DType:
        if t1 and t2:
            return t1 if t1.priority >= t2.priority else t2
        raise TypeError(f"Cannot upcast dtypes {t1.name} and {t2.name}")

    @classmethod
    def _storage_format(cls, dtype: DType) -> str:
        """Return the actual format code used to store data in the buffer."""
        if dtype.fmt == "e" and not _ARRAY_E_SUPPORTED:
            return "H"  # store fp16 as uint16
        elif dtype.fmt == "?":
            return "b"  # store bools as signed char
        elif dtype.fmt is None:
            raise TypeError(f"Unsupported dtype {dtype.name} (no format string)")
        return dtype.fmt

    @classmethod
    def _to_storage(cls, val: Any, dtype: DType) -> Any:
        """Convert val (python scalar) to the representation expected by storage.
        Keeps numeric types fast for the common cases, only struct‑packs for fp16.
        """
        if dtype.fmt == "e" and not _ARRAY_E_SUPPORTED:
            # Use manual conversion when struct 'e' is not supported
            return float16_to_uint16(float(val))
        elif dtype.fmt == "?":
            return 1 if bool(val) else 0
        return val

    @classmethod
    def _from_storage(cls, stored: Any, dtype: DType) -> Any:
        """Inverse of _to_storage, read a python value from raw buffer item."""
        if dtype.fmt == "e" and not _ARRAY_E_SUPPORTED:
            return uint16_to_float16(stored)
        elif dtype.fmt == "?":
            return bool(stored)
        return stored


DTypeLike = Union[str, DType]


def to_dtype(dtype: DTypeLike) -> DType:
    return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())
