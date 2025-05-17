from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Union

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

    float16: Final[DType] = DType.new(9, 2, "float16", "e")
    float32: Final[DType] = DType.new(10, 4, "float32", "f")
    fp16: Final[DType] = float16
    fp32: Final[DType] = float32
    float64: Final[DType] = DType.new(11, 8, "double", "d")

    @staticmethod
    def _upcast(t1: DType, t2: DType) -> DType:
        if t1 and t2:
            return t1 if t1.priority >= t2.priority else t2
        raise TypeError(f"Cannot upcast dtypes {t1.name} and {t2.name}")


DTypeLike = Union[str, DType]


def to_dtype(dtype: DTypeLike) -> DType:
    return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())
