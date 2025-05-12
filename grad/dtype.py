from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Optional, Union

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
    priority: int  # this determines when things get upcasted
    itemsize: int
    name: str
    fmt: FmtStr | None
    count: int
    _scalar: DType | None

    @staticmethod
    def new(priority: int, itemsize: int, name: str, fmt: Optional[FmtStr]):
        return DType(priority, itemsize, name, fmt, 1, None)


class dtypes:
    void: Final[DType] = DType.new(-1, 0, "void", None)
    bool: Final[DType] = DType.new(10, 1, "bool", "?")
    int8: Final[DType] = DType.new(1, 1, "signed char", "b")
    float32: Final[DType] = DType.new(200, 4, "float32", "f")
    fp32: Final[DType] = float32
    float16: Final[DType] = DType.new(180, 2, "float16", "e")
    fp16: Final[DType] = float16
    int32: Final[DType] = DType.new(100, 4, "int32", "i")

    def __init__(self):
        pass


DTypeLike = Union[str, DType]


def to_dtype(dtype: DTypeLike) -> DType:
    return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())
