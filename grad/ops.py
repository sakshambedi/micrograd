from __future__ import annotations

from enum import Enum, auto


class Ops(Enum):
    BUFFER = auto()
    DEVICE = auto()


# @dataclass
# class UOp:
#     op: Ops
#     dtype: DType  # Store the DType
#     src: tuple[UOp, ...] | None = None
#     arg: any | None = None
#     sz: int | None = None

#     @staticmethod
#     def new_buffer(device: str, size: int, dtype: DType):
#         return UOp(Ops.BUFFER, dtype, (UOp(Ops.DEVICE, dtype, arg=device),), sz=size)
