from __future__ import annotations

from typing import Callable, Dict

from grad.buffer import Buffer
from grad.dtype import DTypeLike, DType, dtypes, to_dtype
from grad.kernels import cpu_kernel  # type: ignore


class EwOp:
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    SUB = "sub"


_DISPATCH_TABLE: Dict[
    str, Callable[[cpu_kernel.Buffer, cpu_kernel.Buffer, str], cpu_kernel.Buffer]
] = {
    EwOp.ADD: cpu_kernel.add,
    EwOp.MUL: cpu_kernel.Buffer.mul if hasattr(cpu_kernel.Buffer, "mul") else None,  # type: ignore
    EwOp.DIV: cpu_kernel.Buffer.div if hasattr(cpu_kernel.Buffer, "div") else None,  # type: ignore
}


def elementwise_op(a: Buffer, b: Buffer, op: str, out_dtype: DTypeLike | None = None) -> Buffer:
    if len(a) != len(b):
        raise ValueError(f"Size mismatch: {len(a)} vs {len(b)}")

    if out_dtype is None:
        out_d: DType = dtypes._upcast(a.dtype, b.dtype)
    else:
        out_d = to_dtype(out_dtype)

    fn = _DISPATCH_TABLE.get(op)
    if fn is None:
        raise ValueError(f"Unsupported op '{op}'")

    result_storage = fn(a._storage, b._storage, out_d.name)  # type: ignore[arg-type]

    result = Buffer.__new__(Buffer)
    result.dtype = out_d
    result._storage = result_storage
    return result


def elementwise_add(a: Buffer, b: Buffer, out_dtype: DTypeLike | None = None) -> Buffer:
    return elementwise_op(a, b, EwOp.ADD, out_dtype)
