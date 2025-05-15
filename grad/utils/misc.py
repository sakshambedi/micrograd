from __future__ import annotations

import struct
from typing import Any


def get_val_from_tensor_buffer(tensor: Any, index: int) -> Any:
    item = tensor._buffer[index]
    if tensor.dtype.fmt == "e":
        return struct.unpack("<e", struct.pack("<H", item))[0]
    elif tensor.dtype.fmt == "?":
        return bool(item)
    return item
