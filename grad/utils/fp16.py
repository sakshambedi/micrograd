import math
import struct
from typing import Iterable

# ---------- feature probe ----------
try:
    struct.pack("e", 0.0)  # is the half‑float code available?
    _STRUCT_E_OK = True
except struct.error:
    _STRUCT_E_OK = False


# ---------- uint16 -> float ----------
def uint16_to_float16(word: int) -> float:
    """
    Convert one 16‑bit integer that encodes a binary16 number
    into a Python float (binary64).
    """
    if not (0 <= word < 1 << 16):
        raise ValueError("word must fit in 16 bits")
    if _STRUCT_E_OK:  # fast path when supported
        return struct.unpack("<e", struct.pack("<H", word))[0]

    # manual decode (IEEE‑754 binary16)
    sign = (word >> 15) & 0x1
    exp = (word >> 10) & 0x1F
    frac = word & 0x3FF

    if exp == 0:  # subnormal or zero
        if frac == 0:
            return -0.0 if sign else 0.0
        return (-1) ** sign * 2 ** (-14) * (frac / 2**10)
    if exp == 0x1F:  # inf or nan
        if frac == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")

    # normal number
    return (-1) ** sign * 2 ** (exp - 15) * (1 + frac / 2**10)


def words_to_floats(words: Iterable[int]) -> list[float]:
    return [uint16_to_float16(w) for w in words]


# ---------- float -> uint16 ----------
def float16_to_uint16(value: float) -> int:
    """
    Round a Python float to binary16 and return the 16‑bit word.
    """
    if _STRUCT_E_OK:  # fast path
        return struct.unpack("<H", struct.pack("<e", value))[0]

    # manual encode (round to nearest even)
    if math.isnan(value):
        return 0x7E00  # canonical NaN
    if math.isinf(value):
        return 0xFC00 if value < 0 else 0x7C00
    sign = 0
    if value < 0:
        sign = 1
        value = -value
    if value == 0.0:
        return sign << 15

    mant, exp2 = math.frexp(value)  # value = mant * 2 ** exp2 , 0.5 <= mant < 1
    exp2 -= 1  # shift to 1 <= mant < 2
    mant *= 2

    exp16 = exp2 + 15
    if exp16 >= 0x1F:  # overflow -> inf
        return (sign << 15) | 0x7C00
    if exp16 <= 0:  # subnormal or underflow
        mant = mant * (2 ** (exp16 - 1))
        frac = int(round(mant * 2**10))
        return (sign << 15) | frac

    frac = int(round((mant - 1) * 2**10))
    if frac == 0x400:  # handle rounding overflow
        exp16 += 1
        frac = 0
        if exp16 >= 0x1F:
            return (sign << 15) | 0x7C00
    return (sign << 15) | (exp16 << 10) | (frac & 0x3FF)


def floats_to_words(values: Iterable[float]) -> list[int]:
    return [float16_to_uint16(v) for v in values]


# ---------- convenience pretty‑printer ----------
def formatted_fp16_buffer(buf: memoryview) -> list[float]:
    """
    Expect a memoryview whose format is 'H' (uint16).
    Returns a comma‑separated string of fp16 numbers.
    """
    if buf.format != "H":
        raise TypeError("buffer format must be 'H' for uint16 words")
    floats = words_to_floats(buf)
    return floats
