import pytest

from grad.buffer import Buffer
from grad.dtype import dtypes, to_dtype


class TestBuffer:
    def test_init_with_list(self):
        b = Buffer([1.0, 2.0, 3.0], dtypes.float32)
        assert len(b) == 3
        assert b._dtype.name == "float32"
        assert b.to_list() == [1.0, 2.0, 3.0]

        b = Buffer([1, 2, 3], dtypes.int32)
        assert len(b) == 3
        assert b._dtype.name == "int"  # internal name is int
        assert b.to_list() == [1, 2, 3]

        # b = Buffer("bool", [True, False, True])
        # assert len(b) == 3
        # assert b._dtype.name == "bool"
        # assert b.to_list() == [True, False, True]

    def test_type_conversion(self):
        b = Buffer([1, 2, 3], "float32")
        assert all(isinstance(val, float) for val in b.iterstorage())

        b = Buffer([1.5, 2.7, 3.9], "int32")
        assert b.to_list() == [1, 2, 3]

        b = Buffer([1.0, 0.0, 0.5], "bool")
        assert b.to_list() == [True, False, True]

    def test_empty_buffer(self):
        b = Buffer([], "float32")
        assert len(b) == 0
        assert b.to_list() == []

    def test_filled_buffer(self):
        b = Buffer._filled(dtypes.float16, 5, 3.14)
        assert len(b) == 5
        assert all(abs(val - 3.14) < 1e-3 for val in b.to_list())

        b = Buffer._filled("float32", 4, 3.12)
        assert len(b) == 4
        assert all(abs(val - 3.12) < 1e-6 for val in b.to_list())

        b = Buffer._filled("int32", 3, 42)
        assert len(b) == 3
        assert all(val == 42 for val in b.iterstorage())

        b = Buffer._filled("bool", 4, 1)
        assert len(b) == 4
        assert all(val is True for val in b.iterstorage())

        b = Buffer._filled("bool", 4, 0)
        assert len(b) == 4
        assert all(val is False for val in b.iterstorage())

    def test_getitem_setitem(self):
        b = Buffer([1.0, 2.0, 3.0], "float32")

        assert b[0] == 1.0
        assert b[1] == 2.0
        assert b[2] == 3.0

        b[1] = 5.0
        assert b[1] == 5.0
        assert b.to_list() == [1.0, 5.0, 3.0]

        with pytest.raises(IndexError):
            b[3]  # Out of bounds

        with pytest.raises(IndexError):
            b[-1]  # Negative index

    def test_repr(self):
        b = Buffer([1, 2, 3], "int32")
        assert repr(b) == str([1, 2, 3])

    def test_iterstorage(self):
        values = [1.0, 2.0, 3.0]
        b = Buffer(values, "float32")

        for i, val in enumerate(b.iterstorage()):
            assert val == values[i]

        assert list(b.iterstorage()) == values

    def test_different_dtypes(self):
        dtypes = [
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
        ]

        for dt in dtypes:
            b = Buffer._filled(dt, 3, 1)
            assert b._dtype == to_dtype(dt)
            assert len(b) == 3

    def test_large_values(self):
        large_int = 2**31 - 1  # Max 32-bit int
        b = Buffer([large_int], "int32")
        assert b[0] == large_int

        # Test with 64-bit integers
        larger_int = 2**63 - 1  # Max 64-bit int
        b = Buffer([larger_int], "int64")
        assert b[0] == larger_int

    def test_edge_cases(self):
        b = Buffer([1, 2.5, 3], "float32")
        assert len(b) == 3
        assert b.to_list()[0] == 1.0
        assert b.to_list()[1] == 2.5
        assert b.to_list()[2] == 3.0

        small_float = 1e-30
        large_float = 1e30
        b = Buffer([small_float, large_float], "float64")
        assert abs((b[0] - small_float) / small_float) < 1e-10
        assert abs((b[1] - large_float) / large_float) < 1e-10

    def test_to_list(self):
        original = [1.1, 2.2, 3.3]
        b = Buffer(original, "float32")
        recovered = b.to_list()

        assert len(recovered) == len(original)
        for orig, rec in zip(original, recovered):
            assert abs(orig - rec) < 1e-5

    def test_share_and_clone(self):
        b = Buffer([1.0, 2.0, 3.0], "float32")
        shared = b.share()
        assert shared.to_list() == [1.0, 2.0, 3.0]
        assert b.shares_storage_with(shared)

        shared[0] = 10.0
        assert b[0] == 10.0

        clone = b.clone()
        assert not clone.shares_storage_with(b)
        clone[1] = -5.0
        assert b[1] != -5.0
