import pytest

from grad.kernels import cpu_kernel


class TestBufferConstructors:
    """Tests comparing both Buffer constructors."""

    def test_size_comparison(self):
        buf1 = cpu_kernel.Buffer("float32", 5)
        buf2 = cpu_kernel.Buffer([0.0, 1.0, 2.0, 3.0, 4.0], "f")

        assert buf1.size() == 5
        assert buf2.size() == 5

    def test_dtype_comparison(self):
        dtypes = [
            ("float32", "f"),
            ("int32", "i"),
            ("bool", "?"),
            ("float64", "d"),
            ("int64", "q"),
            ("uint8", "B"),
        ]

        for old_dtype, new_fmt in dtypes:
            buf1 = cpu_kernel.Buffer(old_dtype, 1)
            buf2 = cpu_kernel.Buffer([0], new_fmt)
            assert buf1.get_dtype() == buf2.get_dtype() == old_dtype

    def test_set_get_comparison(self):
        buf1 = cpu_kernel.Buffer("float32", 1)
        buf2 = cpu_kernel.Buffer([0.0], "f")

        buf1[0] = 42.0
        buf2[0] = 42.0

        assert buf1[0] == buf2[0] == pytest.approx(42.0)

    def test_buffer_size(self):
        buf = cpu_kernel.Buffer([0.0, 1.0, 2.0, 3.0, 4.0], "f")
        assert buf.size() == 5

    def test_set_and_get(self):
        buf = cpu_kernel.Buffer([0.0], "f")
        buf[0] = 42.0
        assert buf[0] == pytest.approx(42.0)

    def test_get_dtype(self):
        buf = cpu_kernel.Buffer([0.0], "f")
        assert buf.get_dtype() == "float32"

    def test_get_dtype_int32(self):
        buf = cpu_kernel.Buffer([0], "i")
        assert buf.get_dtype() == "int32"

    def test_get_dtype_bool(self):
        buf = cpu_kernel.Buffer([False], "?")
        assert buf.get_dtype() == "bool"

    def test_invalid_dtype(self):
        with pytest.raises(RuntimeError):
            cpu_kernel.Buffer([0], "invalid")

        with pytest.raises(RuntimeError):
            cpu_kernel.Buffer("invalid", 1)

    def test_buffer_from_sequence(self):
        values = [1.1, 2.2, 3.3, 4.4]
        buf = cpu_kernel.Buffer(values, "f")
        for i, v in enumerate(values):
            assert buf[i] == pytest.approx(v)

    def test_buffer_from_sequence_int(self):
        values = [1, 2, 3, 4]
        buf = cpu_kernel.Buffer(values, "i")
        for i, v in enumerate(values):
            assert buf[i] == v

    def test_buffer_from_sequence_bool(self):
        values = [True, False, True, False]
        buf = cpu_kernel.Buffer(values, "?")
        for i, v in enumerate(values):
            assert buf[i] == v

    def test_buffer_mixed_types(self):
        values = [1, 2, 3.3, 4]
        buf = cpu_kernel.Buffer(values, "f")
        assert buf.get_dtype() == "float32"
        assert buf[2] == pytest.approx(3.3)

    def test_old_constructor_initialization(self):
        buf = cpu_kernel.Buffer("float32", 5)
        for i in range(5):
            assert buf[i] == pytest.approx(0.0)

        buf_int = cpu_kernel.Buffer("int32", 3)
        for i in range(3):
            assert buf_int[i] == 0

        buf_bool = cpu_kernel.Buffer("bool", 2)
        for i in range(2):
            assert buf_bool[i] == False

    def test_equivalent_results(self):
        size = 10
        values = [float(i) for i in range(size)]

        buf1 = cpu_kernel.Buffer("float32", size)
        buf2 = cpu_kernel.Buffer(values, "f")

        for i, v in enumerate(values):
            buf1[i] = v

        for i in range(size):
            assert buf1[i] == buf2[i]
