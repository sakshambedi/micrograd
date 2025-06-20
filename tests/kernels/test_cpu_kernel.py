import pytest

from grad.kernels import cpu_kernel  # type: ignore


class TestBufferConstructors:
    """Tests comparing all Buffer constructors."""

    def test_filled(self):
        buf1 = cpu_kernel.Buffer(5, "float32", 0)
        assert buf1.size() == 5
        for i in range(5):
            assert buf1[i] == 0

        buf1 = cpu_kernel.Buffer(2, "float32", 10)
        assert buf1.size() == 2
        for i in range(2):
            assert buf1[i] == 10

        buf1 = cpu_kernel.Buffer(2, "int32", 0)
        assert buf1.size() == 2
        for i in range(2):
            assert buf1[i] == 0

        buf1 = cpu_kernel.Buffer(2, "int32", 1)
        assert buf1.size() == 2
        for i in range(2):
            assert buf1[i] == 1

        buf1 = cpu_kernel.Buffer(2, "int32", 2)
        assert buf1.size() == 2
        for i in range(2):
            assert buf1[i] == 2

    def test_size_comparison(self):
        buf1 = cpu_kernel.Buffer(5, "float32")
        buf2 = cpu_kernel.Buffer([0.0, 1.0, 2.0, 3.0, 4.0], "float32")

        assert buf1.size() == 5
        assert buf2.size() == 5

    def test_dtype_comparison(self):
        dtypes = ["float32", "int32", "float64", "int64", "uint8"]

        for dtypes in dtypes:
            buf1 = cpu_kernel.Buffer(1, dtypes)
            buf2 = cpu_kernel.Buffer([0], dtypes)
            assert buf1.get_dtype() == buf2.get_dtype() == dtypes

    # def test_set_get_comparison(self):
    #     buf1 = cpu_kernel.Buffer(1, "float32")
    #     buf2 = cpu_kernel.Buffer([0.0], "float32")

    # buf1[0] = 42.0
    # buf2[0] = 42.0

    # assert buf1[0] == buf2[0] == pytest.approx(42.0)

    def test_buffer_size(self):
        buf = cpu_kernel.Buffer([0.0, 1.0, 2.0, 3.0, 4.0], "float32")
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
        # Bool dtype is currently commented out in the implementation
        # Using uint8 as a substitute for boolean values
        buf = cpu_kernel.Buffer([0], "uint8")
        assert buf.get_dtype() == "uint8"

    def test_invalid_dtype(self):
        with pytest.raises(RuntimeError):
            cpu_kernel.Buffer([0], "invalid")

        with pytest.raises(RuntimeError):
            cpu_kernel.Buffer(1, "invalid")

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
        # Bool dtype is currently commented out in the implementation
        # Using uint8 as a substitute for boolean values
        values = [1, 0, 1, 0]
        buf = cpu_kernel.Buffer(values, "uint8")
        for i, v in enumerate(values):
            assert buf[i] == v

    def test_buffer_mixed_types(self):
        values = [1, 2, 3.3, 4]
        buf = cpu_kernel.Buffer(values, "f")
        assert buf.get_dtype() == "float32"
        assert buf[2] == pytest.approx(3.3)

    def test_old_constructor_initialization(self):
        buf = cpu_kernel.Buffer(5, "float32")
        for i in range(5):
            assert buf[i] == pytest.approx(0.0)

        buf_int = cpu_kernel.Buffer(3, "int32")
        for i in range(3):
            assert buf_int[i] == 0

        # Using uint8 as a substitute for boolean values
        buf_bool = cpu_kernel.Buffer(2, "uint8", 0)
        for i in range(2):
            assert buf_bool[i] == 0

    def test_equivalent_results(self):
        size = 10
        values = [float(i) for i in range(size)]

        buf1 = cpu_kernel.Buffer(size, "float32")
        buf2 = cpu_kernel.Buffer(values, "f")

        for i, v in enumerate(values):
            buf1[i] = v

        for i in range(size):
            assert buf1[i] == buf2[i]
