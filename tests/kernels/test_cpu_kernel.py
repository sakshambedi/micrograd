import pytest
from grad.kernels import cpu_kernel


def test_buffer_size():
    buf = cpu_kernel.Buffer("float32", 5)
    assert buf.size() == 5


def test_set_and_get():
    buf = cpu_kernel.Buffer("float32", 1)
    buf[0] = 42.0
    assert buf[0] == pytest.approx(42.0)


def test_get_dtype():
    buf = cpu_kernel.Buffer("float32", 1)
    assert buf.get_dtype() == "float32"


def test_invalid_dtype():
    with pytest.raises(RuntimeError):
        cpu_kernel.Buffer("invalid", 1)
