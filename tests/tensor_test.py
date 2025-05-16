import array
import math

import pytest

from grad.tensor import Tensor, dtypes


def generate_nested_factory_data(shape, value):
    """Helper to generate nested list structure for factory tests."""
    if not shape:
        return value  # Scalar case
    if len(shape) == 1:
        return [value] * shape[0]
    # Recursively build nested list
    inner_shape = shape[1:]
    return [generate_nested_factory_data(inner_shape, value) for _ in range(shape[0])]


def check_tensor_data(tensor: Tensor, expected_nested):
    expected_shape = Tensor._infer_shape(expected_nested)
    assert (
        tensor.shape == expected_shape
    ), f"shape mismatch: expected {expected_shape}, got {tensor.shape}"

    reconstructed = tensor._to_nested()  # new helper in refactor

    def rec(r, e):
        if isinstance(e, (list, tuple)):
            assert isinstance(r, (list, tuple))
            assert len(r) == len(e)
            for rr, ee in zip(r, e):
                rec(rr, ee)
        elif isinstance(e, float):
            assert math.isclose(r, e, rel_tol=1e-3, abs_tol=1e-8)
        else:
            assert r == e

    rec(reconstructed, expected_nested)


class TestTensorInit:
    def test_float32(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = Tensor(data, dtype=dtypes.float32)

        assert t.shape == (2, 2)
        assert t.dtype is dtypes.float32
        assert isinstance(t._buffer.obj, array.array)
        assert t._buffer.format == "f"
        check_tensor_data(t, data)

    def test_int32(self):
        data = [[1, 2], [3, 4]]
        t = Tensor(data, dtype=dtypes.int32)

        assert t.shape == (2, 2)
        assert t.dtype is dtypes.int32
        assert isinstance(t._buffer.obj, array.array)
        assert t._buffer.format == "i"
        check_tensor_data(t, data)

    def test_bool_from_int(self):
        data = [[1, 0], [0, 1]]
        t = Tensor(data, dtype=dtypes.bool)

        assert t.shape == (2, 2)
        assert t.dtype is dtypes.bool
        assert isinstance(t._buffer.obj, array.array)
        assert t._buffer.format == "b"
        check_tensor_data(t, data)

    def test_bool(self):
        data = [[True, False], [False, True]]
        t = Tensor(data, dtype=dtypes.bool)

        assert t.shape == (2, 2)
        assert isinstance(t._buffer.obj, array.array)
        assert t._buffer.format == "b"  # stored as uint8
        assert t.dtype.fmt == "?"
        check_tensor_data(t, data)

    def test_scalar_like(self):
        data = [5.5]
        t = Tensor(data, dtype=dtypes.float32)

        assert t.shape == (1,)
        assert t.dtype is dtypes.float32
        check_tensor_data(t, data)

    def test_1d(self):
        data = [1.0, 2.0, 3.0]
        check_tensor_data(Tensor(data), data)

    def test_3d(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        t = Tensor(data, dtype=dtypes.int32)

        assert t.shape == (2, 2, 2)
        check_tensor_data(t, data)

    def test_inconsistent_shape(self):
        with pytest.raises(IndexError):
            Tensor([[1, 2], [3]], dtype=dtypes.float32)

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            Tensor("hello")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Ones / Zeros / empty-size cases
# --------------------------------------------------------------------------- #
class TestTensorFactories:
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1,)])
    def test_ones_float(self, shape):
        t = Tensor.ones(shape, dtype=dtypes.float32)
        check_tensor_data(t, generate_nested_factory_data(shape, 1.0))

    def test_ones_int(self):
        shape = (1, 4)
        t = Tensor.ones(shape, dtype=dtypes.int32)
        check_tensor_data(t, generate_nested_factory_data(shape, 1))

    def test_zero_size(self):
        t0 = Tensor.ones((0,), dtype=dtypes.float32)
        assert t0._buffer.nbytes == 0

        t2 = Tensor.ones((2, 0), dtype=dtypes.float32)
        assert t2._buffer.nbytes == 0

    def test_zeros(self):
        shape = (3, 2)
        t = Tensor.zeros(shape, dtype=dtypes.float32)
        check_tensor_data(t, generate_nested_factory_data(shape, 0.0))

    def test_zeros_int(self):
        shape = (5,)
        t = Tensor.zeros(shape, dtype=dtypes.int32)
        check_tensor_data(t, generate_nested_factory_data(shape, 0))


class TestFloat16:
    def test_fp16_initialisation(self):
        data = [1.0, 2.5, -3.0, 0.1]
        t = Tensor(data, dtype=dtypes.float16)

        assert t._buffer.format == "H"  # uint16 storage
        assert t._buffer.nbytes == 2 * len(data)
        check_tensor_data(t, data)

    def test_fp16_ones(self):
        shape = (2, 2)
        t = Tensor.ones(shape, dtype=dtypes.float16)
        assert t._buffer.format == "H"
        assert t._buffer.nbytes == 4 * dtypes.float16.itemsize
        expected = generate_nested_factory_data(shape, 1.0)
        check_tensor_data(t, expected)

    def test_fp16_zeros(self):
        shape = (2, 2)
        t = Tensor.zeros(shape, dtype=dtypes.float16)
        assert t._buffer.format == "H"
        assert t._buffer.nbytes == 4 * dtypes.float16.itemsize
        expected = generate_nested_factory_data(shape, 0.0)
        check_tensor_data(t, expected)


def test_flatten_and_shape():
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    flat = list(Tensor._flatten_gen(data))
    assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
    assert Tensor._infer_shape(data) == (2, 2, 2)


def test_repr_contains():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32, device="gpu", requires_grad=True)

    r = repr(t)
    for fragment in (
        "shape=(2, 2)",
        "dtype='float32'",
        "device='gpu'",
        "requires_grad=True",
        "data=[[1.0, 2.0], [3.0, 4.0]]",
    ):
        assert fragment in r


def test_requires_grad_and_grad_defaults():
    t_default = Tensor([1, 2])
    assert t_default.requires_grad is None

    t_true = Tensor([1, 2], requires_grad=True)
    assert t_true.requires_grad is True

    t_false = Tensor([1, 2], requires_grad=False)
    assert t_false.requires_grad is False

    assert t_default.grad is None
