import numpy as np
import pytest

from grad.tensor import Tensor


class TestAddition:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([1, 2, 3], [0.5, 1.0, 1.5]),
            ([1.5, 2.5, 3.5], [1, 2, 3]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            ([[0, 1, 2], [3, 4, 5]], [[1, 1, 1], [1, 1, 1]]),
            ([-1, -2, -3], [1, 2, 3]),
            ([1.5, 2.5, 3.5], [0.5, 1.0, 1.5]),
        ],
    )
    def test_add_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a + t_b
        res = np.array(a) + np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_add_2d(self):
        t0 = Tensor([[1, 2, 3], [4, 5, 6]])
        t1 = Tensor([[10, 20, 30], [40, 50, 60]])
        out = t0 + t1
        expected = np.array([[11, 22, 33], [44, 55, 66]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    # def test_broadcast_add(self):
    #     t0 = Tensor.ones((3, 1))
    #     t1 = Tensor.arange(3)
    #     out = t0 + t1
    #     expected = np.ones((3, 1)) + np.arange(3)
    #     np.testing.assert_array_equal(out.to_numpy(), expected)

    # def test_broadcast_add_mismatched(self):
    #     t0 = Tensor.ones((2, 3))
    #     t1 = Tensor.arange(3)
    #     out = t0 + t1
    #     expected = np.ones((2, 3)) + np.arange(3)
    #     np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_add_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 + t1


class TestSubtraction:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([4, 5, 6], [1, 2, 3]),
            ([[10, 20], [30, 40]], [[1, 2], [3, 4]]),
            ([[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]]),
            ([5, -6], [-1, -2]),
            ([2.5, 3.5], [0.5, 1.0]),
        ],
    )
    def test_sub_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a - t_b
        res = np.array(a) - np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_sub_2d(self):
        t0 = Tensor([[10, 20, 30], [40, 50, 60]])
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        out = t0 - t1
        expected = np.array([[9, 18, 27], [36, 45, 54]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_sub_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 - t1


class TestMultiplication:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            ([[0, 1, 2], [3, 4, 5]], [[1, 1, 1], [1, 1, 1]]),
            ([-1, -2], [5, -6]),
            ([1.5, 2.5], [0.5, 1.0]),
            ([2, 3], [0.5, 1.5]),
        ],
    )
    def test_mul_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a * t_b
        res = np.array(a) * np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_mul_2d(self):
        t0 = Tensor([[1, 2, 3], [4, 5, 6]])
        t1 = Tensor([[10, 20, 30], [40, 50, 60]])
        out = t0 * t1
        expected = np.array([[10, 40, 90], [160, 250, 360]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_mul_with_zeros(self):
        t0 = Tensor([0, 1, 2])
        t1 = Tensor([5, 0, 3])
        out = t0 * t1
        expected = np.array([0, 0, 6])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_mul_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 * t1


class TestDivision:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([4, 6, 8], [2, 2, 2]),
            ([[10, 20], [30, 40]], [[2, 4], [5, 8]]),
            ([[6, 8, 10], [12, 15, 18]], [[2, 2, 2], [3, 3, 3]]),
            ([1.5, 2.0], [0.5, 1.0]),
        ],
    )
    def test_div_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a / t_b
        res = np.array(a) / np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_div_2d(self):
        t0 = Tensor([[10, 20, 30], [40, 50, 60]])
        t1 = Tensor([[2, 4, 5], [8, 10, 12]])
        out = t0 / t1
        expected = np.array([[5.0, 5.0, 6.0], [5.0, 5.0, 5.0]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_div_by_zero(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([1, 0, 3])
        out = t0 / t1
        # According to ops.py, division by zero returns inf
        expected = np.array([1.0, float("inf"), 1.0])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_div_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 / t1


class TestPower:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([2, 3, 4], [2, 2, 2]),
            ([[2, 3], [4, 5]], [[2, 3], [1, 2]]),
            ([1, 2, 3], [0, 1, 2]),
            ([2.0, 3.0], [0.5, 2.0]),
        ],
    )
    def test_pow_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a**t_b
        res = np.array(a) ** np.array(b)
        # Use almost_equal for floating point comparisons
        if any(isinstance(x, float) for x in a) or any(isinstance(x, float) for x in b):
            np.testing.assert_array_almost_equal(t_c.to_numpy(), res, decimal=6)
        else:
            np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_pow_2d(self):
        t0 = Tensor([[2, 3, 4], [5, 6, 7]])
        t1 = Tensor([[2, 2, 2], [1, 3, 2]])
        out = t0**t1
        expected = np.array([[4, 9, 16], [5, 216, 49]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_pow_special_cases(self):
        # Test with base 0 and positive exponent
        t0 = Tensor([0, 1, 2])
        t1 = Tensor([2, 0, 3])
        out = t0**t1
        expected = np.array([0, 1, 8])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_pow_negative_base(self):
        t0 = Tensor([-2, -3])
        t1 = Tensor([2, 3])
        out = t0**t1
        expected = np.array([4, -27])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    # no support for broadcast right now
    # def test_pow_shape_mismatch(self):
    #     t0 = Tensor([1, 2, 3])
    #     t1 = Tensor([[1, 2], [3, 4]])
    #     with pytest.raises(
    #         ValueError, match="operands could not be broadcast together with shapes"
    #     ):
    #         _ = t0**t1


class TestNegation:
    @pytest.mark.parametrize(
        "a",
        [
            [1, 2, 3],
            [[1, 2], [3, 4]],
            [0, -1, 2],
            [-1.5, 2.5, -3.0],
            [[[-1, 2], [3, -4]], [[5, -6], [-7, 8]]],
        ],
    )
    def test_neg_matches_numpy(self, a):
        t_a = Tensor(a)
        t_neg = -t_a
        expected = -np.array(a)
        np.testing.assert_array_equal(t_neg.to_numpy(), expected)

    def test_neg_1d(self):
        t = Tensor([1, -2, 3, -4])
        result = -t
        expected = np.array([-1, 2, -3, 4])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_2d(self):
        t = Tensor([[1, -2, 3], [-4, 5, -6]])
        result = -t
        expected = np.array([[-1, 2, -3], [4, -5, 6]])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_3d(self):
        t = Tensor([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]])
        result = -t
        expected = np.array([[[-1, 2], [-3, 4]], [[5, -6], [7, -8]]])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_zeros(self):
        t = Tensor([0, 0.0, -0])
        result = -t
        expected = np.array([0, -0.0, 0])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_float_precision(self):
        t = Tensor([1.23456789, -9.87654321])
        result = -t
        expected = np.array([-1.23456789, 9.87654321])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_double_negation(self):
        t = Tensor([1, -2, 3])
        result = -(-t)
        expected = np.array([1, -2, 3])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_large_values(self):
        t = Tensor([1e6, -1e6, 1e-6, -1e-6])
        result = -t
        expected = np.array([-1e6, 1e6, -1e-6, 1e-6])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=10)
