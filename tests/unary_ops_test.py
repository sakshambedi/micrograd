import numpy as np
import pytest

from grad.tensor import Tensor


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