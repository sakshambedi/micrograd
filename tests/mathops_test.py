import warnings

import numpy as np
import pytest

from grad.tensor import Tensor


class TestAddition:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            (np.arange(6).reshape(2, 3), np.ones((2, 3))),
        ],
    )
    def test_add_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a + t_b

        c = t_c.numpy()
        np.testing.assert_array_equal(c, np.array(a) + np.array(b))

    def test_broadcasting(self):
        t0 = Tensor([[1, 2, 3], [4, 5, 6]])
        t1 = Tensor([10, 20, 30])
        out = (t0 + t1).numpy()
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
    def test_zero_size(self, shape):
        t = Tensor.empty(*shape)
        assert t.numpy().size == 0


class TestMultiplication:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            (np.arange(6).reshape(2, 3), np.ones((2, 3))),
            ([-1, -2], [5, -6]),
            ([1.5, 2.5], [0.5, 1.0]),
            ([1, 2], [0.5, 1.5]),
        ],
    )
    def test_mul_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a * t_b

        c = t_c.numpy()
        np.testing.assert_array_equal(c, np.array(a) * np.array(b))

    @pytest.mark.parametrize(
        "arr, scalar",
        [
            ([1, 2, 3], 5),
            ([[1, 2], [3, 4]], 2.5),
            (np.arange(6).reshape(2, 3), -1),
            ([1.5, 2.5], 0),
        ],
    )
    def test_mul_scalar(self, arr, scalar):
        t_arr = Tensor(arr)

        # Test tensor * scalar
        t_result1 = t_arr * scalar
        result1 = t_result1.numpy()
        np.testing.assert_array_equal(result1, np.array(arr) * scalar)

        # Test scalar * tensor (__rmul__)
        t_result2 = scalar * t_arr
        result2 = t_result2.numpy()
        np.testing.assert_array_equal(result2, scalar * np.array(arr))

    def test_mul_broadcasting(self):
        t0 = Tensor([[1, 2, 3], [4, 5, 6]])
        t1 = Tensor([10, 20, 30])
        out = (t0 * t1).numpy()
        expected = np.array([[10, 40, 90], [40, 100, 180]])
        np.testing.assert_array_equal(out, expected)

        t2 = Tensor([[10], [20]])
        t2_scalar = Tensor(2)

        out2 = (t2 * t2_scalar).numpy()
        expected2 = np.array([[20], [40]])
        np.testing.assert_array_equal(out2, expected2)

    @pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
    def test_mul_zero_size(self, shape):
        t0 = Tensor.empty(*shape)
        t1 = Tensor.empty(*shape)
        out = (t0 * t1).numpy()
        expected = np.empty(shape, dtype=t0.dtype) # Numpy default is float64 for empty
        np.testing.assert_array_equal(out, expected)

        # scalar and tensor multiplication
        t2 = Tensor(5)
        # out2 = (t0 * t2).numpy() # This will be an empty array with shape `shape`
        # expected2 = np.empty(shape, dtype=t0.dtype) * 5 # an empty array
        # np.testing.assert_array_equal(out2, expected2)

        # Multiplying a zero-sized tensor with a scalar results in a zero-sized tensor
        out2 = (t0 * t2).numpy()
        expected_shape_dtype = np.broadcast_arrays(np.empty(shape, dtype=t0.dtype), np.array(t2.data, dtype=t0.dtype))[0]
        np.testing.assert_array_equal(out2, np.empty(expected_shape_dtype.shape, dtype=expected_shape_dtype.dtype))

        out3 = (t2 * t0).numpy()
        np.testing.assert_array_equal(out3, np.empty(expected_shape_dtype.shape, dtype=expected_shape_dtype.dtype))



class TestSubtraction:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            (np.arange(6).reshape(2, 3), np.ones((2, 3))),
            ([-1, -2], [5, -6]),
            ([1.5, 2.5], [0.5, 1.0]),
            ([1, 2], [0.5, 1.5]),
        ],
    )
    def test_sub_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a - t_b

        c = t_c.numpy()
        np.testing.assert_array_equal(c, np.array(a) - np.array(b))

    def test_sub_broadcasting(self):
        t0 = Tensor([[10, 20, 30], [40, 50, 60]])
        t1 = Tensor([1, 2, 3])
        out = (t0 - t1).numpy()
        expected = np.array([[9, 18, 27], [39, 48, 57]])
        np.testing.assert_array_equal(out, expected)

        t2 = Tensor([[10], [20]])
        t3 = Tensor(5)
        out2 = (t2 - t3).numpy()
        expected2 = np.array([[5], [15]])
        np.testing.assert_array_equal(out2, expected2)

    @pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
    def test_sub_zero_size(self, shape):
        t0 = Tensor.empty(*shape)

        # Subtracting two zero-sized tensors
        t1 = Tensor.empty(*shape)
        out = (t0 - t1).numpy()
        expected = np.empty(shape, dtype=t0.dtype)
        np.testing.assert_array_equal(out, expected)

        # Subtracting a scalar from a zero-sized tensor
        # t2 = Tensor(5) # This would make t2 a Tensor with data=[5]
        # out2 = (t0 - t2).numpy()
        # # Broadcasting np.empty(shape) - np.array([5]) would error if shapes not compatible
        # # or result in an empty array if broadcasting rules for empty arrays apply.
        # # For empty - scalar_tensor, numpy behavior is empty_array - scalar_value -> empty_array
        # expected2 = np.empty(shape, dtype=t0.dtype) - np.array(5, dtype=t0.dtype)
        # np.testing.assert_array_equal(out2, expected2)

        # Subtracting a scalar value (not Tensor) from a zero-sized tensor
        scalar_val = 5
        out2 = (t0 - Tensor(scalar_val)).numpy() # t0 - Tensor(5)
        expected2_np_behavior = np.empty(shape, dtype=t0.dtype) - np.array(scalar_val, dtype=t0.dtype) # Operates element-wise, results in empty
        np.testing.assert_array_equal(out2, expected2_np_behavior)


        # Subtracting a zero-sized tensor from a scalar value (not Tensor)
        # out3 = (Tensor(scalar_val) - t0).numpy() # Tensor(5) - t0
        # expected3_np_behavior = np.array(scalar_val, dtype=t0.dtype) - np.empty(shape, dtype=t0.dtype)
        # np.testing.assert_array_equal(out3, expected3_np_behavior)

        # Let's test with Tensor(scalar) - zero_size_tensor and scalar_value - zero_size_tensor
        t_scalar = Tensor(5)
        out3 = (t_scalar - t0).numpy()
        # This broadcasts self.data (scalar) with other.data (empty)
        # np.array(5) - np.empty(shape) -> empty array of shape `shape`
        expected_broadcast_shape = np.broadcast_arrays(t_scalar.data, t0.data)[0].shape
        np.testing.assert_array_equal(out3, np.empty(expected_broadcast_shape, dtype=np.promote_types(t_scalar.dtype, t0.dtype)))

class TestDivision:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 2, 1]),
            ([[1, 2], [3, 4]], [[1, 1], [2, 2]]),
            (np.arange(6).reshape(2, 3), np.ones((2, 3))),
        ],
    )
    def test_true_div_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a / t_b
        np.testing.assert_array_equal(t_c.numpy(), np.array(a) / np.array(b))

    @pytest.mark.parametrize(
        "arr, scalar",
        [
            ([10, 20, 30], 10),
            ([[1, 2], [3, 4]], 2.5),
            (np.arange(6).reshape(2, 3), -1),
        ],
    )
    def test_true_div_scalar(self, arr, scalar):
        t_arr = Tensor(arr)

        # tensor / scalar
        t_res1 = t_arr / scalar
        np.testing.assert_allclose(t_res1.numpy(), np.array(arr) / scalar, rtol=1e-5, atol=1e-8)

        # scalar / tensor (__rtruediv__)
        t_res2 = scalar / t_arr
        np.testing.assert_allclose(t_res2.numpy(), scalar / np.array(arr))

    def test_true_div_broadcasting(self):
        t0 = Tensor([[10, 20, 30], [5, 25, 50]])
        t1 = Tensor([10, 5, 2])
        out = (t0 / t1).numpy()
        expected = np.array([[1, 4, 15], [0.5, 5, 25]])
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
    def test_true_div_zero_size(self, shape):
        t0 = Tensor.empty(*shape)
        t1 = Tensor.empty(*shape)
        out = (t0 / t1).numpy()
        # numpy empty-empty true division yields empty with same shape
        expected = (np.empty(shape) / np.empty(shape))
        np.testing.assert_array_equal(out, expected)

    def test_true_division_by_zero(self):
        # division by scalar zero
        t0 = Tensor([1, 2, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            div0 = (t0 / 0).numpy()
            expected0 = np.array([1, 2, 3], dtype=t0.dtype) / 0
            np.testing.assert_array_equal(div0, expected0)

        # division where tensor contains zero
        t1 = Tensor([1, 0, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            div1 = (t0 / t1).numpy()
            expected1 = np.array([1, 2, 3]) / np.array([1, 0, 2])
            np.testing.assert_array_equal(div1, expected1)
    @pytest.mark.parametrize(
        "a,b",
        [
            ([5, 7, 9], [2, 3, 4]),
            ([[10, 20], [30, 40]], [[3, 4], [5, 6]]),
        ],
    )
    def test_floor_div_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a // t_b
        np.testing.assert_array_equal(t_c.numpy(), np.array(a) // np.array(b))

    @pytest.mark.parametrize(
        "arr, scalar",
        [
            ([9, 18, 27], 3),
            ([[8, 16], [24, 32]], 8),
        ],
    )
    def test_floor_div_scalar(self, arr, scalar):
        t_arr = Tensor(arr)

        # tensor // scalar
        t_res1 = t_arr // scalar
        np.testing.assert_array_equal(t_res1.numpy(), np.array(arr) // scalar)

        # scalar // tensor (__rfloordiv__)
        t_res2 = scalar // t_arr
        np.testing.assert_array_equal(t_res2.numpy(), scalar // np.array(arr))

    def test_floor_div_broadcasting(self):
        t0 = Tensor([[9, 18, 27], [12, 24, 36]])
        t1 = Tensor([3, 6, 9])
        out = (t0 // t1).numpy()
        expected = np.array([[3, 3, 3], [4, 4, 4]])
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
    def test_floor_div_zero_size(self, shape):
        t0 = Tensor.empty(*shape)
        t1 = Tensor.empty(*shape)
        out = (t0 // t1).numpy()
        expected = np.empty(shape, dtype=t0.dtype)
        np.testing.assert_array_equal(out, expected)

    def test_floor_division_by_zero_errors(self):
        t0 = Tensor([1, 2, 3])
        with pytest.raises(ZeroDivisionError):
            _ = t0 // 0

        t1 = Tensor([1, 0, 2])
        with pytest.raises(ZeroDivisionError):
            _ = t0 // t1

    def test_reverse_true_and_floor_div(self):
        # reverse true division
        t = Tensor([2, 4, 8])
        out_t = (16 / t).numpy()
        np.testing.assert_array_equal(out_t, np.array([8, 4, 2]))

        # reverse floor division
        t2 = Tensor([5, 10, 20])
        out_f = (100 // t2).numpy()
        np.testing.assert_array_equal(out_f, np.array([20, 10, 5]))

        # reverse floor with mixed tensor shapes
        t3 = Tensor([2, 5])
        t4 = Tensor([10, 2])
        out_m = (t3 // t4).numpy()
        expected_m = np.array([2 // 10, 5 // 2])
        np.testing.assert_array_equal(out_m, expected_m)

        # ensure zero divisor still errors in reverse floor
        with pytest.raises(ZeroDivisionError):
            _ = 100 // Tensor([1, 0, 2])
