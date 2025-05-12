import array
import math

import pytest

from grad.tensor import Tensor, dtypes


# Helper function to check tensor data consistency, including fp16 handling nuances.
# It infers the shape from the expected data and compares it,
# then checks if the data reconstructed from the buffer matches the expected data.
# It handles floating point comparisons with almostEqual for floating point types.
def check_tensor_data(tensor, expected_nested_data):
    expected_shape = Tensor._infer_shape(expected_nested_data)
    assert (
        tensor.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
    reconstructed_nested_data = tensor._buffer_to_nested()

    def recursive_compare(reco, expected):
        if isinstance(expected, list) or isinstance(expected, tuple):
            assert isinstance(reco, list) or isinstance(
                reco, tuple
            ), "Type mismatch during recursive comparison"
            assert len(reco) == len(
                expected
            ), f"Length mismatch in nested structure: expected {len(expected)}, got {len(reco)}"
            for r_item, e_item in zip(reco, expected):
                recursive_compare(r_item, e_item)
        elif isinstance(expected, float):
            assert isinstance(reco, float), "Type mismatch for float comparison"

            assert math.isclose(
                reco, expected, rel_tol=1e-3, abs_tol=1e-8
            ), f"Float values not close: expected {expected}, got {reco}"

        else:
            assert reco == expected, f"Value mismatch: expected {expected}, got {reco}"

    recursive_compare(reconstructed_nested_data, expected_nested_data)


class TestTensor:
    def test_initialization_basic_float32(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        tensor = Tensor(data, dtype=dtypes.float32)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == dtypes.float32
        assert isinstance(tensor._raw, array.array)
        assert tensor._raw.typecode == "f"
        check_tensor_data(tensor, data)

    def test_initialization_basic_int32(self):
        data = [[1, 2], [3, 4]]
        tensor = Tensor(data, dtype=dtypes.int32)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == dtypes.int32
        assert isinstance(tensor._raw, array.array)
        assert tensor._raw.typecode == "i"  # 'i' for int32
        check_tensor_data(tensor, data)

    def test_initialization_bool_from_int(self):
        data = [[1, 0], [0, 1]]
        tensor = Tensor(data, dtype=dtypes.bool)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == dtypes.bool
        print(tensor)
        assert isinstance(tensor._raw, bytearray)
        check_tensor_data(tensor, data)

    def test_initialization_basic_bool(self):
        data = [[True, False], [False, True]]
        tensor = Tensor(data, dtype=dtypes.bool)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == dtypes.bool
        assert isinstance(tensor._raw, bytearray)
        check_tensor_data(tensor, data)

    def test_initialization_scalar_like(self):
        data_list = [5.5]
        tensor = Tensor(data_list, dtype=dtypes.float32)
        assert tensor.shape == (1,)
        assert tensor.dtype == dtypes.float32
        check_tensor_data(tensor, data_list)

    def test_initialization_1d(self):
        data = [1.0, 2.0, 3.0]
        tensor = Tensor(data, dtype=dtypes.float32)
        assert tensor.shape == (3,)
        check_tensor_data(tensor, data)

    def test_initialization_3d(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        tensor = Tensor(data, dtype=dtypes.int32)
        assert tensor.shape == (2, 2, 2)
        check_tensor_data(tensor, data)

    def test_initialization_nested_lists_varying_depth(self):
        data_2d = [[1.0, 2.0], [3.0, 4.0]]
        tensor_2d = Tensor(data_2d)  # Defaults to float32
        assert tensor_2d.shape == (2, 2)
        check_tensor_data(tensor_2d, data_2d)

        data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        tensor_3d = Tensor(data_3d, dtype=dtypes.int32)
        assert tensor_3d.shape == (2, 2, 2)
        check_tensor_data(tensor_3d, data_3d)

        data_1d = [10, 20, 30]
        tensor_1d = Tensor(data_1d, dtype=dtypes.int32)
        assert tensor_1d.shape == (3,)
        assert tensor_1d.dtype == dtypes.int32
        assert isinstance(tensor_1d._raw, array.array)
        assert tensor_1d._raw.typecode == "i"  # 'q' for int64
        check_tensor_data(tensor_1d, data_1d)

    def test_initialization_unsupported_data_type(self):
        with pytest.raises(TypeError):
            Tensor("hello")  # String is not supported

    def test_initialization_inconsistent_shape(self):
        data_inconsistent = [[1, 2], [3]]
        # _flatten([[1, 2], [3]]) -> [1, 2, 3]
        # _infer_shape([[1, 2], [3]]) -> (2, 2) based on the first element's structure.
        # _nest([1, 2, 3], [2, 2]) will attempt to reconstruct a (2, 2) shape from a flat list of size 3, which will fail.
        with pytest.raises(IndexError):
            Tensor(data_inconsistent, dtype=dtypes.float32)

    def test_ones_basic(self):
        shape = (2, 3)
        tensor = Tensor.ones(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.float32
        expected_data = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        check_tensor_data(tensor, expected_data)

    def test_ones_int(self):
        shape = (1, 4)
        tensor = Tensor.ones(shape, dtype=dtypes.int32)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.int32
        expected_data = [[1, 1, 1, 1]]
        check_tensor_data(tensor, expected_data)

    def test_ones_scalar_shape(self):
        shape = (1,)
        tensor = Tensor.ones(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        expected_data = [1.0]
        check_tensor_data(tensor, expected_data)

    def test_ones_zero_size(self):
        shape = (0,)
        tensor = Tensor.ones(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        assert tensor._buffer.nbytes == 0

        shape = (2, 0)
        tensor = Tensor.ones(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        assert tensor._buffer.nbytes == 0

    def test_zeros_basic(self):
        shape = (3, 2)
        tensor = Tensor.zeros(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.float32
        expected_data = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        check_tensor_data(tensor, expected_data)

    def test_zeros_int(self):
        shape = (5,)
        tensor = Tensor.zeros(shape, dtype=dtypes.int32)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.int32
        expected_data = [0, 0, 0, 0, 0]
        check_tensor_data(tensor, expected_data)

    def test_zeros_scalar_shape(self):
        shape = (1,)
        tensor = Tensor.zeros(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        expected_data = [0.0]
        check_tensor_data(tensor, expected_data)

    def test_zeros_zero_size(self):
        shape = (0,)
        tensor = Tensor.zeros(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        assert tensor._buffer.nbytes == 0

        shape = (2, 0)
        tensor = Tensor.zeros(shape, dtype=dtypes.float32)
        assert tensor.shape == shape
        assert tensor._buffer.nbytes == 0

    def test_flatten(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        flat_data = Tensor._flatten(data)
        assert flat_data == [1, 2, 3, 4, 5, 6, 7, 8]

        data_mixed = [1, [2, [3, 4]], 5]
        flat_data_mixed = Tensor._flatten(data_mixed)
        assert flat_data_mixed == [1, 2, 3, 4, 5]

        data_flat = [1, 2, 3]
        flat_data_flat = Tensor._flatten(data_flat)
        assert flat_data_flat == [1, 2, 3]

        data_empty = []
        flat_data_empty = Tensor._flatten(data_empty)
        assert flat_data_empty == []

        data_nested_empty = [[]]
        flat_data_nested_empty = Tensor._flatten(data_nested_empty)
        assert flat_data_nested_empty == []

    def test_infer_shape(self):
        data_2d = [[1, 2], [3, 4]]
        assert Tensor._infer_shape(data_2d) == (2, 2)

        data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        assert Tensor._infer_shape(data_3d) == (2, 2, 2)

        data_1d = [1, 2, 3]
        assert Tensor._infer_shape(data_1d) == (3,)

        data_scalar_like = [5]
        assert Tensor._infer_shape(data_scalar_like) == (1,)

        data_empty = []
        assert Tensor._infer_shape(data_empty) == (0,)

        data_nested_empty = [[]]
        assert Tensor._infer_shape(data_nested_empty) == (1, 0)

    def test_repr(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        tensor = Tensor(data, dtype=dtypes.float32, device="gpu", requires_grad=True)
        # The repr string should match the expected format.
        expected_repr_start = "Tensor(shape=(2, 2), data=[[1.0, 2.0], [3.0, 4.0]], device=gpu, dtype=float32, requires_grad=True)"
        assert repr(tensor) == expected_repr_start

        data_int = [1, 2, 3]
        tensor_int = Tensor(data_int, dtype=dtypes.int32, device="cpu", requires_grad=False)
        expected_repr_int_start = (
            "Tensor(shape=(3,), data=[1, 2, 3], device=cpu, dtype=int32, requires_grad=False)"
        )
        assert repr(tensor_int) == expected_repr_int_start

        data_float16 = [1.0, 2.5, -3.0]
        tensor_fp16 = Tensor(data_float16, dtype=dtypes.float16)

        repr_str = repr(tensor_fp16)
        assert "Tensor(shape=(3,), data=" in repr_str
        assert "device=cpu" in repr_str
        assert "dtype=float16" in repr_str
        assert "requires_grad=None" in repr_str

        # Check the data part more closely by evaluating the string representation of the list
        # A more robust test would parse the data string, but for simplicity, we check for expected float representations.
        # Note: the actual string representation of float16 values after conversion and back might slightly differ
        # from the input string (e.g., 2.5 might become 2.500...). Check for approximate value representations.
        # Let's check for the presence of key parts of the data string representation.
        assert "[1.0" in repr_str
        # Due to fp16 precision, check for string representation that is close to 2.5 or -3.0
        assert (
            "2.5" in repr_str or "2.50" in repr_str or "2.500" in repr_str
        )  # Check for 2.5 representation
        assert "-3.0" in repr_str or "-3.00" in repr_str  # Check for -3.0 representation

    def test_initialization_float16_fallback(self):
        # Test the path where array.array does NOT support 'e' (uses bytearray and struct)
        data = [1.0, 2.5, -3.0, 0.1]
        tensor = Tensor(data, dtype=dtypes.float16)
        assert tensor.shape == (4,)
        assert tensor.dtype == dtypes.float16
        assert isinstance(tensor._raw, bytearray)
        assert len(tensor._raw) == 4 * dtypes.float16.itemsize  # 4 elements, 2 bytes each for fp16
        check_tensor_data(tensor, data)

    def test_initialization_float16_array(self):
        data = [1.0, 2.5, -3.0, 0.1]
        tensor = Tensor(data, dtype=dtypes.float16)
        assert tensor.shape == (4,)
        assert tensor.dtype == dtypes.float16
        assert isinstance(tensor._raw, bytearray)
        assert len(tensor._raw) == len(data) * 2
        check_tensor_data(tensor, data)

    def test_ones_float16_fallback(self):
        shape = (2, 2)
        tensor = Tensor.ones(shape, dtype=dtypes.float16)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.float16
        assert isinstance(tensor._raw, bytearray)
        assert len(tensor._raw) == 4 * dtypes.float16.itemsize
        expected_data = [[1.0, 1.0], [1.0, 1.0]]
        check_tensor_data(tensor, expected_data)

    def test_ones_float16_array(self):
        shape = (2, 2)
        tensor = Tensor.ones(shape, dtype=dtypes.float16)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.float16
        assert isinstance(tensor._raw, bytearray)
        assert len(tensor._raw) == math.prod(shape) * dtypes.float16.itemsize
        expected_data = [[1.0, 1.0], [1.0, 1.0]]
        check_tensor_data(tensor, expected_data)

    def test_zeros_float16_fallback(self):
        shape = (2, 2)
        tensor = Tensor.zeros(shape, dtype=dtypes.float16)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.float16
        assert isinstance(tensor._raw, bytearray)
        assert len(tensor._raw) == 4 * dtypes.float16.itemsize  # 4 elements, 2 bytes each
        expected_data = [[0.0, 0.0], [0.0, 0.0]]
        check_tensor_data(tensor, expected_data)

    def test_zeros_float16_array(self):
        shape = (2, 2)
        tensor = Tensor.zeros(shape, dtype=dtypes.float16)
        assert tensor.shape == shape
        assert tensor.dtype == dtypes.float16
        assert isinstance(tensor._raw, bytearray)
        assert len(tensor._raw) == 8
        expected_data = [[0.0, 0.0], [0.0, 0.0]]
        check_tensor_data(tensor, expected_data)

    def test_requires_grad_initialization(self):
        data = [1, 2]
        tensor_default = Tensor(data)
        assert tensor_default.requires_grad is None

        tensor_true = Tensor(data, requires_grad=True)
        assert tensor_true.requires_grad is True

        tensor_false = Tensor(data, requires_grad=False)
        assert tensor_false.requires_grad is False

    def test_grad_initialization(self):
        data = [1, 2]
        tensor = Tensor(data)
        assert tensor.grad is None
