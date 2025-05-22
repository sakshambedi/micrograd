# import array
# import math

# import pytest

# from grad.tensor import Tensor, dtypes
# from grad.utils.fp16 import (
#     float16_to_uint16,
#     formatted_fp16_buffer,
#     uint16_to_float16,
# )


# def generate_nested_factory_data(shape, value):
#     """Helper to generate nested list structure for factory tests."""
#     if not shape:
#         return value  # Scalar case
#     if len(shape) == 1:
#         return [value] * shape[0]
#     # Recursively build nested list
#     inner_shape = shape[1:]
#     return [generate_nested_factory_data(inner_shape, value) for _ in range(shape[0])]


# def check_tensor_data(tensor: Tensor, expected_nested):
#     expected_shape = Tensor._infer_shape(expected_nested)
#     assert (
#         tensor.shape == expected_shape
#     ), f"shape mismatch: expected {expected_shape}, got {tensor.shape}"

#     reconstructed = tensor._to_nested()  # new helper in refactor

#     def rec(r, e):
#         if isinstance(e, (list, tuple)):
#             assert isinstance(r, (list, tuple))
#             assert len(r) == len(e)
#             for rr, ee in zip(r, e):
#                 rec(rr, ee)
#         elif isinstance(e, float):
#             assert math.isclose(r, e, rel_tol=1e-3, abs_tol=1e-8)
#         else:
#             assert r == e

#     rec(reconstructed, expected_nested)


# class TestTensorInit:
#     def test_float32(self):
#         data = [[1.0, 2.0], [3.0, 4.0]]
#         t = Tensor(data, dtype=dtypes.float32)

#         assert t.shape == (2, 2)
#         assert t.dtype is dtypes.float32
#         assert isinstance(t.storage.obj, array.array)
#         assert t.storage.format == "f"
#         check_tensor_data(t, data)

#     def test_int32(self):
#         data = [[1, 2], [3, 4]]
#         t = Tensor(data, dtype=dtypes.int32)

#         assert t.shape == (2, 2)
#         assert t.dtype is dtypes.int32
#         assert isinstance(t.storage.obj, array.array)
#         assert t.storage.format == "i"
#         check_tensor_data(t, data)

#     def test_bool_from_int(self):
#         data = [[1, 0], [0, 1]]
#         t = Tensor(data, dtype=dtypes.bool)

#         assert t.shape == (2, 2)
#         assert t.dtype is dtypes.bool
#         assert isinstance(t.storage.obj, array.array)
#         assert t.storage.format == "b"
#         check_tensor_data(t, data)

#     def test_bool(self):
#         data = [[True, False], [False, True]]
#         t = Tensor(data, dtype=dtypes.bool)

#         assert t.shape == (2, 2)
#         assert isinstance(t.storage.obj, array.array)
#         assert t.storage.format == "b"  # stored as uint8
#         assert t.dtype.fmt == "?"
#         check_tensor_data(t, data)

#     def test_scalar_like(self):
#         data = [5.5]
#         t = Tensor(data, dtype=dtypes.float32)

#         assert t.shape == (1,)
#         assert t.dtype is dtypes.float32
#         check_tensor_data(t, data)

#     def test_1d(self):
#         data = [1.0, 2.0, 3.0]
#         check_tensor_data(Tensor(data), data)

#     def test_3d(self):
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
#         t = Tensor(data, dtype=dtypes.int32)

#         assert t.shape == (2, 2, 2)
#         check_tensor_data(t, data)

#     def test_inconsistent_shape(self):
#         with pytest.raises(IndexError):
#             Tensor([[1, 2], [3]], dtype=dtypes.float32)

#     def test_unsupported_type(self):
#         with pytest.raises(TypeError):
#             Tensor("hello")  # type: ignore[arg-type]


# # --------------------------------------------------------------------------- #
# # Ones / Zeros / empty-size cases
# # --------------------------------------------------------------------------- #
# class TestTensorFactories:
#     @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1,)])
#     def test_ones_float(self, shape):
#         t = Tensor.ones(shape, dtype=dtypes.float32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 1.0))

#     def test_ones_int(self):
#         shape = (1, 4)
#         t = Tensor.ones(shape, dtype=dtypes.int32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 1))

#     def test_zero_size(self):
#         t0 = Tensor.ones((0,), dtype=dtypes.float32)
#         assert t0.storage.nbytes == 0

#         t2 = Tensor.ones((2, 0), dtype=dtypes.float32)
#         assert t2.storage.nbytes == 0

#     def test_zeros(self):
#         shape = (3, 2)
#         t = Tensor.zeros(shape, dtype=dtypes.float32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 0.0))

#     def test_zeros_int(self):
#         shape = (5,)
#         t = Tensor.zeros(shape, dtype=dtypes.int32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 0))


# class TestFloat16:
#     """Test suite for Float16 tensor operations and conversions."""

#     def test_fp16_initialisation(self):
#         """Test creating FP16 tensors from Python lists."""
#         data = [1.0, 2.5, -3.0, 0.1]
#         t = Tensor(data, dtype=dtypes.float16)

#         assert t.dtype is dtypes.float16
#         assert t.storage.dtype.fmt == "H"  # uint16 storage
#         assert t.storage.nbytes == 2 * len(data)
#         check_tensor_data(t, data)

#     def test_fp16_ones(self):
#         """Test creating FP16 tensors filled with ones."""
#         shape = (2, 2)
#         t = Tensor.ones(shape, dtype=dtypes.float16)
#         assert t.dtype is dtypes.float16
#         assert t.storage._storage.format == "H"
#         assert t.storage._storage.nbytes == 4 * dtypes.float16.itemsize
#         expected = generate_nested_factory_data(shape, 1.0)
#         check_tensor_data(t, expected)

#     def test_fp16_zeros(self):
#         """Test creating FP16 tensors filled with zeros."""
#         shape = (2, 2)
#         t = Tensor.zeros(shape, dtype=dtypes.float16)
#         assert t.dtype is dtypes.float16
#         assert t.storage._storage.format == "H"
#         assert t.storage._storage.nbytes == 4 * dtypes.float16.itemsize
#         expected = generate_nested_factory_data(shape, 0.0)
#         check_tensor_data(t, expected)

#     def test_fp16_roundtrip_conversion(self):
#         """Test roundtrip conversion between float and FP16 representation."""
#         test_values = [
#             0.0,
#             -0.0,  # zeros
#             1.0,
#             -1.0,
#             2.0,
#             -2.0,  # simple values
#             0.1,
#             -0.1,
#             0.5,
#             -0.5,  # fractional values
#             65504.0,
#             -65504.0,  # largest normal fp16 values
#             6.104e-5,
#             -6.104e-5,  # smallest normal fp16 values
#             5.96e-8,
#             -5.96e-8,  # subnormal values
#             float("inf"),
#             float("-inf"),  # infinities
#             float("nan"),  # NaN
#         ]

#         for val in test_values:
#             uint16_val = float16_to_uint16(val)
#             reconverted = uint16_to_float16(uint16_val)

#             # Special handling for NaN since NaN != NaN
#             if math.isnan(val):
#                 assert math.isnan(reconverted), f"Expected NaN, got {reconverted}"
#             # Special handling for infinities
#             elif math.isinf(val):
#                 assert math.isinf(reconverted), f"Expected inf, got {reconverted}"
#                 assert (val > 0) == (reconverted > 0), "Infinity sign doesn't match"
#             # Normal value comparison with tolerance
#             else:
#                 tol = 1e-3 if abs(val) > 1e-3 else 1e-6
#                 assert (
#                     abs(val - reconverted) < tol
#                 ), f"Conversion error too large: {val} -> {reconverted}"

#     def test_tensor_representation(self):
#         """Test that FP16 tensor values are correctly represented."""
#         # Test a range of values including edge cases
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]
#         t = Tensor(values, dtype=dtypes.float16)

#         # Get the nested representation
#         nested = t._to_nested()

#         # Check values are close to original
#         for orig, conv in zip(values, nested):
#             assert abs(orig - conv) < 1e-3, f"Value mismatch: {orig} vs {conv}"

#     def test_memoryview_conversion(self):
#         """Test conversion from uint16 memoryview to float values."""
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]
#         uint16_values = [float16_to_uint16(v) for v in values]

#         # Create a memoryview of uint16 values
#         arr = array.array("H", uint16_values)
#         view = memoryview(arr)

#         # Convert using formatted_fp16_buffer
#         converted = formatted_fp16_buffer(view)

#         # Check conversion accuracy
#         for orig, conv in zip(values, converted):
#             assert abs(orig - conv) < 1e-3, f"Conversion error: {orig} vs {conv}"


# def test_flatten_and_shape():
#     data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
#     flat = list(Tensor._flatten_gen(data))
#     assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
#     assert Tensor._infer_shape(data) == (2, 2, 2)


# def test_repr_contains():
#     t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32, device="gpu", requires_grad=True)

#     r = repr(t)
#     for fragment in (
#         "shape=(2, 2)",
#         "dtype='float32'",
#         "device='gpu'",
#         "requires_grad=True",
#         "data=[[1.0, 2.0], [3.0, 4.0]]",
#     ):
#         assert fragment in r


# def test_requires_grad_and_grad_defaults():
#     t_default = Tensor([1, 2])
#     assert t_default.requires_grad is None

#     t_true = Tensor([1, 2], requires_grad=True)
#     assert t_true.requires_grad is True

#     t_false = Tensor([1, 2], requires_grad=False)
#     assert t_false.requires_grad is False

#     assert t_default.grad is None


# # Check if fp16 is natively supported
# ARRAY_E_SUPPORTED = "e" in array.typecodes


# class TestFP16Conversions:
#     """Test suite for FP16 conversion utilities."""

#     @pytest.mark.parametrize(
#         "value",
#         [
#             0.0,
#             -0.0,  # zeros
#             1.0,
#             -1.0,
#             2.0,
#             -2.0,  # simple values
#             0.1,
#             -0.1,
#             0.5,
#             -0.5,  # fractional values
#             65504.0,
#             -65504.0,  # largest normal fp16 values
#             6.104e-5,
#             -6.104e-5,  # smallest normal fp16 values
#             5.96e-8,
#             -5.96e-8,  # subnormal values
#         ],
#     )
#     def test_normal_value_roundtrip(self, value):
#         """Test roundtrip conversion of normal FP16 values."""
#         uint16_val = float16_to_uint16(value)
#         reconverted = uint16_to_float16(uint16_val)

#         # Tolerance depends on magnitude
#         tol = 1e-3 if abs(value) > 1e-3 else 1e-6
#         assert abs(value - reconverted) < tol, f"Value {value} converted to {reconverted}"

#     def test_special_values(self):
#         """Test conversion of special values: infinities and NaN."""
#         # Test positive infinity
#         pos_inf_bits = float16_to_uint16(float("inf"))
#         assert pos_inf_bits == 0x7C00, f"Expected 0x7c00 for +inf, got 0x{pos_inf_bits:04x}"
#         assert math.isinf(uint16_to_float16(pos_inf_bits))
#         assert uint16_to_float16(pos_inf_bits) > 0

#         # Test negative infinity
#         neg_inf_bits = float16_to_uint16(float("-inf"))
#         assert neg_inf_bits == 0xFC00, f"Expected 0xfc00 for -inf, got 0x{neg_inf_bits:04x}"
#         assert math.isinf(uint16_to_float16(neg_inf_bits))
#         assert uint16_to_float16(neg_inf_bits) < 0

#         # Test NaN
#         nan_bits = float16_to_uint16(float("nan"))
#         assert (nan_bits & 0x7C00) == 0x7C00 and (
#             nan_bits & 0x03FF
#         ) != 0, f"Expected NaN pattern, got 0x{nan_bits:04x}"
#         assert math.isnan(uint16_to_float16(nan_bits))

#     def test_edge_values(self):
#         """Test edge case values in FP16 range."""
#         # Maximum normal value
#         max_val = 65504.0
#         max_bits = float16_to_uint16(max_val)
#         assert max_bits == 0x7BFF, f"Expected 0x7bff for max value, got 0x{max_bits:04x}"

#         # Minimum positive normal value
#         min_normal = 6.104e-5
#         min_normal_bits = float16_to_uint16(min_normal)
#         assert (
#             min_normal_bits == 0x0400
#         ), f"Expected 0x0400 for min normal, got 0x{min_normal_bits:04x}"

#         # Minimum positive subnormal value
#         min_subnormal = 5.96e-8
#         min_subnormal_bits = float16_to_uint16(min_subnormal)
#         assert (
#             min_subnormal_bits == 0x0001
#         ), f"Expected 0x0001 for min subnormal, got 0x{min_subnormal_bits:04x}"


# class TestFP16TensorOperations:
#     """Test suite for operations with FP16 tensors."""

#     def test_basic_fp16_tensor_creation(self):
#         """Test creating basic FP16 tensors with different values."""
#         # Test a range of values including edge cases
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]
#         t = Tensor(values, dtype=dtypes.float16)

#         assert t.dtype is dtypes.float16
#         assert t.shape == (len(values),)

#         # Verify storage format
#         assert t.storage._storage.format == "H"  # type: ignore

#         # Get the nested representation and check accuracy
#         nested = t._to_nested()
#         for orig, conv in zip(values, nested):
#             assert abs(orig - conv) < 1e-3, f"Value mismatch: {orig} vs {conv}"

#     def test_fp16_tensor_operations(self):
#         """Test basic operations with FP16 tensors."""
#         # This is a placeholder for future tensor operation tests
#         # Add more tests when operations are implemented
#         pass

#     def test_fp16_tensor_conversion(self):
#         """Test conversion between different dtypes with FP16 tensors."""
#         # This is a placeholder for future tensor conversion tests
#         # Add more tests when conversions are implemented
#         pass


# class TestFP16Utils:
#     """Test suite for FP16 utility functions."""

#     def test_memoryview_conversion(self):
#         """Test conversion from uint16 memoryview to float values."""
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]
#         uint16_values = [float16_to_uint16(v) for v in values]

#         # Create a memoryview of uint16 values
#         arr = array.array("H", uint16_values)
#         view = memoryview(arr)

#         # Convert using formatted_fp16_buffer
#         converted = formatted_fp16_buffer(view)

#         # Check conversion accuracy
#         for orig, conv in zip(values, converted):
#             assert abs(orig - conv) < 1e-3, f"Conversion error: {orig} vs {conv}"

#     def test_invalid_format_raises_error(self):
#         """Test that using wrong format for memoryview raises error."""
#         # Create a memoryview with non-uint16 format
#         arr = array.array("i", [1, 2, 3])
#         view = memoryview(arr)

#         # Should raise TypeError for wrong format
#         with pytest.raises(TypeError):
#             formatted_fp16_buffer(view)
