# import array
# import math

# import pytest

# from grad.tensor import Tensor, dtypes


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
#         assert isinstance(t.buffer.obj, array.array)
#         assert t.buffer.format == "f"
#         check_tensor_data(t, data)

#     def test_int32(self):
#         data = [[1, 2], [3, 4]]
#         t = Tensor(data, dtype=dtypes.int32)

#         assert t.shape == (2, 2)
#         assert t.dtype is dtypes.int32
#         assert isinstance(t.buffer.obj, array.array)
#         assert t.buffer.format == "i"
#         check_tensor_data(t, data)

#     def test_bool_from_int(self):
#         data = [[1, 0], [0, 1]]
#         t = Tensor(data, dtype=dtypes.bool)

#         assert t.shape == (2, 2)
#         assert t.dtype is dtypes.bool
#         assert isinstance(t.buffer.obj, array.array)
#         assert t.buffer.format == "b"
#         check_tensor_data(t, data)

#     def test_bool(self):
#         data = [[True, False], [False, True]]
#         t = Tensor(data, dtype=dtypes.bool)

#         assert t.shape == (2, 2)
#         assert isinstance(t.buffer.obj, array.array)
#         assert t.buffer.format == "b"  # stored as uint8
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


# class TestTensorFactories:
#     @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1,)])
#     def test_ones_float(self, shape):
#         t = Tensor.ones(shape, dtype=dtypes.float32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 1.0))

#     def test_ones_int(self):
#         shape = (1, 4)
#         t = Tensor.ones(shape, dtype=dtypes.int32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 1))

#     def test_zeros(self):
#         shape = (3, 2)
#         t = Tensor.zeros(shape, dtype=dtypes.float32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 0.0))

#     def test_zeros_int(self):
#         shape = (5,)
#         t = Tensor.zeros(shape, dtype=dtypes.int32)
#         check_tensor_data(t, generate_nested_factory_data(shape, 0))


# class TestFloat16:
#     """Tests for Float16 tensor creation and operations"""

#     def test_fp16_initialisation(self):
#         """Input: [1.0, 2.5, -3.0, 0.1]"""
#         data = [1.0, 2.5, -3.0, 0.1]
#         t = Tensor(data, dtype=dtypes.float16)

#         assert t.dtype is dtypes.float16
#         assert t.dtype.fmt == "e"  # uint16 storage
#         assert t.buffer.nbytes == 2 * len(data)
#         check_tensor_data(t, data)

#     def test_fp16_ones(self):
#         """Input: shape=(2,2)"""
#         shape = (2, 2)
#         t = Tensor.ones(shape, dtype=dtypes.float16)
#         assert t.dtype is dtypes.float16
#         assert t.dtype.fmt == "e"
#         assert t.buffer.nbytes == 4 * dtypes.float16.itemsize
#         expected = generate_nested_factory_data(shape, 1.0)
#         check_tensor_data(t, expected)

#     def test_fp16_zeros(self):
#         """Input: shape=(2,2)"""
#         shape = (2, 2)
#         t = Tensor.zeros(shape, dtype=dtypes.float16)
#         assert t.dtype is dtypes.float16
#         assert t.dtype.fmt == "e"
#         assert t.buffer.nbytes == 4 * dtypes.float16.itemsize
#         expected = generate_nested_factory_data(shape, 0.0)
#         check_tensor_data(t, expected)

#     def test_fp16_roundtrip_conversion(self):
#         """Tests edge cases including: zeros, normal values, subnormal values, infinities, NaN"""
#         test_values = [
#             0.0,
#             -0.0,  # zeros
#             1.0,
#             -1.0,
#             2.0,
#             -2.0,
#             0.1,
#             -0.1,
#             0.5,
#             -0.5,
#             65504.0,
#             -65504.0,
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
#         """Input: Normal and edge case values for FP16"""
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]
#         t = Tensor(values, dtype=dtypes.float16)

#         # Get the nested representation
#         nested = t._to_nested()

#         # Check values are close to original
#         for orig, conv in zip(values, nested):
#             assert abs(orig - conv) < 1e-3, f"Value mismatch: {orig} vs {conv}"

#     def test_memoryview_conversion(self):
#         """Input: Standard FP16 test values"""
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]

#         # Create a memoryview of uint16 values
#         arr = array.array("H", values)
#         view = memoryview(arr)

#         # converted = formatted_fp16_buffer(view)

#         # Check conversion accuracy
#         for orig, conv in zip(values, converted):
#             assert abs(orig - conv) < 1e-3, f"Conversion error: {orig} vs {conv}"

#     def test_flatten_and_shape(self):
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
#         flat = list(Tensor._flatten_gen(data))
#         assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
#         assert Tensor._infer_shape(data) == (2, 2, 2)

#     def test_repr_contains(self):
#         t = Tensor(
#             [[1.0, 2.0], [3.0, 4.0]],
#             dtype=dtypes.float32,
#             device="gpu",
#             requires_grad=True,
#         )

#         r = repr(t)
#         for fragment in (
#             "shape=(2, 2)",
#             "dtype=float32",
#             "device=gpu",
#             "requires_grad=True",
#             "data=[[1.0, 2.0], [3.0, 4.0]]",
#         ):
#             assert fragment in r

#     def test_requires_grad_and_grad_defaults(self):
#         t_default = Tensor([1, 2])
#         assert t_default.requires_grad is None

#         t_true = Tensor([1, 2], requires_grad=True)
#         assert t_true.requires_grad is True

#         t_false = Tensor([1, 2], requires_grad=False)
#         assert t_false.requires_grad is False

#         assert t_default.grad is None


# ARRAY_E_SUPPORTED = "e" in array.typecodes


# class TestFP16Conversions:
#     """Tests for float16<->uint16 conversions"""

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
#         """Input: Normal range FP16 values including fractional, max normal, min normal and subnormal"""
#         uint16_val = float16_to_uint16(value)
#         reconverted = uint16_to_float16(uint16_val)

#         # Tolerance depends on magnitude
#         tol = 1e-3 if abs(value) > 1e-3 else 1e-6
#         assert abs(value - reconverted) < tol, f"Value {value} converted to {reconverted}"

#     def test_special_values(self):
#         """Input: Infinity and NaN values"""
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
#         """Input: Max normal (65504.0), min normal (6.104e-5), min subnormal (5.96e-8)"""
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
#     """Tests for FP16 tensor math operations"""

#     def test_basic_fp16_tensor_creation(self):
#         """Input: Range of FP16 values from 0.0 to min normal"""
#         # Test a range of values including edge cases
#         values = [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 6.104e-5]
#         t = Tensor(values, dtype=dtypes.float16)

#         assert t.dtype is dtypes.float16
#         assert t.shape == (len(values),)

#         # Verify storage format
#         assert t.buffer.format == "H"  # type: ignore

#         # Get the nested representation and check accuracy
#         nested = t._to_nested()
#         for orig, conv in zip(values, nested):
#             assert abs(orig - conv) < 1e-3, f"Value mismatch: {orig} vs {conv}"

#     def test_fp16_tensor_operations(self):
#         """TODO: Implement FP16 tensor operations tests"""
#         pass

#     def test_fp16_tensor_conversion(self):
#         """TODO: Implement FP16 tensor dtype conversion tests"""
#         pass


# class TestFP16Utils:
#     """Tests for FP16 utilities and memory management"""

#     def test_memoryview_conversion(self):
#         """Input: Standard FP16 test values for memoryview conversion"""
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
#         """Input: Array with non-uint16 format"""
#         # Create a memoryview with non-uint16 format
#         arr = array.array("i", [1, 2, 3])
#         view = memoryview(arr)

#         # Should raise TypeError for wrong format
#         with pytest.raises(TypeError):
#             formatted_fp16_buffer(view)


# class TestTensorStride:
#     def test_stride_1d(self):
#         t = Tensor([1, 2, 3, 4])
#         stride = t._stride
#         assert stride == (1,)

#     def test_stride_2d(self):
#         t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
#         assert t.shape == (3, 4) and t._stride == (4, 1)

#         t = Tensor([[1, 2, 3], [4, 5, 6]])
#         assert t._stride == (3, 1)

#     def test_stride_3d(self):
#         t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#         stride = t._stride
#         assert stride == (4, 2, 1)

#     def test_stride_scalar(self):
#         t = Tensor(42)

#         stride = t._stride
#         assert stride == ()

#     def test_stride_non_contiguous(self):
#         t = Tensor([[1, 2, 3], [4, 5, 6]])
#         if hasattr(t, "__getitem__"):
#             stride = t._stride
#             assert stride == (3, 1)


# class TestTensorTranspose:
#     def test_Tensor_T_1d(self):
#         data = [1.0, 2.0, 3.0]
#         t = Tensor(data)
#         tT = Tensor.T(t)
#         # For 1D, should be identity
#         check_tensor_data(tT, data)
#         assert tT.shape == (3,)

#     def test_Tensor_T_2d(self):
#         data = [[1.0, 2.0], [3.0, 4.0]]
#         t = Tensor(data)
#         tT = Tensor.T(t)
#         assert t[0, 0] == tT[0, 0]
#         assert t[0, 1] == tT[1, 0]
#         assert t[1, 0] == tT[0, 1]
#         assert t[1, 1] == tT[1, 1]
#         assert tT.shape == (2, 2)

#     def test_Tensor_T_3d_raises(self):
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
#         t = Tensor(data)
#         with pytest.raises(BufferError):
#             Tensor.T(t)


# class TestTensorPermute:
#     def test_permute_2d_transpose(self):
#         """Input: 2D tensor (3,2) permuted to (2,3)"""
#         data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # Shape (3, 2)
#         t = Tensor(data, dtype=dtypes.float32)
#         permuted_t = Tensor._contiguous_tensor(Tensor.permute(t, 1, 0))  # Permute to shape (2, 3)

#         expected_data = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]  # Transposed data
#         check_tensor_data(permuted_t, expected_data)
#         assert permuted_t.shape == (2, 3)
#         assert permuted_t.stride() == (3, 1)
#         assert t.stride() == (2, 1)
#         assert permuted_t._contiguous

#     def test_permute_3d_swap_0_1(self):
#         """Input: 3D tensor (2,2,2) permuted to swap dims 0,1"""
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape (2, 2, 2)
#         t = Tensor(data, dtype=dtypes.int32)
#         # Permute (0, 1, 2) -> (1, 0, 2)
#         permuted_t = Tensor.permute(t, 1, 0, 2)

#         expected_data = [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]  # Shape (2, 2, 2)
#         check_tensor_data(permuted_t, expected_data)
#         assert permuted_t.shape == (2, 2, 2)
#         assert permuted_t.stride() == (2, 4, 1)  # Check calculated stride

#     def test_permute_3d_swap_0_2(self):
#         """Input: 3D tensor (2,2,2) permuted to swap dims 0,2"""
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape (2, 2, 2)
#         t = Tensor(data, dtype=dtypes.int32)
#         # Permute (0, 1, 2) -> (2, 1, 0)
#         permuted_t = Tensor.permute(t, 2, 1, 0)

#         expected_data = [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]  # Shape (2, 2, 2)
#         check_tensor_data(permuted_t, expected_data)
#         assert permuted_t.shape == (2, 2, 2)
#         assert permuted_t.stride() == (1, 2, 4)

#     def test_permute_3d_cycle(self):
#         """Input: 3D tensor (2,2,2) with cyclic permutation (0,1,2)->(1,2,0)"""
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape (2, 2, 2)
#         t = Tensor(data, dtype=dtypes.int32)
#         # Permute (0, 1, 2) -> (1, 2, 0)
#         permuted_t = Tensor.permute(t, 1, 2, 0)

#         # Manually construct expected data based on index mapping: (i, j, k) -> (j, k, i)
#         # t[0, 0, 0] -> t_perm[0, 0, 0]
#         # t[0, 0, 1] -> t_perm[0, 1, 0]
#         # t[0, 1, 0] -> t_perm[1, 0, 0]
#         # t[0, 1, 1] -> t_perm[1, 1, 0]
#         # t[1, 0, 0] -> t_perm[0, 0, 1]
#         # t[1, 0, 1] -> t_perm[0, 1, 1]
#         # t[1, 1, 0] -> t_perm[1, 0, 1]
#         # t[1, 1, 1] -> t_perm[1, 1, 1]
#         expected_data = [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]  # Shape (2, 2, 2)
#         check_tensor_data(permuted_t, expected_data)
#         assert permuted_t.shape == (2, 2, 2)
#         assert permuted_t.stride() == (2, 1, 4)  # Check calculated stride

#     def test_permute_identity(self):
#         """Input: 2D tensor (2,2) with identity permutation"""
#         data = [[1.0, 2.0], [3.0, 4.0]]  # Shape (2, 2)
#         t = Tensor(data, dtype=dtypes.float32)
#         permuted_t = Tensor.permute(t, 0, 1)  # Permute to (0, 1)

#         # Should be a view but logically identical to the original
#         check_tensor_data(permuted_t, data)
#         assert permuted_t.shape == (2, 2)
#         assert permuted_t.stride() == (2, 1)  # Should have the same stride as original contiguous
#         assert not permuted_t._contiguous  # Views are marked non-contiguous by default

#     def test_permute_scalar(self):
#         """Input: Scalar tensor (shape=())"""
#         t = Tensor(5.0)  # Shape ()
#         permuted_t = Tensor.permute(t)  # No indices expected for scalar

#         check_tensor_data(permuted_t, 5.0)
#         assert permuted_t.shape == ()
#         assert permuted_t.stride() == ()
#         assert not permuted_t._contiguous

#     def test_permute_1d(self):
#         """Input: 1D tensor of length 3"""
#         data = [1.0, 2.0, 3.0]
#         t = Tensor(data)
#         permuted_t = Tensor.permute(t, 0)

#         check_tensor_data(permuted_t, data)
#         assert permuted_t.shape == (3,)
#         assert permuted_t.stride() == (1,)

#         assert not permuted_t._contiguous

#     def test_permute_invalid_num_indices(self):
#         """Input: 2D tensor with wrong number of permutation indices"""
#         data = [[1.0, 2.0], [3.0, 4.0]]
#         t = Tensor(data)
#         with pytest.raises(ValueError):
#             Tensor.permute(t, 0)
#         with pytest.raises(ValueError):
#             Tensor.permute(t, 0, 1, 2)

#     def test_permute_invalid_index_value(self):
#         """Input: 3D tensor with out-of-range indices"""
#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape (2, 2, 2), dimensions 0, 1, 2
#         t = Tensor(data)
#         with pytest.raises(ValueError):
#             Tensor.permute(t, 0, 1, 3)
#         with pytest.raises(ValueError):
#             Tensor.permute(t, -1, 1, 0)

#     def test_permute_non_permutation(self):
#         """Input: 3D tensor with duplicate indices"""

#         data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # Shape (2, 2, 2)
#         t = Tensor(data)
#         with pytest.raises(ValueError):
#             Tensor.permute(t, 0, 0, 1)  # duplicate index
