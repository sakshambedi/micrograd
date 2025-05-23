# import array
# import math
# import struct
# from typing import Any, List

# import pytest

# from grad.dtype import dtypes
# from grad.tensor import Tensor
# from grad.utils.fp16 import (
#     _STRUCT_E_OK,
#     float16_to_uint16,
#     formatted_fp16_buffer,
#     uint16_to_float16,
#     words_to_floats,
# )

# # Check if FP16 is natively supported
# ARRAY_E_SUPPORTED = "e" in array.typecodes

# # Test value sets
# NORMAL_VALUES = [0.0, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 0.5, -0.5]
# EXTREME_VALUES = [65504.0, -65504.0, 6.104e-5, -6.104e-5, 5.96e-8, -5.96e-8]
# SPECIAL_VALUES = [float("inf"), float("-inf"), float("nan")]
# ALL_TEST_VALUES = NORMAL_VALUES + EXTREME_VALUES + SPECIAL_VALUES


# class TestFP16Conversion:
#     """Tests for low-level FP16 conversion functions."""

#     def test_environment_info(self):
#         """Test environment setup and native FP16 support."""
#         print(f"\nNative FP16 support:")
#         print(f"- struct module supports 'e' format: {_STRUCT_E_OK}")
#         print(f"- array module supports 'e' format: {ARRAY_E_SUPPORTED}")

#     @pytest.mark.parametrize("value", NORMAL_VALUES + EXTREME_VALUES)
#     def test_roundtrip_conversion(self, value):
#         """Test roundtrip conversion for each value type."""
#         uint16_val = float16_to_uint16(value)
#         reconverted = uint16_to_float16(uint16_val)

#         # Use appropriate tolerance based on magnitude
#         if abs(value) > 1.0:
#             # For larger values, use relative tolerance
#             assert (
#                 abs(value - reconverted) / abs(value) < 1e-3
#             ), f"Large value {value} didn't convert properly, got {reconverted}"
#         elif abs(value) > 1e-5:
#             # For medium values, use small absolute tolerance
#             assert (
#                 abs(value - reconverted) < 1e-3
#             ), f"Medium value {value} didn't convert properly, got {reconverted}"
#         else:
#             # For tiny values, use larger absolute tolerance
#             assert (
#                 abs(value - reconverted) < 1e-7
#             ), f"Tiny value {value} didn't convert properly, got {reconverted}"

#     def test_special_values(self):
#         """Test special values (inf, -inf, nan)."""
#         # Infinity
#         pos_inf_bits = float16_to_uint16(float("inf"))
#         assert pos_inf_bits == 0x7C00, f"Expected 0x7c00 for +inf, got 0x{pos_inf_bits:04x}"
#         assert math.isinf(uint16_to_float16(pos_inf_bits))
#         assert uint16_to_float16(pos_inf_bits) > 0

#         # Negative infinity
#         neg_inf_bits = float16_to_uint16(float("-inf"))
#         assert neg_inf_bits == 0xFC00, f"Expected 0xfc00 for -inf, got 0x{neg_inf_bits:04x}"
#         assert math.isinf(uint16_to_float16(neg_inf_bits))
#         assert uint16_to_float16(neg_inf_bits) < 0

#         # NaN
#         nan_bits = float16_to_uint16(float("nan"))
#         assert (nan_bits & 0x7C00) == 0x7C00, f"Expected NaN exponent bits, got 0x{nan_bits:04x}"
#         assert (nan_bits & 0x03FF) != 0, f"Expected non-zero NaN fraction, got 0x{nan_bits:04x}"
#         assert math.isnan(uint16_to_float16(nan_bits))

#     def test_known_bit_patterns(self):
#         """Test conversion of known bit patterns."""
#         patterns = [
#             (0x0000, 0.0),  # Zero
#             (0x8000, -0.0),  # Negative zero
#             (0x3C00, 1.0),  # One
#             (0xBC00, -1.0),  # Negative one
#             (0x3800, 0.5),  # Half
#             (0xB800, -0.5),  # Negative half
#             (0x7BFF, 65504.0),  # Max normal
#             (0xFBFF, -65504.0),  # Min normal
#             (0x7C00, float("inf")),  # Infinity
#             (0xFC00, float("-inf")),  # Negative infinity
#             (0x7E00, float("nan")),  # Canonical NaN
#         ]

#         for bits, expected in patterns:
#             if math.isnan(expected):
#                 assert math.isnan(uint16_to_float16(bits)), f"Expected NaN from 0x{bits:04x}"
#             elif math.isinf(expected):
#                 assert math.isinf(uint16_to_float16(bits)), f"Expected inf from 0x{bits:04x}"
#                 assert (expected > 0) == (uint16_to_float16(bits) > 0), "Infinity sign wrong"
#             else:
#                 assert math.isclose(
#                     uint16_to_float16(bits), expected, rel_tol=1e-3, abs_tol=1e-10
#                 ), f"Expected {expected} from 0x{bits:04x}, got {uint16_to_float16(bits)}"

#     def test_batch_conversion(self):
#         """Test batch conversion of values."""
#         test_values = [0.0, 1.0, -1.0, 0.5, -0.5]
#         words = [float16_to_uint16(v) for v in test_values]
#         result = words_to_floats(words)

#         for orig, conv in zip(test_values, result):
#             assert math.isclose(
#                 orig, conv, rel_tol=1e-3, abs_tol=1e-10
#             ), f"Batch conversion failed: {orig} -> {conv}"


# class TestFP16Buffer:
#     """Tests for buffer conversions with FP16."""

#     def test_memoryview_conversion(self):
#         """Test conversion of FP16 values through memoryview."""
#         values = NORMAL_VALUES + EXTREME_VALUES
#         uint16_values = [float16_to_uint16(v) for v in values]

#         # Create a memoryview of uint16 values
#         arr = array.array("H", uint16_values)
#         view = memoryview(arr)

#         # Convert using formatted_fp16_buffer
#         converted = formatted_fp16_buffer(view)

#         # Check conversion accuracy
#         for orig, conv in zip(values, converted):
#             if abs(orig) > 1.0:
#                 assert (
#                     abs(orig - conv) / abs(orig) < 1e-3
#                 ), f"Large value {orig} wasn't converted correctly, got {conv}"
#             else:
#                 assert (
#                     abs(orig - conv) < 1e-3
#                 ), f"Value {orig} wasn't converted correctly, got {conv}"

#     def test_buffer_format_check(self):
#         """Test that buffer format is checked."""
#         # Create a buffer with incorrect format
#         arr = array.array("I", [0, 1, 2, 3])  # 'I' is unsigned int, not uint16
#         view = memoryview(arr)

#         # Should raise TypeError
#         with pytest.raises(TypeError, match="buffer format must be 'H'"):
#             formatted_fp16_buffer(view)


# class TestFP16Tensor:
#     """Tests for Tensor with FP16 dtype."""

#     def test_scalar_tensor(self):
#         """Test creating scalar FP16 tensors."""
#         for value in [0.0, 1.0, -1.0, 0.5, -0.5, 65504.0]:
#             t = Tensor(value, dtype=dtypes.float16)
#             assert t.shape == (), f"Scalar tensor should have empty shape, got {t.shape}"
#             assert t.dtype == dtypes.float16, f"Dtype should be float16, got {t.dtype}"

#             # Get the value and check it's close to original
#             nested = t._to_nested()
#             assert abs(nested - value) < 1e-3, f"Expected {value}, got {nested}"

#     def test_vector_tensor(self):
#         """Test creating vector FP16 tensors."""
#         values = NORMAL_VALUES + EXTREME_VALUES
#         t = Tensor(values, dtype=dtypes.float16)
#         assert t.shape == (len(values),), f"Expected shape {(len(values),)}, got {t.shape}"
#         assert t.dtype == dtypes.float16, f"Dtype should be float16, got {t.dtype}"

#         # Get nested representation and verify values
#         nested = t._to_nested()
#         for i, (orig, conv) in enumerate(zip(values, nested)):
#             if abs(orig) > 1.0:
#                 assert (
#                     abs(orig - conv) / abs(orig) < 1e-3
#                 ), f"Value at index {i}: expected {orig}, got {conv}"
#             else:
#                 assert abs(orig - conv) < 1e-3, f"Value at index {i}: expected {orig}, got {conv}"

#     def test_multidimensional_tensor(self):
#         """Test creating multidimensional FP16 tensors."""
#         # Create a 2x4 tensor
#         values = [[0.0, 1.0, -1.0, 0.5], [-0.5, 65504.0, -65504.0, 6.104e-5]]
#         t = Tensor(values, dtype=dtypes.float16)
#         assert t.shape == (2, 4), f"Expected shape (2, 4), got {t.shape}"
#         assert t.dtype == dtypes.float16, f"Dtype should be float16, got {t.dtype}"

#         # Get nested representation and verify values
#         nested = t._to_nested()
#         for i in range(2):
#             for j in range(4):
#                 orig = values[i][j]
#                 conv = nested[i][j]
#                 if abs(orig) > 1.0:
#                     assert (
#                         abs(orig - conv) / abs(orig) < 1e-3
#                     ), f"Value at [{i}, {j}]: expected {orig}, got {conv}"
#                 else:
#                     assert (
#                         abs(orig - conv) < 1e-3
#                     ), f"Value at [{i}, {j}]: expected {orig}, got {conv}"

#     def test_special_values_tensor(self):
#         """Test tensor with special values (inf, nan)."""
#         values = [float("inf"), float("-inf"), float("nan")]
#         t = Tensor(values, dtype=dtypes.float16)
#         assert t.shape == (3,), f"Expected shape (3,), got {t.shape}"

#         nested = t._to_nested()
#         assert math.isinf(nested[0]) and nested[0] > 0, "Expected +inf"
#         assert math.isinf(nested[1]) and nested[1] < 0, "Expected -inf"
#         assert math.isnan(nested[2]), "Expected NaN"

#     def test_empty_tensor(self):
#         """Test creating empty FP16 tensors."""
#         t = Tensor([], dtype=dtypes.float16)
#         assert t.shape == (0,), f"Empty tensor should have shape (0,), got {t.shape}"
#         assert t.dtype == dtypes.float16, f"Dtype should be float16, got {t.dtype}"
#         nested = t._to_nested()
#         assert nested == [], f"Empty tensor should have empty nested representation, got {nested}"

#     def test_storage_format(self):
#         """Test that storage uses the right format for FP16."""
#         t = Tensor([1.0, 2.0], dtype=dtypes.float16)

#         # Check storage format when native support is missing
#         if not ARRAY_E_SUPPORTED:
#             assert (
#                 t.storage._storage.format == "H"
#             ), f"Expected 'H' format, got {t.storage._storage.format}"
#         else:
#             assert (
#                 t.storage._storage.format == "e"
#             ), f"Expected 'e' format, got {t.storage._storage.format}"


# if __name__ == "__main__":
#     print(f"Running FP16 tests (Native support: {_STRUCT_E_OK})")
#     # This allows running basic tests without pytest
#     TestFP16Conversion().test_environment_info()

#     # Run a sample conversion test
#     test_val = 1.0
#     print(f"\nTesting roundtrip for {test_val}:")
#     TestFP16Conversion().test_roundtrip_conversion(test_val)

#     # Test a tensor
#     print("\nTesting vector tensor:")
#     TestFP16Tensor().test_vector_tensor()

#     print("\nAll manual tests passed! Run with pytest for complete test suite.")
