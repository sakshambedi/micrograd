// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>
#include <string>

// Test DType conversions for all supported data types
TEST(DTypeTest, BasicTypeConversions) {
  // EXPECT_EQ(dtype_from_string("bool"), DType::BOOL);
  EXPECT_EQ(dtype_from_string("int8"), DType::INT8);
  EXPECT_EQ(dtype_from_string("uint8"), DType::UINT8);
  EXPECT_EQ(dtype_from_string("int16"), DType::INT16);
  EXPECT_EQ(dtype_from_string("uint16"), DType::UINT16);
  EXPECT_EQ(dtype_from_string("int32"), DType::INT32);
  EXPECT_EQ(dtype_from_string("uint32"), DType::UINT32);
  EXPECT_EQ(dtype_from_string("int64"), DType::INT64);
  EXPECT_EQ(dtype_from_string("uint64"), DType::UINT64);
  EXPECT_EQ(dtype_from_string("float16"), DType::FLOAT16);
  EXPECT_EQ(dtype_from_string("float32"), DType::FLOAT32);
  EXPECT_EQ(dtype_from_string("float64"), DType::FLOAT64);
}

// Test round-trip conversions between string and enum
// TEST(DTypeTest, RoundTripConversions) {
//   for (const auto &type_name :
//        {"bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
//        "int64",
//         "uint64", "float16", "float32", "float64"}) {
//     DType dtype = dtype_from_string(type_name);
//     std::string str = dtype_to_string(dtype);
//     EXPECT_EQ(str, type_name);
//   }
// }

// Test error handling for unknown types
TEST(DTypeTest, UnknownTypes) {
  EXPECT_THROW(dtype_from_string("complex64"), std::runtime_error);
  EXPECT_THROW(dtype_from_string("complex128"), std::runtime_error);
  EXPECT_THROW(dtype_from_string("string"), std::runtime_error);
  EXPECT_THROW(dtype_from_string("object"), std::runtime_error);
  EXPECT_THROW(dtype_from_string(""), std::runtime_error);
  EXPECT_THROW(dtype_from_string("x"), std::runtime_error);
}

// Test case sensitivity
TEST(DTypeTest, CaseSensitivity) {
  EXPECT_THROW(dtype_from_string("FLOAT32"), std::runtime_error);
  EXPECT_THROW(dtype_from_string("Float32"), std::runtime_error);
  EXPECT_THROW(dtype_from_string("BOOL"), std::runtime_error);
}

// Test whitespace handling
TEST(DTypeTest, WhitespaceHandling) {
  EXPECT_THROW(dtype_from_string(" float32"), std::runtime_error);
  EXPECT_THROW(dtype_from_string("float32 "), std::runtime_error);
  EXPECT_THROW(dtype_from_string("float 32"), std::runtime_error);
}

// Test dtype_to_string function
TEST(DTypeTest, DTypeToString) {
  // EXPECT_EQ(dtype_to_string(DType::BOOL), "bool");
  EXPECT_EQ(dtype_to_string(DType::INT8), "int8");
  EXPECT_EQ(dtype_to_string(DType::UINT8), "uint8");
  EXPECT_EQ(dtype_to_string(DType::INT16), "int16");
  EXPECT_EQ(dtype_to_string(DType::UINT16), "uint16");
  EXPECT_EQ(dtype_to_string(DType::INT32), "int32");
  EXPECT_EQ(dtype_to_string(DType::UINT32), "uint32");
  EXPECT_EQ(dtype_to_string(DType::INT64), "int64");
  EXPECT_EQ(dtype_to_string(DType::UINT64), "uint64");
  EXPECT_EQ(dtype_to_string(DType::FLOAT16), "float16");
  EXPECT_EQ(dtype_to_string(DType::FLOAT32), "float32");
  EXPECT_EQ(dtype_to_string(DType::FLOAT64), "float64");
}
