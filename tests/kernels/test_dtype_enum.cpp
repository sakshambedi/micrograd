// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>
#include <string>

// Test DTypeEnum conversions for all supported data types
TEST(DTypeEnumTest, AllTypeConversions) {
  EXPECT_EQ(get_dtype_enum("bool"), DTypeEnum::BOOL);
  EXPECT_EQ(get_dtype_enum("int8"), DTypeEnum::INT8);
  EXPECT_EQ(get_dtype_enum("uint8"), DTypeEnum::UINT8);
  EXPECT_EQ(get_dtype_enum("int16"), DTypeEnum::INT16);
  EXPECT_EQ(get_dtype_enum("uint16"), DTypeEnum::UINT16);
  EXPECT_EQ(get_dtype_enum("int32"), DTypeEnum::INT32);
  EXPECT_EQ(get_dtype_enum("uint32"), DTypeEnum::UINT32);
  EXPECT_EQ(get_dtype_enum("int64"), DTypeEnum::INT64);
  EXPECT_EQ(get_dtype_enum("uint64"), DTypeEnum::UINT64);
  EXPECT_EQ(get_dtype_enum("float16"), DTypeEnum::FLOAT16);
  EXPECT_EQ(get_dtype_enum("float32"), DTypeEnum::FLOAT32);
  EXPECT_EQ(get_dtype_enum("float64"), DTypeEnum::FLOAT64);

  EXPECT_EQ(get_dtype_enum("?"), DTypeEnum::BOOL);
  EXPECT_EQ(get_dtype_enum("b"), DTypeEnum::INT8);
  EXPECT_EQ(get_dtype_enum("B"), DTypeEnum::UINT8);
  EXPECT_EQ(get_dtype_enum("h"), DTypeEnum::INT16);
  EXPECT_EQ(get_dtype_enum("H"), DTypeEnum::UINT16);
  EXPECT_EQ(get_dtype_enum("i"), DTypeEnum::INT32);
  EXPECT_EQ(get_dtype_enum("I"), DTypeEnum::UINT32);
  EXPECT_EQ(get_dtype_enum("q"), DTypeEnum::INT64);
  EXPECT_EQ(get_dtype_enum("Q"), DTypeEnum::UINT64);
  EXPECT_EQ(get_dtype_enum("e"), DTypeEnum::FLOAT16);
  EXPECT_EQ(get_dtype_enum("f"), DTypeEnum::FLOAT32);
  EXPECT_EQ(get_dtype_enum("d"), DTypeEnum::FLOAT64);
}

TEST(DTypeEnumTest, UnknownTypes) {
  EXPECT_EQ(get_dtype_enum("complex64"), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum("complex128"), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum("string"), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum("object"), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum(""), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum("x"), DTypeEnum::UNKNOWN); // Non-existent code
}

TEST(DTypeEnumTest, CaseSensitivity) {
  EXPECT_NE(get_dtype_enum("FLOAT32"), DTypeEnum::FLOAT32);
  EXPECT_NE(get_dtype_enum("Float32"), DTypeEnum::FLOAT32);
  EXPECT_EQ(get_dtype_enum("FLOAT32"), DTypeEnum::UNKNOWN);
}

TEST(DTypeEnumTest, WhitespaceHandling) {
  EXPECT_EQ(get_dtype_enum(" float32"), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum("float32 "), DTypeEnum::UNKNOWN);
  EXPECT_EQ(get_dtype_enum("float 32"), DTypeEnum::UNKNOWN);
}
