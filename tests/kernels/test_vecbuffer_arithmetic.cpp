// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.

#include "../../kernels/cpu_kernel.h"
#include <gtest/gtest.h>

TEST(VecBufferArithmeticTest, Addition) {

  {
    VecBuffer<float> vec1(4);
    VecBuffer<float> vec2(4);

    for (size_t i = 0; i < 4; ++i) {
      vec1[i] = static_cast<float>(i);
      vec2[i] = static_cast<float>(2 * i + 1);
    }

    vec1 = vec1 + vec2;

    for (size_t i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(vec1[i], static_cast<float>(i + (2 * i + 1)));
      EXPECT_FLOAT_EQ(vec2[i], static_cast<float>(2 * i + 1)); // unchanged
    }
  }
}

TEST(VecBufferArithmeticTest, Subtraction) {

  {
    VecBuffer<float> vec1(4);
    VecBuffer<float> vec2(4);

    for (size_t i = 0; i < 4; ++i) {
      vec1[i] = static_cast<float>(i);
      vec2[i] = static_cast<float>(2 * i + 1);
    }

    vec2 = vec2 - vec1;

    for (size_t i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(vec2[i], static_cast<float>(((2 * i) + 1) - i));
      EXPECT_FLOAT_EQ(vec1[i], static_cast<float>(i)); // unchanged
    }
  }
}

TEST(VecBufferArithmeticTest, InPlaceAddition) {

  {
    VecBuffer<float> vec1(4);
    VecBuffer<float> vec2(4);

    for (size_t i = 0; i < 4; ++i) {
      vec1[i] = static_cast<float>(i);
      vec2[i] = static_cast<float>(2 * i + 1);
    }

    vec1 += vec2;

    for (size_t i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(vec1[i], static_cast<float>(i + (2 * i + 1)));
      EXPECT_FLOAT_EQ(vec2[i], static_cast<float>(2 * i + 1)); // unchanged
    }
  }

  // double case
  {
    VecBuffer<double> vec1(3);
    VecBuffer<double> vec2(3);

    vec1[0] = 1.5;
    vec1[1] = 2.5;
    vec1[2] = 3.5;

    vec2[0] = 0.5;
    vec2[1] = 1.0;
    vec2[2] = 1.5;

    vec1 += vec2;

    EXPECT_DOUBLE_EQ(vec1[0], 2.0);
    EXPECT_DOUBLE_EQ(vec1[1], 3.5);
    EXPECT_DOUBLE_EQ(vec1[2], 5.0);
  }

  // case :  ints
  {
    VecBuffer<int> vec1(3);
    VecBuffer<int> vec2(3);

    vec1[0] = 10;
    vec1[1] = 20;
    vec1[2] = 30;

    vec2[0] = 1;
    vec2[1] = 2;
    vec2[2] = 3;

    vec1 += vec2;

    EXPECT_EQ(vec1[0], 11);
    EXPECT_EQ(vec1[1], 22);
    EXPECT_EQ(vec1[2], 33);
  }

  // case: FP16
  {
    VecBuffer<Eigen::half> vec1(3);
    VecBuffer<Eigen::half> vec2(3);

    vec1[0] = Eigen::half(1.0f);
    vec1[1] = Eigen::half(2.0f);
    vec1[2] = Eigen::half(3.0f);

    vec2[0] = Eigen::half(0.5f);
    vec2[1] = Eigen::half(1.0f);
    vec2[2] = Eigen::half(1.5f);

    vec1 += vec2;

    EXPECT_NEAR(static_cast<float>(vec1[0]), 1.5f, 0.01f);
    EXPECT_NEAR(static_cast<float>(vec1[1]), 3.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(vec1[2]), 4.5f, 0.01f);
  }
}

TEST(VecBufferArithmeticTest, InPlaceSubtraction) {

  {
    VecBuffer<float> vec1(4);
    VecBuffer<float> vec2(4);

    for (size_t i = 0; i < 4; ++i) {
      vec1[i] = static_cast<float>(10 + i);
      vec2[i] = static_cast<float>(i);
    }

    vec1 -= vec2;

    for (size_t i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(vec1[i], static_cast<float>(10));
      EXPECT_FLOAT_EQ(vec2[i], static_cast<float>(i));
    }
  }

  // case : double
  {
    VecBuffer<double> vec1(3);
    VecBuffer<double> vec2(3);

    vec1[0] = 5.5;
    vec1[1] = 7.5;
    vec1[2] = 9.5;

    vec2[0] = 0.5;
    vec2[1] = 1.5;
    vec2[2] = 2.5;

    vec1 -= vec2;

    EXPECT_DOUBLE_EQ(vec1[0], 5.0);
    EXPECT_DOUBLE_EQ(vec1[1], 6.0);
    EXPECT_DOUBLE_EQ(vec1[2], 7.0);
  }

  // case : int
  {
    VecBuffer<int> vec1(3);
    VecBuffer<int> vec2(3);

    vec1[0] = 10;
    vec1[1] = 20;
    vec1[2] = 30;

    vec2[0] = 5;
    vec2[1] = 10;
    vec2[2] = 15;

    vec1 -= vec2;

    EXPECT_EQ(vec1[0], 5);
    EXPECT_EQ(vec1[1], 10);
    EXPECT_EQ(vec1[2], 15);
  }

  // case: FP16
  {
    VecBuffer<Eigen::half> vec1(3);
    VecBuffer<Eigen::half> vec2(3);

    vec1[0] = Eigen::half(3.0f);
    vec1[1] = Eigen::half(5.0f);
    vec1[2] = Eigen::half(7.0f);

    vec2[0] = Eigen::half(1.0f);
    vec2[1] = Eigen::half(2.0f);
    vec2[2] = Eigen::half(3.0f);

    vec1 -= vec2;

    EXPECT_NEAR(static_cast<float>(vec1[0]), 2.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(vec1[1]), 3.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(vec1[2]), 4.0f, 0.01f);
  }
}

TEST(VecBufferArithmeticTest, EdgeCases) {

  {
    VecBuffer<float> vec1(0);
    VecBuffer<float> vec2(0);

    vec1 += vec2;
    vec1 -= vec2;

    EXPECT_EQ(vec1.size(), 0);
    EXPECT_EQ(vec2.size(), 0);
  }

  {
    VecBuffer<float> vec1(2);
    VecBuffer<float> vec2(2);

    vec1[0] = 10000.0f;
    vec1[1] = -10000.0f;

    vec2[0] = 5000.0f;
    vec2[1] = 2000.0f;

    VecBuffer<float> vecAdd = vec1;
    vecAdd += vec2;

    VecBuffer<float> vecSub = vec1;
    vecSub -= vec2;

    EXPECT_FLOAT_EQ(vecAdd[0], 15000.0f);
    EXPECT_FLOAT_EQ(vecAdd[1], -8000.0f);

    EXPECT_FLOAT_EQ(vecSub[0], 5000.0f);
    EXPECT_FLOAT_EQ(vecSub[1], -12000.0f);
  }
}

TEST(VecBufferArithmeticTest, ChainedOperations) {
  VecBuffer<float> vec1(3);
  VecBuffer<float> vec2(3);
  VecBuffer<float> vec3(3);

  for (size_t i = 0; i < 3; ++i) {
    vec1[i] = static_cast<float>(i);         // [0, 1, 2]
    vec2[i] = static_cast<float>(2 * i);     // [0, 2, 4]
    vec3[i] = static_cast<float>(i + 2 * i); // [0, 3, 6]
  }

  VecBuffer<float> result = (vec1 + vec2) - vec3;

  for (size_t i = 0; i < 3; ++i) {
    // vec1[i] + vec2[i] - vec3[i] = i + 2*i - (i + 2*i) = 0
    EXPECT_FLOAT_EQ(result[i], 0.0f);

    // Ensure original vectors are unchanged
    EXPECT_FLOAT_EQ(vec1[i], static_cast<float>(i));
    EXPECT_FLOAT_EQ(vec2[i], static_cast<float>(2 * i));
    EXPECT_FLOAT_EQ(vec3[i], static_cast<float>(i + 2 * i));
  }
}

TEST(VecBufferArithmeticTest, PreciseCalculation) {
  const size_t size = 5;
  VecBuffer<double> vec1(size);
  VecBuffer<double> vec2(size);

  vec1[0] = 1.0;
  vec1[1] = 2.0;
  vec1[2] = 3.0;
  vec1[3] = 4.0;
  vec1[4] = 5.0;

  vec2[0] = 0.1;
  vec2[1] = 0.2;
  vec2[2] = 0.3;
  vec2[3] = 0.4;
  vec2[4] = 0.5;

  VecBuffer<double> vecAdd = vec1;
  VecBuffer<double> vecSub = vec1;

  vecAdd += vec2;
  vecSub -= vec2;

  EXPECT_DOUBLE_EQ(vecAdd[0], 1.1);
  EXPECT_DOUBLE_EQ(vecAdd[1], 2.2);
  EXPECT_DOUBLE_EQ(vecAdd[2], 3.3);
  EXPECT_DOUBLE_EQ(vecAdd[3], 4.4);
  EXPECT_DOUBLE_EQ(vecAdd[4], 5.5);

  EXPECT_DOUBLE_EQ(vecSub[0], 0.9);
  EXPECT_DOUBLE_EQ(vecSub[1], 1.8);
  EXPECT_DOUBLE_EQ(vecSub[2], 2.7);
  EXPECT_DOUBLE_EQ(vecSub[3], 3.6);
  EXPECT_DOUBLE_EQ(vecSub[4], 4.5);
}
