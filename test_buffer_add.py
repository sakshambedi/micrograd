#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for Buffer add functionality."""

from grad.kernels import cpu_kernel
import unittest

Buffer = cpu_kernel.Buffer


class TestBufferAdd(unittest.TestCase):

    def test_add_same_type(self):
        # Test addition of two buffers with the same data type
        a = Buffer([1, 2, 3, 4, 5], "float32")
        b = Buffer([5, 4, 3, 2, 1], "float32")

        # Default output type (same as inputs)
        c = cpu_kernel.add(a, b)
        self.assertEqual(c.get_dtype(), "float32")
        self.assertEqual(c.size(), 5)
        for i in range(5):
            self.assertEqual(c[i], a[i] + b[i])

        # Test with __add__ operator
        d = a + b
        self.assertEqual(d.get_dtype(), "float32")
        for i in range(5):
            self.assertEqual(d[i], a[i] + b[i])

    def test_add_different_types(self):
        # Test addition of buffers with different data types
        a = Buffer([1, 2, 3], "int32")
        b = Buffer([0.5, 1.5, 2.5], "float32")

        # Result should be float32 (higher precision)
        c = cpu_kernel.add(a, b)
        self.assertEqual(c.get_dtype(), "float32")
        self.assertEqual(c[0], 1.5)
        self.assertEqual(c[1], 3.5)
        self.assertEqual(c[2], 5.5)

    def test_add_with_specified_output_type(self):
        # Test addition with explicit output type
        a = Buffer([1, 2, 3], "int32")
        b = Buffer([4, 5, 6], "int32")

        # Force result to be float64
        c = cpu_kernel.add(a, b, "float64")
        self.assertEqual(c.get_dtype(), "float64")
        for i in range(3):
            self.assertEqual(c[i], float(a[i] + b[i]))

    def test_add_mixed_precision(self):
        # Test with mixed precision types
        a = Buffer([1, 2, 3], "int8")
        b = Buffer([1000, 2000, 3000], "int32")

        # Should promote to int32
        c = cpu_kernel.add(a, b)
        self.assertEqual(c.get_dtype(), "int32")
        for i in range(3):
            self.assertEqual(c[i], a[i] + b[i])

    def test_add_float_and_int(self):
        # Test adding float and int
        a = Buffer([1, 2, 3], "int64")
        b = Buffer([0.5, 1.5, 2.5], "float64")

        # Should promote to float64
        c = cpu_kernel.add(a, b)
        self.assertEqual(c.get_dtype(), "float64")
        for i in range(3):
            self.assertEqual(c[i], a[i] + b[i])


if __name__ == "__main__":
    unittest.main()
