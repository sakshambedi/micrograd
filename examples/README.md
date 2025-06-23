# MicroGrad C++ Kernel Examples

This directory contains examples demonstrating how to use the MicroGrad C++ kernel from Python.

## Prerequisites

Before running the examples, make sure you have:

1. **Built the C++ kernel**:

   ```bash
   ./build.sh --debug --tests
   ```

2. **Python 3.6+** installed

3. **Optional: NumPy** for array interface examples:

   ```bash
   pip install numpy
   ```

## Examples

### Basic Usage Example

Run the main example script:

```bash
python examples/build_example.py
```

This example demonstrates:

- Buffer creation with different data types
- Element access and buffer properties
- Type casting between different data types
- NumPy array interface (if NumPy is available)
- Edge cases and error handling

### Expected Output

```
MicroGrad C++ Kernel Example
============================
✓ Successfully imported cpu_kernel module

=== Testing Buffer Creation ===
Float buffer: Buffer(dtype=float32, size=4, data=[1.0f, 2.0f, 3.0f, 4.0f])
Int buffer: Buffer(dtype=int32, size=4, data=[1, 2, 3, 4])
Float buffer[0]: 1.0
Int buffer[1]: 2
Float buffer size: 4
Float buffer dtype: float32
✓ Test passed

=== Testing Buffer Casting ===
Original buffer: Buffer(dtype=float32, size=3, data=[1.5f, 2.7f, 3.2f])
Cast to int32: Buffer(dtype=int32, size=3, data=[1, 2, 3])
Cast to float64: Buffer(dtype=float64, size=3, data=[1.5, 2.7, 3.2])
✓ Test passed

=== Summary ===
Tests passed: 4/4
🎉 All tests passed!
```

## Using the C++ Kernel in Your Code

### Basic Buffer Operations

```python
import cpu_kernel

# Create buffers
a = cpu_kernel.Buffer([1, 2, 3, 4], "float32")
b = cpu_kernel.Buffer([5, 6, 7, 8], "float32")

# Access elements
print(a[0])  # 1.0

# Get buffer properties
print(a.size())  # 4
print(a.get_dtype())  # 'float32'

# Cast to different type
c = a.cast("int32")
```

### Working with NumPy

```python
import numpy as np
import cpu_kernel

# Create buffer
buffer = cpu_kernel.Buffer([1, 2, 3, 4], "float32")

# Convert to NumPy array
np_array = np.array(buffer)
print(np_array)  # [1. 2. 3. 4.]

# Convert NumPy array to buffer
np_data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
buffer_from_np = cpu_kernel.Buffer(np_data, "float32")
```

### Supported Data Types

The C++ kernel supports the following data types:

- **Integer types**: `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
- **Float types**: `float16`, `float32`, `float64`

### Error Handling

The kernel provides proper error handling:

```python
import cpu_kernel

try:
    # This will raise an error for invalid dtype
    buffer = cpu_kernel.Buffer([1, 2, 3], "invalid_type")
except Exception as e:
    print(f"Error: {e}")

try:
    # This will raise an error for out-of-bounds access
    buffer = cpu_kernel.Buffer([1, 2, 3], "float32")
    value = buffer[10]  # Index out of range
except Exception as e:
    print(f"Error: {e}")
```

## Troubleshooting

### Import Error

If you get an import error:

```
ImportError: No module named 'cpu_kernel'
```

Make sure you have:

1. Built the project: `./build.sh --debug --tests`
2. The compiled module exists in the `build/` directory
3. You're running the example from the project root

### Build Errors

If you encounter build errors, see the main `Agent.md` file for troubleshooting steps.

## Next Steps

After running the examples, you can:

1. **Explore the C++ code** in the `kernels/` directory
2. **Add new operations** following the patterns in `operations.cpp`
3. **Extend the Python interface** in `cpu_kernel.cpp`
4. **Write tests** in the `tests/kernels/` directory

For more detailed information, see the main `Agent.md` documentation.
