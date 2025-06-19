# micrograd: A Learning Journey into Deep Learning Frameworks

<p align="center">
  <img src="./imgs/micrograd-banner-m.jpeg" alt="Micrograd-Grad" width="600"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/sakshambedi/micrograd)

This project is a personal exploration and implementation of a simplified tensor library, inspired by [tinygrad](https://github.com/geohot/tinygrad) and [PyTorch](https://pytorch.org/). The primary goal is to understand the fundamental concepts and mechanics behind modern deep learning frameworks by building one from scratch.

## What is this?

**micrograd** is a Python-based library with a high-performance C++ backend that provides a `Tensor` object, similar to those found in NumPy, PyTorch, or TensorFlow. It aims to replicate some of the core functionalities, particularly focusing on automatic differentiation and tensor operations, to gain a deeper insight into their inner workings.

This is an educational project. While the aim is to create functional components, it's not intended to be a production-ready deep learning library.

## Current Features

As of now, `micrograd` supports:

### High-Performance Backend

- **C++ extension with Eigen integration:**
  - Vectorized operations using Eigen's SIMD capabilities
  - Efficient memory management with contiguous storage
  - Fast numeric operations through native C++ implementation
  - Cross-platform support via pybind11 bindings
- **Multiple data type support through variant-based implementation:**
  - Seamless handling of all numeric types in a single backend
  - Zero-copy buffer protocol implementation for interoperability

### Tensor Creation and Data Types

- **Multiple creation methods:**
  - Python lists or tuples: `Tensor([1, 2, 3])`
  - Single values (scalars): `Tensor(5)` or `Tensor(3.14)`
  - Empty tensors: `Tensor(None)` (creates a scalar tensor with value 0)
- **Comprehensive data type support:**
  - `float32`, `float16`, `float64` for floating-point numbers
  - `int32`, `int16`, `int8` for signed integers
  - `uint16`, `uint8` for unsigned integers
  - `bool` for boolean values
  - Native `float16` support through Eigen
- **Factory methods:**
  - `Tensor.ones((2, 3))` - Create tensors filled with ones
  - `Tensor.zeros((2, 3))` - Create tensors filled with zeros
  - `Tensor.full((2, 3), value)` - Create tensors filled with a specific value
  - `Tensor.arange(start, end, step)` - Create 1D tensors with sequential values
  - `Tensor.randn(*shape)` - Create tensors with random normal distribution
- **Automatic shape inference from nested structures**

### Mathematical Operations

- **Binary arithmetic operations:**
  - Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`), Power (`**`)
  - All operations support element-wise computation between tensors of the same shape
  - High-performance implementation using Eigen when tensors are contiguous
  - Proper error handling for shape mismatches and division by zero
- **Unary operations:**
  - Negation (`-tensor`)
  - Support for double negation and chaining operations
- **Reduction operations:**
  - `Tensor.sum()` with dtype preservation and gradient tracking
  - `Tensor.mean()` (planned implementation)
- **Advanced operations:**
  - Element-wise operations with proper type casting and upsampling
  - Special value handling (infinity, NaN, subnormal numbers)

### Memory Management and Performance

- **C++ buffer management system:**
  - Efficient memory allocation through Eigen's aligned allocator
  - Direct access through buffer protocol
  - Variant-based storage for flexible data type handling
- **Efficient storage:**
  - Contiguous memory layout using native C++ arrays
  - Zero-copy operations through views and strides
  - Memory sharing between Python and C++ without overhead

### Shape Manipulation and Views

- **Reshaping operations:**
  - `view(*shape)` / `reshape(*shape)` for changing tensor shapes
  - `transpose(dim0, dim1)` for swapping dimensions
  - `Tensor.T(tensor)` static method for 2D tensor transpose
  - `Tensor.permute(tensor, *indices)` for arbitrary dimension reordering
- **Advanced indexing:**
  - Multi-dimensional indexing: `tensor[0, 1]` for 2D tensors
  - Negative indexing support: `tensor[-1]`
  - Assignment through indexing: `tensor[0, 1] = value`
- **Stride manipulation:**
  - Direct access to tensor strides via `stride()` method
  - Non-contiguous tensor support through advanced stride handling
  - Automatic contiguity management

### Automatic Differentiation System

- **Gradient tracking:**
  - `requires_grad` attribute for enabling gradient computation
  - `grad` attribute for storing computed gradients
  - `grad_fn` attribute tracking computation history
- **Function-based autograd system:**
  - Base `Function` class with forward/backward methods
  - Context saving mechanism for backpropagation
  - Implemented operations: Add, Sub, Mul, Div, Pow, Neg (forward pass complete)
  - Thread-local autograd state management
- **Gradient propagation:**
  - Automatic gradient requirement propagation through operations
  - Support for computational graph construction

### Data Conversion and Interoperability

- **NumPy integration:**
  - `to_numpy()` method for seamless NumPy array conversion
  - Automatic shape and dtype preservation
- **String representations:**
  - Comprehensive `__repr__` showing shape, dtype, device, and data
  - Clean `__str__` for data visualization
  - Nested list representation matching tensor structure

### Device Support Framework

- **Device abstraction:**
  - Device class for future multi-device support
  - `device` attribute on tensors
  - Foundation for GPU computation (future work)

### Error Handling and Edge Cases

- **Robust error handling:**
  - Shape mismatch detection and reporting
  - Division by zero handling (returns infinity following NumPy behavior)
  - Index out of bounds protection
  - Inconsistent tensor shape validation
- **Special value support:**
  - Proper handling of infinity, negative infinity, and NaN
  - Zero-size tensor operations
  - Subnormal number handling in float16

## Test Coverage

The project includes comprehensive test coverage across multiple domains:

### Test Statistics

- **50+ test cases** covering core functionality
- **100% test pass rate** ensuring reliability
- **Performance benchmarks** for memory and computation efficiency

### Test Categories

- **Binary Operations** (`binary_ops_test.py`): Addition, subtraction, multiplication, division, power operations
- **Unary Operations** (`unary_ops_test.py`): Negation, double negation, special values
- **Tensor Fundamentals** (`tensor_test.py`): Creation, indexing, reshaping, transposition, permutation
- **Buffer Management** (`test_buffer.py`): C++ buffer handling, memory efficiency, performance optimization
- **Data Type Handling**: Float16, integer types, boolean values, type conversions
- **Edge Cases**: Large tensors, special values, concurrent operations
- **C++ Extensions**: Eigen integration, pybind11 bindings, buffer protocol implementation

## Project Structure

```
micrograd/
├── grad/
│   ├── __init__.py           # Package initialization
│   ├── tensor.py            # Core tensor implementation
│   ├── dtype.py             # Data type definitions and utilities
│   ├── buffer.py            # Advanced buffer management with C++ bindings
│   ├── device.py            # Device abstraction layer
│   ├── autograd/
│   │   ├── __init__.py      # Autograd package initialization
│   │   ├── function.py      # Base Function class for autograd
│   │   └── ops.py           # Mathematical operation implementations
│   └── utils/
│       ├── __init__.py      # Utils package initialization
│       ├── misc.py          # Miscellaneous helper functions
│       └── constants.py     # System constants and feature detection
├── kernels/
│   ├── __init__.py          # Kernels package initialization
│   ├── cpu_kernel.cpp       # C++ implementation of buffer operations with Eigen
│   └── cpu_kernel.h         # Header file for C++ kernel implementations
├── tests/
│   ├── __init__.py          # Test package initialization
│   ├── binary_ops_test.py   # Comprehensive binary operation tests
│   ├── unary_ops_test.py    # Unary operation tests
│   ├── tensor_test.py       # Core tensor functionality tests
│   └── test_buffer.py       # Buffer management and performance tests
├── CMakeLists.txt           # CMake configuration for C++ extensions
├── setup.py                 # Package setup and build configuration
└── README.md                # Project documentation
```

## Performance Optimizations

- **Eigen-powered vectorized operations** for significant performance improvements
- **SIMD instructions** automatically leveraged through Eigen
- **Contiguous memory access** patterns for optimal cache performance
- **Zero-copy views** for reshape and transpose operations where possible
- **C++/Python interoperability** with minimal overhead through pybind11

## Installation & Build

To build and install the package:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sakshambedi/micrograd.git
   cd micrograd
   ```

2. **Install dependencies and build C++ extensions:**

   ```bash
   # Install prerequisites
   ./setup_prerequisite.sh

   # Build C++ extensions
   pip install -e .
   ```

3. **Run the comprehensive test suite:**

   ```bash
   pytest -v
   # Expected output: 50+ tests passed
   ```

## How to Use / Explore

```python
from grad.tensor import Tensor
from grad.dtype import dtypes

# Advanced tensor creation
a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, requires_grad=True)
b = Tensor.randn(2, 3, dtype=dtypes.float32, requires_grad=True)

# Complex operations
c = (a * b + Tensor.ones((2, 3))) ** 2
print(f"Result shape: {c.shape}, requires_grad: {c.requires_grad}")

# Advanced shape manipulation
d = Tensor.arange(24).view(2, 3, 4)
e = Tensor.permute(d, 2, 0, 1)  # Reorder dimensions
f = Tensor.T(a)  # Transpose for 2D tensors

# Memory-efficient operations
large_tensor = Tensor.zeros((1000, 1000), dtype=dtypes.float16)
reshaped = large_tensor.view(1000000)  # Zero-copy reshape

# Data type conversions and NumPy integration
numpy_array = c.to_numpy()
print(f"Converted to NumPy: {type(numpy_array)}")
```

## Future Plans

### Short-term Goals

- **Complete Eigen-powered operations:**
  - Implement remaining vector operations using C++ backend
  - Add matrix multiplication with Eigen optimizations
  - Implement broadcasting support for tensors of different shapes
  - Complete backward pass implementations for all operations

### Medium-term Goals

- **Neural network components:**
  - Activation functions (ReLU, Sigmoid, Tanh, GELU)
  - Loss functions (MSE, Cross-Entropy, BCE)
  - Basic optimizers (SGD, Adam, RMSprop)
  - Layer abstractions (Linear, Convolution)
- **Performance optimizations:**
  - Just-in-time compilation for computational graphs
  - Multi-threading for large tensor operations
  - Memory pooling and reuse strategies

### Long-term Goals

- **GPU acceleration:**
  - CUDA backend for GPU computation
  - OpenCL support for cross-platform GPU acceleration
  - Automatic CPU/GPU memory management
- **Advanced features:**
  - Dynamic computational graphs
  - Model serialization and loading
  - Distributed computing support
  - Integration with existing ML ecosystems

## Contributing

Since this is primarily a learning project, direct contributions might not be the focus. However, if you're also on a similar learning path, feel free to:

- Fork the repository and experiment with your own implementations
- Suggest improvements or point out areas for learning via Issues
- Share resources or insights that could be helpful
- Submit test cases for edge cases or performance scenarios

## Performance Benchmarks

Recent performance improvements through Eigen integration:

- **Operation speed**: Up to 10x faster for elementwise operations on large tensors
- **Memory efficiency**: Aligned memory allocation for optimal SIMD operations
- **Thread safety**: Full concurrent operation support
- **Vectorization**: Automatic SIMD instruction utilization on supported platforms

### Building the SIMD backend

```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
python -m pip install ..
```

On an M2 laptop, adding one million `float32` elements takes around **4 ms**
with the new xsimd kernels compared to **20 ms** using the old interpreter
path.

## License

This project is available under the MIT License.

---
