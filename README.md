# Petit-Grad: A Learning Journey into Deep Learning Frameworks

This project is a personal exploration and implementation of a simplified tensor library, inspired by [tinygrad](https://github.com/geohot/tinygrad) and [PyTorch](https://pytorch.org/). The primary goal is to understand the fundamental concepts and mechanics behind modern deep learning frameworks by building one from scratch (or, at least, a significant portion of it!).

## What is this?

`petit-grad` is a Python-based library that provides a `Tensor` object, similar to those found in NumPy, PyTorch, or TensorFlow. It aims to replicate some of the core functionalities, particularly focusing on automatic differentiation and tensor operations, to gain a deeper insight into their inner workings.

This is an educational project. While the aim is to create functional components, it's not intended to be a production-ready deep learning library.

## Current Features

As of now, `petit-grad` supports:

- **Tensor Creation:**
  - Creating tensors from Python lists or tuples.
  - Specifying `dtype` using the `dtypes` class (defaults to `dtypes.float32`). Supported dtypes include `float32`, `float16`, `int32`, `int8`, `bool`.
  - Creating tensors filled with ones (`Tensor.ones()`).
  - Creating tensors filled with zeros (`Tensor.zeros()`).
  - _Note: Creating empty tensors (`Tensor.empty()`) is not currently supported in this version._
- **Basic Mathematical Operations:**
  - Addition (`+`)
  - Multiplication (`*`)
  - Subtraction (`-`)
  - True Division (`/`)
  - Floor Division (`//`)
- **Operator Overloading:** These operations can be used directly with Python's arithmetic operators.
- **Broadcasting:** Operations between tensors of different (but compatible) shapes are supported, following NumPy-like broadcasting rules.
- **Internal Data Representation:**
  - Tensors store their data internally using Python's `memoryview` backed by `array.array` or `bytearray`, not NumPy arrays. This helps understand lower-level data handling.
- **Data Access:**
  - Tensor data can be viewed as nested Python lists via the `__repr__` method for inspection.
- **Reverse Operations:** Reverse versions of multiplication (`__rmul__`), true division (`__rtruediv__`), and floor division (`__rfloordiv__`) are implemented.
- **Zero-Size Tensor Handling:** Operations involving tensors with zero-sized dimensions are handled, generally following NumPy's behavior (e.g., returning empty tensors or handling division by zero appropriately for true division).
- **Division by Zero:**
  - **True Division (`/`):** Follows NumPy's behavior, resulting in `inf` or `nan` and potentially raising a `RuntimeWarning` (which can be caught/ignored).
  - **Floor Division (`//`):** Raises a `ZeroDivisionError` when division by zero is attempted, which is consistent with Python and NumPy.
- **Rudimentary `requires_grad` Tracking:** A `requires_grad` attribute is present, hinting at future automatic differentiation capabilities (currently basic propagation for binary ops).

## Project Structure

- `grad/tensor.py`: Contains the core `Tensor` class definition and its associated methods.
- `tests/mathops_test.py`: Pytest-based unit tests for the mathematical operations implemented in the `Tensor` class. This helps ensure correctness and mimics the testing practices of larger libraries.

## Goals & Learning Objectives

- Understand the internal data representation of tensors.
- Implement common tensor operations and their broadcasting rules.
- Grasp the concept of computational graphs (future work).
- Implement backpropagation and automatic differentiation (future work).
- Explore different device supports (e.g., CPU, GPU - very future work!).
- Appreciate the complexities and design decisions in mature frameworks like PyTorch and tinygrad.

## How to Use / Explore

1. **Clone the repository:**

   ```bash
   gh repo clone sakshambedi/petit-grad
   cd petit-grad
   ```

2. **Explore the code:**

   - Check out `grad/tensor.py` to see the `Tensor` implementation.
   - Run the tests using `pytest` in the root directory to see the operations in action:

     ```bash
     pytest
     ```

3. **Experiment:**
   You can open a Python interpreter in the project's root directory and play with the `Tensor` class:

   ```python
   from grad.tensor import Tensor
   from grad.dtype import dtypes

   # Basic Tensor Creation and Operation
   a = Tensor([1, 2, 3], dtype=dtypes.int32) # Specify dtype
   b = Tensor([4, 5, 6], dtype=dtypes.int32)
   c = a + b
   print(c)
   # Output: Tensor(shape=(3,), data=[5, 7, 9], device=cpu, dtype=int32, requires_grad=None)

   # Broadcasting example
   d = Tensor([[1,2],[3,4]], dtype=dtypes.float32)
   e = Tensor([10], dtype=dtypes.float32)
   f = d * e # Broadcasting
   print(f)
   # Output:
   # Tensor(shape=(2, 2), data=[[10.0, 20.0], [30.0, 40.0]], device=cpu, dtype=float32, requires_grad=None)

   # Using ones and zeros
   ones_tensor = Tensor.ones((2, 3), dtype=dtypes.float16)
   print(ones_tensor)
   # Output: Tensor(shape=(2, 3), data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=cpu, dtype=float16, requires_grad=None)

   zeros_tensor = Tensor.zeros((2, 2)) # Defaults to float32
   print(zeros_tensor)
   # Output: Tensor(shape=(2, 2), data=[[0.0, 0.0], [0.0, 0.0]], device=cpu, dtype=float32, requires_grad=None)
   ```

## Future Plans (Potential Learning Areas)

- **Automatic Differentiation Engine:**
  - Implement `_backward()` methods for operations.
  - Build a simple computational graph.
  - Implement the chain rule for gradient calculation.
- **More Operations:** Expand the range of supported mathematical and tensor manipulation operations (e.g., `sum`, `mean`, `max`, `pow`, `exp`, `log`, matrix multiplication, reshaping, transposing).
- **Activation Functions:** Implement common activation functions (ReLU, Sigmoid, Tanh).
- **Optimizers:** Basic optimizers like SGD.
- **Simple Neural Network Layers:** Linear layers.

## Contributing

Since this is primarily a learning project, direct contributions might not be the focus. However, if you're also on a similar learning path, feel free to:

- Fork the repository and experiment with your own implementations.
- Suggest improvements or point out areas for learning via Issues.
- Share resources or insights that could be helpful.

---

This project is a testament to the idea that the best way to understand complex systems is often to try and build them yourself!
