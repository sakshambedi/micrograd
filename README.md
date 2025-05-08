# TinyGrad-Homemade: A Learning Journey into Deep Learning Frameworks

This project is a personal exploration and implementation of a simplified tensor library, inspired by [tinygrad](https://github.com/geohot/tinygrad) and [PyTorch](https://pytorch.org/). The primary goal is to understand the fundamental concepts and mechanics behind modern deep learning frameworks by building one from scratch (or, at least, a significant portion of it!).

## What is this?

`tinygrad-homemade` is a Python-based library that provides a `Tensor` object, similar to those found in NumPy, PyTorch, or TensorFlow. It aims to replicate some of the core functionalities, particularly focusing on automatic differentiation and tensor operations, to gain a deeper insight into their inner workings.

This is an educational project. While the aim is to create functional components, it's not intended to be a production-ready deep learning library.

## Current Features

As of now, `tinygrad-homemade` supports:

- **Tensor Creation:**
  - Creating tensors from Python lists or NumPy arrays.
  - Specifying `dtype` (currently defaults to `np.float32`).
  - Creating empty tensors with `Tensor.empty()`.
- **Basic Mathematical Operations:**
  - Addition (`+`)
  - Multiplication (`*`)
  - Subtraction (`-`)
  - True Division (`/`)
  - Floor Division (`//`)
- **Operator Overloading:** These operations can be used directly with Python's arithmetic operators.
- **Broadcasting:** Operations between tensors of different (but compatible) shapes are supported, following NumPy-like broadcasting rules.
- **NumPy Interoperability:**
  - Tensors store their data as NumPy arrays (`self.data`).
  - Easy conversion to NumPy arrays using the `.numpy()` method.
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
    gh repo clone sakshambedi/tinygrad-homemade
    cd tinygrad-homemade
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
    import numpy as np

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    print(c.numpy()) # Output: [5. 7. 9.]

    d = Tensor([[1,2],[3,4]])
    e = Tensor([10])
    f = d * e # Broadcasting
    print(f.numpy())
    # Output:
    # [[10. 20.]
    #  [30. 40.]]

    # Division by zero (true division)
    t_arr = Tensor([1.0, 2.0, 3.0])
    t_zero = Tensor([0.0])
    # result = t_arr / t_zero # This would typically result in [inf, inf, inf] or similar
    # print(result.numpy())
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
