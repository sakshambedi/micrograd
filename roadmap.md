# Breaking down weekly tasks

The user wants a more detailed version of the 6-8 week plan, splitting it into daily tasks for each week. The general plan is: Week 1-2 focus on foundations, Week 3-4 on the autograd engine, Week 5 on optimizers and the training loop, Week 6 on utilities and packaging, and Week 7-8 for extensions. They haven’t specified working days, but I'll assume they mean 5 days per week, considering their 10-15 hours per week. Still, if their schedule is more flexible, I might include weekends.

Here’s a detailed 6-week plan broken into 5-day (Mon–Fri) sprints. Adjust days as needed if you work weekends or have other commitments.

## Week 1 (Foundations, core Tensor + basic ops)

- **Day 1, Repo & Tensor shell**
  • Initialize GitHub repo “homemade-grad”
  • Create `grad/tensor.py` with empty `Tensor` class (wraps NumPy array)
  • Sketch `Function` base class (holds `forward`, `backward`, `prev`)

- **Day 2, Basic data storage & grad placeholders**
  • Implement `Tensor.data` and `Tensor.requires_grad`
  • Add `Tensor.grad` attribute, zero-init logic
  • Write simple tests for creation, data access

- **Day 3, Add & Multiply ops**
  • In `ops/add.py`, define `Add` forward/backward
  • In `ops/mul.py`, define `Mul` forward/backward
  • Hook them into `Tensor.__add__` and `Tensor.__mul__`

- **Day 4, Testing scalar ops**
  • Write `tests/test_ops.py` to compare forward results vs NumPy
  • Add numerical-gradient checks (finite differences) for add, mul

- **Day 5, Matmul & reshape**
  • Implement `MatMul` op in `ops/matmul.py`
  • Implement `Reshape` op in `ops/reshape.py`
  • Add basic tests for matrix multiplication, reshape behavior

## Week 2 (Autograd engine & simple graph)

- **Day 1, Graph tracking**
  • In `Tensor`, store `.grad_fn` (the Function that created it)
  • Populate `Function.prev` links to input Tensors

- **Day 2, Backward pass traversal**
  • Implement `Tensor.backward()` to walk graph in reverse topological order
  • Accumulate gradients into each `.grad`

- **Day 3, Test backprop on scalars**
  • Create simple functions (e.g. $z = x*y + x$)
  • Verify analytic vs numeric gradients

- **Day 4, Broadcast & shape-aware gradient logic**
  • Handle broadcasting cases in backward (sum-reduce gradients)
  • Add tests for broadcasted add or mul

- **Day 5, Mini-MLP smoke test**
  • Build a 2-layer MLP in `examples/mlp_test.py` using your Tensor and ops
  • Run forward/backward on random data to ensure no crashes

---

## Week 3 (Optimizers & training loop)

- **Day 1, Optimizer base & SGD**
  • Create `optim/optimizer.py` with `Optimizer` base class
  • Implement `SGD` in `optim/sgd.py`

- **Day 2, Adam optimizer**
  • Implement `Adam` (moment estimates, bias correction) in `optim/adam.py`
  • Write tests checking parameter updates on a simple quadratic loss

- **Day 3, Training loop scaffold**
  • In `examples/train_loop.py`, write generic loop:

  1. zero grads, 2) forward, 3) backward, 4) optimizer.step()

- **Day 4, Toy dataset integration**
  • Load small NumPy toy dataset (e.g. blobs or tiny MNIST subset)
  • Hook it into your training loop

- **Day 5, End-to-end verify**
  • Train MLP for a few epochs, log loss decrease
  • Add tests to ensure loss decreases on a simple linear regression

---

## Week 4 (Utilities, packaging & docs)

- **Day 1, Data loader & batching**
  • Create `grad/utils/data.py` with a basic `DataLoader` (shuffle, batch)
  • Test iterator yields correct shapes

- **Day 2, Learning-rate scheduler**
  • Implement step-LR scheduler in `grad/utils/lr_scheduler.py`
  • Write tests for learning-rate decay behavior

- **Day 3, Checkpointing & config**
  • Add save/load state (model params + optimizer state)
  • Example in `examples/checkpoint.py`

- **Day 4, Packaging setup**
  • Write `setup.py` (or `pyproject.toml`) with metadata, dependencies
  • Ensure `grad/` is discoverable as a package

- **Day 5, README & minimal docs**
  • Draft usage examples in `README.md` (installation, basic code snippets)
  • Add badges for Python version, build status (placeholder)

---

## Week 5 (Extensions for NLP basics)

- **Day 1, Embedding layer**
  • In `grad/utils/nlp_layers.py`, implement `Embedding` (lookup + gradient)
  • Test forward/backward on small vocabulary

- **Day 2, SimpleRNN block**
  • Build `SimpleRNNCell` (single-step) with hidden state gradient logic
  • Write step-through tests

- **Day 3, Sequence module**
  • Wrap `SimpleRNNCell` into full `RNN` over sequence dimension
  • Test on toy sequence loss

- **Day 4, Loss functions for NLP**
  • Implement `CrossEntropyLoss` for classification
  • Add tests for loss value and gradient

- **Day 5, NLP example script**
  • In `examples/nlp_demo.py`, load tiny text, tokenize, train next-token task
  • Verify loss decreases

---

## Week 6 (CI, release, optional GPU & advanced)

- **Day 1, GitHub Actions CI**
  • Create `.github/workflows/ci.yml` to run `pytest` on push/PR
  • Add linting step (flake8 or mypy)

- **Day 2, Coverage & badges**
  • Integrate `pytest-cov`, upload badge to README
  • Ensure coverage ≥ 80%

- **Day 3, Example validations**
  • In CI, run your `examples/*.py` to catch regressions
  • Add logs or smoke tests

- **Day 4, Release v0.1.0**
  • Tag `v0.1.0` in GitHub, update version in `setup.py`
  • Create GitHub Release notes

- **Day 5, (Optional) GPU backend**
  • Experiment with CuPy in `tensor.py` to push `.data` to GPU
  • Write one minimal test for a GPU add or mul

---

# Week 2 Agenda : Your `Tensor` class in `micrograd/grad/tensor.py` already covers a lot of the foundational features needed for a minimal deep learning (DL) framework, including:

- Basic tensor creation (`zeros`, `ones`, `full`)
- Shape manipulation (`view`, `reshape`, `transpose`, `permute`)
- Arithmetic operations (`+`, `-`, `*`, `/`, `**`, negation)
- Indexing and assignment
- Buffer and dtype management
- Gradients and autograd hooks (`requires_grad`, `grad`, `grad_fn`)
- Device support (though only `"cpu"` is mentioned)
- Conversion to NumPy

However, to approach the core functionality of a simple PyTorch-like DL framework, here are some **key missing or incomplete features** and suggestions for improvement:

## 1. **Random Tensor Initialization**

- `arange` and `randn` are stubbed (`...`). You need:
  - `arange`: Like `np.arange` or `torch.arange`
  - `randn`: Like `np.random.randn` or `torch.randn`
- Also consider `rand` (uniform random) and `eye` (identity matrix).

## 2. **Broadcasting**

- PyTorch supports broadcasting for arithmetic operations. Your current binary ops (`__add__`, etc.) do not mention broadcasting.
- Implement broadcasting logic for shape compatibility in binary operations.

## 3. **Reduction Operations**

- Common reductions: `sum`, `mean`, `min`, `max`, `argmax`, `argmin`, etc.
- Support for reduction over specific axes.

## 4. **Slicing and Advanced Indexing**

- Your `__getitem__` and `__setitem__` only support basic indexing.
- Add support for slicing (`tensor[1:3, ...]`), boolean masks, and advanced indexing.

## 5. **Autograd/Backward**

- While you have `requires_grad`, `grad`, and `grad_fn`, you need:
  - A `backward()` method to trigger backpropagation.
  - A mechanism to accumulate gradients and traverse the computation graph.
- PyTorch’s `Tensor.backward()` is a core feature.

## 6. **In-place Operations**

- PyTorch supports in-place ops (e.g., `add_`, `mul_`).
- Consider supporting or explicitly forbidding them for safety.

## 7. **Device Management**

- Only `"cpu"` is supported. For a minimal framework, this is fine, but PyTorch supports `"cuda"` (GPU).
- If you want to extend, consider a device abstraction layer.

## 8. **Data Type Conversion**

- Methods like `.float()`, `.int()`, `.to(dtype=...)`, `.to(device=...)`.

## 9. **Gradient Control**

- Methods like `.detach()`, `.requires_grad_()`, `.zero_()` for gradients.

## 10. **Module/Layer Abstraction**

- PyTorch has `nn.Module` for neural network layers.
- You’ll want a base `Module` class and some basic layers (e.g., `Linear`, `ReLU`).

## 11. **Loss Functions**

- Basic losses: `MSELoss`, `CrossEntropyLoss`, etc.

## 12. **Optimizers**

- SGD, Adam, etc., for parameter updates.

## 13. **Serialization**

- Saving/loading tensors and models (`state_dict`, `load_state_dict`).

## 14. **Utilities**

- Shape/size introspection: `.size()`, `.numel()`, `.ndim`, etc.
- Printing and pretty formatting.

### **Summary Table**

| Feature Area      | Status in Your Code | Needed for PyTorch-like DL    |
| ----------------- | ------------------- | ----------------------------- |
| Tensor creation   | Partial             | Add `arange`, `randn`, etc.   |
| Shape ops         | Good                |                               |
| Arithmetic ops    | Good                | Add broadcasting              |
| Reductions        | Missing             | Add `sum`, `mean`, etc.       |
| Indexing/slicing  | Basic               | Add slicing, advanced         |
| Autograd/backward | Partial             | Add `backward()`              |
| In-place ops      | Missing             | Add or forbid                 |
| Device support    | Basic               | Optional: add GPU             |
| Dtype conversion  | Missing             | Add `.to()`, `.float()`, etc. |
| Gradient control  | Partial             | Add `.detach()`, `.zero_()`   |
| Module/layers     | Missing             | Add `Module`, layers          |
| Loss functions    | Missing             | Add basic losses              |
| Optimizers        | Missing             | Add SGD, Adam                 |
| Serialization     | Missing             | Add save/load                 |
| Utilities         | Partial             | Add `.size()`, `.numel()`     |
