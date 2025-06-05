## Background

We currently use Python `array.array` to store raw buffers in `Buffer` and `BufferPool`.
To unlock both higher performance (SIMD, vectorization) and seamless NumPy/PyTorch interop (via buffer protocol), we want to migrate to a C++ extension that uses Eigen underneath.

Our existing Python dtype definitions live in `micrograd/grad/dtype.py`.  We need to map each of those to a corresponding C++ scalar type and wrap them in an `Eigen::VectorX<...>` or aligned buffer.

## Goals

1. Define a clear mapping from Python `DType` fmt → C++ type → Eigen container.
2. Implement a thin C++ class (e.g. `class EigenBuffer<T>`) that:
   - Allocates aligned contiguous memory
   - Exposes `.data()`, `.size()`, `.resize()`, `.fill()`
   - Implements the Python buffer protocol
3. Wire up `Buffer` and `BufferPool` to call into our new extension instead of `array.array`.
4. Preserve current pooling heuristics, thread-safety, and fallback mode if C++ extension fails to load.

## Proposed Mapping

| Python fmt | C++ type       | Eigen type                             |
|------------|----------------|----------------------------------------|
| `?`        | `bool`         | `Eigen::Matrix<bool, Dynamic, 1>`      |
| `b`/`B`    | `int8_t`/`uint8_t`  | `Eigen::Matrix<int8_t, Dynamic,1>` / `Eigen::Matrix<uint8_t, Dynamic,1>` |
| `h`/`H`    | `int16_t`/`uint16_t` | … similar …                           |
| `i`/`I`    | `int32_t`/`uint32_t` | …                                    |
| `q`/`Q`    | `int64_t`/`uint64_t` | …                                    |
| `e`        | `Eigen::half`  | `Eigen::Matrix<Eigen::half, Dynamic,1>` |
| `f`        | `float`        | `Eigen::VectorXf`                     |
| `d`        | `double`       | `Eigen::VectorXd`                     |

## Tasks

- [ ] Create `buffer_ext/` C++ extension scaffold (CMake or pybind11).
- [ ] Implement `EigenBuffer<T>` template class with:
  - `resize()`, `setZero()`, `setConstant()`
  - `T* data()` access
  - Python buffer‐protocol support
- [ ] Expose factory functions: `get_buffer(fmt, size)`, `release_buffer(ptr)`
- [ ] Integrate into `Buffer` & `BufferPool` in Python.
- [ ] Write unit tests (dtype roundtrips, pooling stats, thread-safety).
- [ ] Benchmarks vs. `array.array` baseline.
