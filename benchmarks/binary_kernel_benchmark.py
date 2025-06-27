import time
import numpy as np
from grad.autograd import operations
from grad.kernels import cpu_kernel

def run_bench(n=1_000_000):
    n = int(n)
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    buf_a = cpu_kernel.Buffer(a, "float32")
    buf_b = cpu_kernel.Buffer(b, "float32")
    start = time.time()
    result = operations.binary_op(buf_a, buf_b, operations.BinaryOpType.ADD, "float32")
    elapsed = time.time() - start
    print(f"size={n} elapsed={elapsed:.6f} sec")
    return elapsed

if __name__ == "__main__":
    run_bench()
