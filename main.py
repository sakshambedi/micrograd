# %%

from grad.dtype import dtypes
from grad.tensor import Tensor

a = Tensor([[1, 2, 3, 4], [21, 123, 123, 123]], dtype=dtypes.fp16)
print(a)

# %%
b = Tensor([2, 3, 4, 5], dtype=dtypes.int32)
print(b)

# # %%
c = Tensor.ones((2, 2), dtype=dtypes.int8)
print(c)

# %%
d = Tensor.zeros((5, 2), dtype=dtypes.float16)
print(d)
