import numpy as np
from grad.tensor import Tensor


def test_tensor_add_basic():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    np.testing.assert_array_equal(c._to_nested(), [5, 7, 9])
