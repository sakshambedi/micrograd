
import pytest

from grad.buffer import Buffer
from grad.dtype import dtypes


class TestBufferSharing:
    def test_share_reflects_changes(self):
        buf_a = Buffer(dtypes.int32, [1, 2, 3])
        buf_b = buf_a.share()

        assert buf_a.shares_storage_with(buf_b)
        buf_b[1] = 99
        assert buf_a[1] == 99

    def test_clone_independent(self):
        buf_a = Buffer(dtypes.int32, [1, 2, 3])
        buf_b = buf_a.clone()

        assert not buf_a.shares_storage_with(buf_b)
        buf_b[0] = 42
        assert buf_a[0] == 1


class TestTensorViewSharing:
    def test_view_shares_buffer(self):
        from grad.tensor import Tensor

        t = Tensor([1, 2, 3, 4], dtype=dtypes.int32)
        v = t.view(2, 2)

        assert t.storage is not v.storage  # different Buffer objects
        assert t.storage is not None and v.storage is not None
        assert t.storage.shares_storage_with(v.storage)
        v.storage[0] = 10
        assert t.storage[0] == 10

