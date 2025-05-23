from __future__ import annotations


def tensor_stride(shape) -> tuple[int, ...]:
    """
    O(len(shape)) time, O(1) extra memory.
    pytorch stride : https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html
    """
    if not shape:
        return ()

    acc = 1
    stride = []
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    stride.reverse()
    return tuple(stride)
