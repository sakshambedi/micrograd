from __future__ import annotations

from itertools import product as iter_product


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


def _nd_indices(shape):
    """
    Yields all possible n-dimensional indices for a given shape.
    Example: shape = (2, 3) yields (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    """

    return iter_product(*(range(s) for s in shape))


def broadcast_shape(shp1: tuple[int, ...], shp2: tuple[int, ...]) -> list[int]:
    return [1, 2]


def can_broadcast(shp1: tuple[int, ...], shp2: tuple[int, ...]):
    for a_dim, b_dim in zip(reversed(shp1), reversed(shp2)):
        if a_dim != b_dim and a_dim != 1 and b_dim != 1:
            return False
    return True
