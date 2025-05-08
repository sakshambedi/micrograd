# %%
import numpy as np
from numpy.typing import DTypeLike


class Tensor:
    def __init__(
        self,
        data: list | np.ndarray | float,
        dtype: DTypeLike = np.float32,
        device: str | tuple | list = "cpu",
        requires_grad: bool | None = None
    ):
        self.data = np.array(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.device: str | tuple | list = device
        self.grad: Tensor | None = None
        self.requires_grad: bool | None = requires_grad

    def realize(self):
        return self

    def numpy(self):
        return self.data

    @classmethod
    def empty(cls, *shape, **kwargs):
        dtype = kwargs.get("dtype", np.float32)
        device = kwargs.get("device", "cpu")
        requires_grad = kwargs.get("requires_grad", False)
        return cls(
            np.empty(shape, dtype=dtype),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, dtype={self.dtype}, device={self.device})"

    def __add__(self, arr2) -> "Tensor":
        result_requires_grad = self.requires_grad is True
        if isinstance(arr2, Tensor):
            other_arr = arr2.data
            result_requires_grad = result_requires_grad or (arr2.requires_grad is True)
        elif isinstance(arr2, np.ndarray) or isinstance(arr2, list):
            other_arr = np.array(arr2)
        else:
            raise TypeError(
                f"Addition operation between a tensor and {type(arr2)} not supported !"
            )

        output_arr = self.data + other_arr

        return Tensor(
            output_arr,
            dtype=output_arr.dtype,
            device=self.device,
            requires_grad=result_requires_grad,
        )

    def __mul__(self, other) -> "Tensor":
        result_requires_grad = self.requires_grad is True
        if isinstance(other, Tensor):
            other_tensor = other.data
            result_requires_grad = result_requires_grad or (other.requires_grad is True)
        elif isinstance(other, (np.ndarray, list, int, float)):
            other_tensor = np.array(other)
        else:
            raise TypeError(
                f"Multiplication operation between a tensor and {type(other)} not supported !"
            )
        output_arr = self.data * other_tensor

        return Tensor(
            output_arr,
            dtype=output_arr.dtype,
            device=self.device,
            requires_grad=result_requires_grad,
        )

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __sub__(self, arr2) -> "Tensor":
        return self + (arr2 * -1)

    def __truediv__(self, other) -> "Tensor":
        result_requires_grad = self.requires_grad is True
        other_data = None
        if isinstance(other, Tensor):
            other_data = other.data
            result_requires_grad = result_requires_grad or (other.requires_grad is True)
        elif isinstance(other, (np.ndarray, list, int, float)):
            other_data = np.array(other)
            # Scalars/arrays don't have requires_grad
        else:
            raise TypeError(
                f"Division operation between a Tensor and {type(other)} not supported!"
            )

        # Let NumPy handle division by zero for true division (inf/nan results)
        output_arr = self.data / other_data
        return Tensor(
            np.asarray(output_arr),
            dtype=output_arr.dtype,  # Result dtype might change (e.g., int / int -> float)
            device=self.device,
            requires_grad=result_requires_grad,
        )

    def __floordiv__(self, other) -> "Tensor":
        result_requires_grad = self.requires_grad is True
        other_data = None
        if isinstance(other, Tensor):
            other_data = other.data
            result_requires_grad = result_requires_grad or (other.requires_grad is True)
        elif isinstance(other, (np.ndarray, list, int, float)):
            other_data = np.array(other)
        else:
            raise TypeError(
                f"Floor division operation between a Tensor and {type(other)} not supported!"
            )

        if np.isscalar(other_data):
            if other_data == 0:
                raise ZeroDivisionError("Scalar floor division by zero")
        elif np.any(other_data == 0):
            raise ZeroDivisionError(
                "Floor division by zero in divisor tensor/array element(s)"
            )

        output_arr = self.data // other_data
        return Tensor(
            np.asarray(output_arr),
            dtype=output_arr.dtype,
            device=self.device,
            requires_grad=result_requires_grad,
        )

    # This is : other / self
    def __rtruediv__(self, other) -> "Tensor":
        if isinstance(other, (Tensor, np.ndarray, list, int, float)):
            other_tensor = Tensor(other) if not isinstance(other, Tensor) else other
            return other_tensor / self
        else:
            raise TypeError(
                f"Reverse true division not supported for type {type(other)} with Tensor"
            )

    def __rfloordiv__(self, other) -> "Tensor":  # other // self
        if isinstance(other, (Tensor, np.ndarray, list, int, float)):
            other_tensor = Tensor(other) if not isinstance(other, Tensor) else other
            return other_tensor // self
        else:
            raise TypeError(
                f"Reverse floor division not supported for type {type(other)} with Tensor"
            )


# %%
t = Tensor(data=[1.0, 2.0], dtype=np.float32)
t_float = Tensor(data=[1.0, 2.5, 3.0], dtype=np.float32)
t_int = Tensor(data=[1, 2, 3], dtype=np.float32)
t_int2 = Tensor(data=[4, 5, 6], dtype=np.float32)
print(t)
print(t_int2)

# %%
