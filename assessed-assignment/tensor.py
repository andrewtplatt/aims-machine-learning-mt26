from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Union

import numpy as np

Scalar = int | float
TensorInput = Union["Tensor", np.ndarray | Scalar]


class Tensor:
    ###
    # differential programming class based on numpy ndarrays

    def __init__(
        self, data: Sequence[Scalar] | np.ndarray, _children: tuple[Tensor, ...] = ()
    ):
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.bool_):
            data = data.astype(float)

        self._data = data
        self.grad: np.ndarray = np.zeros_like(data, dtype=float)

        # internal variables used for autograd graph construction
        self._backward = lambda: None  # this will be called on the backward pass
        self._prev = set(_children)

    def as_numpy(self) -> np.ndarray:
        """
        Return a view over the raw data as a numpy ndarray.
        """
        return self._data

    @property
    def shape(self) -> tuple:
        return self._data.shape

    def all(self) -> bool:
        return self._data.all()  # pyright: ignore[reportReturnType]

    def max(self, axis=None) -> Tensor:
        out = Tensor(np.max(self._data, axis=axis, keepdims=True), (self,))

        def _backward():
            mask = self._data == out._data
            count = mask.sum(axis=axis, keepdims=True)

            self.grad += out.grad * mask / count

        out._backward = _backward
        return out

    def argmax(self, axis=None) -> Tensor:
        out = Tensor(np.argmax(self._data, axis=axis, keepdims=False))  # pyright: ignore[reportArgumentType]
        return out

    def sum(self, axis=None, keepdims=False) -> Tensor:
        out = Tensor(np.sum(self._data, axis=axis, keepdims=keepdims), (self,))

        def _backward():
            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)

            self.grad += np.broadcast_to(grad, self._data.shape)

        out._backward = _backward
        return out

    def copy(self) -> Tensor:
        return Tensor(self._data.copy())

    def item(self) -> float | int | bool:
        return self._data.item()

    def backward(self):
        topo = []
        visited: set[Tensor] = {self}
        stack: list[tuple[Tensor, Iterator[Tensor]]] = [(self, iter(self._prev))]

        while stack:
            node, children = stack[-1]

            try:
                child = next(children)

                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child._prev)))

            except StopIteration:
                topo.append(node)
                stack.pop()

        if self._data.size != 1:
            raise ValueError("backward() requires a scalar loss")

        self.grad = np.ones_like(self._data, dtype=float)

        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return repr(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __add__(self, other: TensorInput) -> Tensor:
        """
        Element-wise addition of two tensors
        """
        other = _op_convert(other)
        out = Tensor(self._data + other._data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self._data.shape)
            other.grad += _unbroadcast(out.grad, other._data.shape)

        out._backward = _backward

        return out

    def __sub__(self, other: TensorInput) -> Tensor:
        """
        Element-wise subtraction of two tensors
        """
        return self + (-other)

    def __mul__(self, other: TensorInput) -> Tensor:
        """
        Element-wise multiplication of two tensors
        """
        other = _op_convert(other)
        out = Tensor(self._data * other._data, (self, other))

        def _backward():
            self.grad += _unbroadcast(other._data * out.grad, self._data.shape)
            other.grad += _unbroadcast(self._data * out.grad, other._data.shape)

        out._backward = _backward

        return out

    def __truediv__(self, other: TensorInput) -> Tensor:
        """
        Element-wise division of two tensors
        """
        other = _op_convert(other)
        return self * (other**-1)

    def __pow__(self, other: Scalar) -> Tensor:
        """
        Element-wise power of a tensor
        """
        out = Tensor(self._data**other, (self,))

        def _backward():
            self.grad += other * (self._data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> Tensor:
        """
        Element-wise exponential of a tensor
        """
        out = Tensor(np.exp(self._data), (self,))

        def _backward():
            self.grad += out._data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        """
        Element-wise natural logarithm of a tensor
        """
        out = Tensor(np.log(self._data), (self,))

        def _backward():
            self.grad += out.grad / self._data

        out._backward = _backward
        return out

    def gradient_descend(self, lr: float):
        """
        Perform a gradient descent step with the given learning rate.
        Assumes that the gradient has already been backpropagated.
        """
        self._data -= lr * self.grad

    def __gt__(self, other: TensorInput) -> Tensor:
        other = _op_convert(other)
        return Tensor(self._data > other._data, (self, other))

    def __ge__(self, other: TensorInput) -> Tensor:
        other = _op_convert(other)
        return Tensor(self._data >= other._data)

    def __lt__(self, other: TensorInput) -> Tensor:
        other = _op_convert(other)
        return ~(other >= self)

    def __le__(self, other: TensorInput) -> Tensor:
        other = _op_convert(other)
        return ~(other > self)

    def eq(self, other: TensorInput) -> Tensor:
        """
        Element-wise equality of a tensor
        """
        other = _op_convert(other)
        return Tensor(self._data == other._data)

    def neq(self, other: TensorInput) -> Tensor:
        """
        Element-wise inequality of a tensor
        """
        return ~self.eq(other)

    def __invert__(self) -> Tensor:
        """
        Element-wise inversion of a tensor (usually for booleans...)
        """
        out = Tensor(~self._data, (self,))

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __matmul__(self, other: TensorInput) -> Tensor:
        """
        Matrix multiplication of two tensors
        """
        other = _op_convert(other)
        out = Tensor(self._data @ other._data, (self, other))

        def _backward():
            self.grad += out.grad @ other._data.T
            other.grad += self._data.T @ out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: TensorInput) -> Tensor:
        return self + other

    def __rmul__(self, other: TensorInput) -> Tensor:
        return self * other

    def __rsub__(self, other: TensorInput) -> Tensor:
        return _op_convert(other) - self

    def __rtruediv__(self, other: TensorInput) -> Tensor:
        return _op_convert(other) / self


def _op_convert(x: TensorInput) -> Tensor:
    if isinstance(x, np.ndarray):
        x = Tensor(x)
    elif isinstance(x, Scalar):
        x = Tensor(np.array(x))
    elif not isinstance(x, Tensor):
        return NotImplemented

    return x


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)
