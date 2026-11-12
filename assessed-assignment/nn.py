import abc

import numpy as np
from tensor import Tensor


class Module(abc.ABC):
    def __init__(self):
        self._parameters: list[Tensor] = []

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self._parameters:
            p.grad = np.zeros_like(p.grad)

    def gradient_descend(self, lr: float) -> None:
        for p in self._parameters:
            p.gradient_descend(lr)

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        mask = x > 0
        return x * mask


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.b: Tensor = Tensor(np.random.randn(output_size))
        self.W: Tensor = Tensor(
            np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        )
        self.input_size = input_size
        self.output_size = output_size
        self._parameters += [self.W, self.b]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules
        for m in modules:
            self._parameters += m._parameters

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module.forward(x)
        return x


if __name__ == "__main__":
    relu = ReLU()
    assert relu.forward(Tensor([-2, -1, 0, 1, 2])).eq(Tensor([0, 0, 0, 1, 2])).all()
    nn = Sequential(
        Linear(5, 5),
        ReLU(),
    )
    print(nn.forward(Tensor([-2, -1, 0, 1, 2])))
