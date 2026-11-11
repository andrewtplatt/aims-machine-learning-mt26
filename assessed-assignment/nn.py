import abc

import numpy as np
from tensor import Tensor


class Module(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        mask = x > 0
        return x * mask


class ShiftScale(Module):
    def __init__(
        self, lb: float, ub: float, target_lb: float, target_ub: float
    ) -> None:
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.target_lb = target_lb
        self.target_ub = target_ub

    def forward(self, x: Tensor) -> Tensor:
        proportion = (x - self.lb) / (self.ub - self.lb)
        return self.target_lb + proportion * (self.target_ub - self.target_lb)


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        x -= x.max(axis=-1)
        exp_x = x.exp()
        return exp_x / exp_x.sum(axis=-1)


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        self.b: float = np.random.random()
        self.W: Tensor = Tensor(np.random.random(size=(input_size, output_size)))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return self.W @ x + self.b


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module.forward(x)
        return x


if __name__ == "__main__":
    relu = ReLU()
    assert (relu.forward(Tensor([-2, -1, 0, 1, 2])) == Tensor([0, 0, 0, 1, 2])).all()
    nn = Sequential(
        Linear(5, 5),
        ReLU(),
    )
    print(nn.forward(Tensor([-2, -1, 0, 1, 2])))
