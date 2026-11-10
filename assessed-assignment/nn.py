import abc

import numpy as np


class Module(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x > 0
        return x * mask


class Tanh(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


class ShiftScale(Module):
    def __init__(
        self, lb: float, ub: float, target_lb: float, target_ub: float
    ) -> None:
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.target_lb = target_lb
        self.target_ub = target_ub

    def forward(self, x: np.ndarray) -> np.ndarray:
        proportion = (x - self.lb) / (self.ub - self.lb)
        return self.target_lb + proportion * (self.target_ub - self.target_lb)


class Softmax(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x -= np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        self.b: float = np.random.random()
        self.W: np.ndarray = np.random.random(size=(input_size, output_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x)
        return x


if __name__ == "__main__":
    relu = ReLU()
    assert (
        relu.forward(np.array([-2, -1, 0, 1, 2])) == np.array([0, 0, 0, 1, 2])
    ).all()
    nn = Sequential(
        Linear(5, 5),
        ReLU(),
    )
    print(nn.forward(np.array([-2, -1, 0, 1, 2])))
