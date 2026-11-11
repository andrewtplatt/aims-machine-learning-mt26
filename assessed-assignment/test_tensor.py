import numpy as np
from tensor import Tensor


def assert_grad(tensor, expected):
    np.testing.assert_allclose(tensor.grad, expected, rtol=1e-6, atol=1e-7)


def test_add_forward_and_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0, 6.0]))
    result = a + b
    np.testing.assert_allclose(result.as_numpy(), [5.0, 7.0, 9.0])
    result.sum().backward()
    assert_grad(a, [1.0, 1.0, 1.0])
    assert_grad(b, [1.0, 1.0, 1.0])


def test_multiply_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0, 6.0]))
    (a * b).sum().backward()
    assert_grad(a, [4.0, 5.0, 6.0])
    assert_grad(b, [1.0, 2.0, 3.0])


def test_power_backward():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    (x**3).sum().backward()
    assert_grad(x, [3.0, 12.0, 27.0])


def test_reciprocal_backward():
    x = Tensor(np.array([1.0, 2.0, 4.0]))
    (x**-1).sum().backward()
    assert_grad(x, [-1.0, -0.25, -0.0625])


def test_exp_backward():
    x = Tensor(np.array([0.0, 1.0, 2.0]))
    x.exp().sum().backward()
    assert_grad(x, np.exp([0.0, 1.0, 2.0]))


def test_shared_graph():
    x = Tensor(np.array([2.0, 3.0]))
    y = x * x
    (y + y).sum().backward()
    assert_grad(x, [8.0, 12.0])


def test_broadcasting_backward():
    a = Tensor(np.ones((2, 3)))
    b = Tensor(np.array([1.0, 2.0, 3.0]))
    (a + b).sum().backward()
    assert_grad(a, np.ones((2, 3)))
    assert_grad(b, [2.0, 2.0, 2.0])


def test_sum_axis_backward():
    x = Tensor(np.arange(6.0, dtype=float).reshape(2, 3))
    x.sum(axis=1).sum().backward()
    assert_grad(x, np.ones((2, 3)))


def test_max_backward_unique():
    x = Tensor(np.array([[1.0, 4.0, 2.0], [6.0, 3.0, 5.0]]))
    x.max(axis=1).sum().backward()
    assert_grad(x, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_matmul_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    (a @ b).sum().backward()
    assert_grad(a, [[11.0, 15.0], [11.0, 15.0]])
    assert_grad(b, [[4.0, 4.0], [6.0, 6.0]])


def test_numerical_gradient():
    x0 = np.array([0.5, 1.5, 2.5])
    x = Tensor(x0.copy())
    ((x**2) * x.exp()).sum().backward()
    expected = (2 * x0 + x0**2) * np.exp(x0)
    assert_grad(x, expected)


def test_scalar_division():
    x = Tensor(np.array([2.0, 4.0]))
    (x / 2).sum().backward()
    assert_grad(x, [0.5, 0.5])
