import math


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # this will be called on the backward pass
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data**other, (self,), "**")

        def _backward():
            # out = self ** other
            # dself = other * self ** other - 1
            # dother = self ** other * log(self)
            self.grad += other * (self.data ** (other - 1)) * out.grad
            ## TODO AP: check other.grad += self.data ** other.data * math.log(self.data)

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # out = e ** self
            # dself = e ** self
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        if self.data >= 0:
            return self

        out = Value(0, (self,))

        def _backward():
            pass

        out._backward = _backward
        return out

    def sigmoid(self):
        return ((-self).exp() + 1) ** -1

    def cos(self):
        out = Value(math.cos(self.data), (self,), "cos")

        def _backward():
            self.grad -= math.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def sin(self):
        return (self - math.pi / 2).cos()

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v: Value):
            # this needs to fill topo with the nodes in a topological ordering of the graph
            # so we can visit them and call their _backward() in the correct order

            # non-recursive implementation to avoid... issues.
            stack = [v]
            while len(stack) != 0:
                node = stack[-1]
                if len(node._prev) == 0 or all(x in visited for x in node._prev):
                    topo.append(node)
                    stack.pop()
                for child in node._prev:
                    if child not in visited:
                        stack.append(child)
                visited.add(node)


        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * (other**-1)

    def __rtruediv__(self, other):  # other / self
        return (self**-1) * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"