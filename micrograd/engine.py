from __future__ import annotations

class Value:
    def __init__(self, data, _children: tuple = (), _op: str = None):
        self.data = data
        # reference to the children Value that construct this Value.
        self._children = set(_children)
        self._op = _op
        # the lambda will be called on backward pass
        self._backward = lambda: None
        # the cummulative gradient traced from the final output where "backward" is called.
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")
        def _backward():
            # increase the gradient for multi-variable case, ie:
            # c = a + b
            # d = a * 2
            # out = c + d
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data
        output._backward = _backward

    def __neg__(self): # -self
        return self * -1.0

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return -self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return other * other**-1

    def __rtruediff__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self):
        # first, we topological sort all children in the graph
        # then we call _backward in that order

        # store the values in order
        topo = []
        # track the Value that has been visited
        visited = set()

        def visit(v: Value):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    visit(c)
                # after visit all children, append the node
                topo.append(v)

        # gradient to self is always 1
        self.grad = 1.0
        # self is the last element in topo, hence reverse
        for v in reversed(topo):
            v._backward()
