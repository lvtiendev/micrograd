from __future__ import annotations

import random
from micrograd.engine import Value

class Module:
    def zero_grad(self):
        for v in self.parameters():
            v.grad = 0.0

    def parameters(self) -> list[Value]:
        return []

class Neuron(Module):
    """
    The input to a Neuron has in_dim dimension.
    The output is a single value.
    """
    def __init__(self, in_dim: int, non_linear=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_dim)]
        self.b = Value(0) # the bias
        self.non_linear = non_linear

    def __call__(self, x) -> Value:
        # dot product between weights and input
        activation = sum(((wi*xi) for wi,xi in zip(self.w, x)), self.b)
        return activation.relu() if self.non_linear else activation

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    Layer is a list of Neuron.
    When applied to a input, each Neuron returns a single Value.
    The final output is the vector of Value.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        self.neurons = [Neuron(in_dim, **kwargs) for _ in range(out_dim)]

    def __call__(self, x) -> list[Value] | Value:
        activations = [n(x) for n in self.neurons]
        # common case for the last layer
        return activations[0] if len(activations) == 0 else activations

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    Multiple Layers
    """

    def __init__(self, in_dim: int, out_dims: list[int]):
        sz = [in_dim] + out_dims
        return [
            # don't ReLU for the last Layer
            Layer(sz[i], sz[i+1], non_linear=i<len(out_dims)-1)
            for i in range(out_dims)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for l in self.layers for p in l.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
