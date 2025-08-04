'''
    Scalar Core
'''

from typing import Union, Tuple, List, Set, Callable

class Scalar:
    """Scalar value with grad here"""

    def __init__(
        self, data: float, _children: Tuple["Scalar", ...] = (), _op: str = ""
    ) -> None:
        self.data: float = data
        self.grad: float = 0
        # for graph-based operations
        self._prev = set(_children)
        # for debugging purpose (or graphviz)
        self._op = _op
        # only for root, now
        self._backward: Callable = lambda: None

    def __add__(self, other: Union["Scalar", float]) -> "Scalar":
        other: Scalar = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), "+")
        def _backward() -> None:
            self.grad += out.grad # out.grad * d(out)/d(self) = out.grad * 1
            other.grad += out .grad # same here
        out._backward = _backward
        return out

    def __radd__(self, rother: Union["Scalar", float]) -> "Scalar":
        return self + rother

    def __sub__(self, other: Union["Scalar", float]) -> "Scalar":
        return self + (-other)

    def __rsub__(self, rother: Union["Scalar", float]) -> "Scalar":
        return rother + (-self)

    def __mul__(self, other: Union["Scalar", float]) -> "Scalar":
        other: Scalar = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), "*")
        def _backward() -> None:
            self.grad += other.data * out.grad # follow README to understand this part
            other.grad += self.data * out.grad # same
        out._backward = _backward
        return out

    def __rmul__(self, rother: Union["Scalar", float]) -> "Scalar":
        return self * rother

    def __truediv__(self, other: Union["Scalar", float]) -> "Scalar":
        return self * other**-1

    def __rtruediv__(self, rother: Union["Scalar", float]) -> "Scalar":
        return rother * self**-1

    def __pow__(self, other: Union[int, float]) -> "Scalar":
        assert isinstance(other, (int, float)), "[warn]: direct support for int/floats"
        out: Scalar = Scalar(self.data ** other, (self,), f"**{other}")
        def _backward() -> None:
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward
        return out
    
    def __neg__(self) -> "Scalar":
        return self * -1

    def relu(self) -> "Scalar":
        out: Scalar = Scalar(max(0, self.data), (self,), "ReLU")
        def _backward() -> None:
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out
        
    def backward(self) -> None:
        topology: Vector = []
        visited: Set[Scalar] = set()
        def build_topology(node: Scalar) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topology(child)
                topology.append(node)
        build_topology(self)
        # autograd
        self.grad = 1 # df/df = 1
        # reversed, because build_topology is recursive
        for node in reversed(topology):
            node._backward()
  
    def __repr__(self) -> str:
        return f"Scalar(data={self.data}, grad={self.grad})"

Vector = List[Scalar]