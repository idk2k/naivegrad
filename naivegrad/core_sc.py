
class Scalar:
    """ Scalar value with grad here """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # for graph-based operations
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None # only for root, now

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out .grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "direct support for int/floats"
        out = Scalar(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward

        return out

    def relu(self):
        out = Scalar(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward

        return out
        
    def backward(self):
        topology = []
        visited = set()
        def build_topology(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topology(child)
                topology.append(v)
        build_topology(self)

        # autograd
        self.grad = 1 # df/df = 1
        # reversed, because build_topology is recursive
        for v in reversed(topology):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, rother):
        return self + rother

    def __rmul__(self, rother):
        return self * rother

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, rother):
        return rother + (-self)
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, rother):
        return other * self**-1
    
    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"