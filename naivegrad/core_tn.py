
'''
    Tensor Core

    For more autograd mechanics:
    - https://github.com/pytorch/pytorch/blob/main/torch/autograd/function.py
    - https://docs.pytorch.org/docs/stable/notes/autograd.html
    - https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
'''

import numpy as np
from functools import partialmethod

class Ctx:
    def __init__(self, arg, *tns):
        self.arg = arg
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        '''
        for default OPS tensors saved as needed,
        but not for custom
        '''
        self.saved_tensors.extend(tensors)

class Tensor:
    '''Tensor-valued with grad abilitys here'''
    def __init__(self, data) -> None:
        assert isinstance(data, np.ndarray), (
            "[err]: failed to construct Tensor instance"
        )
        self.data = data
        self.grad = None

        # inspired by pytorch
        self._ctx = None
    
    def backward(self, allow_fill=True) -> None:
        if self._ctx is None:
            return
        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert self.grad is not None

        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print("[err]: grad musth match with tensor")
                assert(False)
            t.grad = g
            t.backward(False)

    def __str__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

class Function:
    def apply(self, arg, *x):
        ctx = Context(arg, self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret
    
# bind it with tensor
def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# ** OPS **

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad.output.copy()
        grad_input[input < 0] = 0
        return grad_input
register('relu', ReLU)

# https://github.com/Emperor-WS/PyEmber/blob/main/ember/autograd/function.py
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# 2D.dot(2D) = 2D @ 2D (matrix mult)
class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight
register('dot', Dot)

# https://numpy.org/doc/stable/reference/generated/numpy.sum.html
class Sum(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        return np.array(input.sum())

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register('sum', Sum)

# as simple as possible to understand
class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        # not dot and not matrix multiplication
        # * - multiplication each term of x with each corresponding term in y
        # @ - matmul
        return x * y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsume(x):
            c = x.max(axis=1)
            return c + np.log( np.exp( x - c.reshape((-1, 1)) ).sum(axis=1) )
        output = input - logsume(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output) * grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)
