'''
    Tensor Core

    For more autograd mechanics:
    - https://github.com/pytorch/pytorch/blob/main/torch/autograd/function.py
    - https://docs.pytorch.org/docs/stable/notes/autograd.html
    - https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
'''

import numpy as np
from functools import partialmethod

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
    
    # allow_fill only for root Tensor, next i use t.backward(False)
    def backward(self, allow_fill=True) -> None:
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            assert self.data.size == 1
            # hard-grad to 1 (useful for root of computation graph)
            self.grad = np.ones_like(self.data)
        assert self.grad is not None

        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        # for each parent a
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print(f"[err]: grad must match with tensor in _ctx={self._ctx}, g.shape={g.shape} t.data.shape={t.data.shape}")
                assert(False)
            t.grad = g
            # recursive for each parent
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1 / self.data.size]))
        return self.sum().mul(div)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

# class instance of Function stores implicit Context of OP
class Function:
    # implicit context inside Function
    def __init__(self, *tensors):
        self.parents = tensors # input tensors, including on which applied
        self.saved_tensors = [] # for backward pass

    def save_for_backward(self, *tns):
        '''
        for default OPS tensors saved as needed,
        but not for custom
        '''
        self.saved_tensors.extend(tns)

    def apply(self, arg, *x):
        if type(arg) == Tensor:
            op = self
            x = [arg] + list(x)
        else:
            op = arg
            x = [self] + list(x)
        ctx = op(*x)
        ret = Tensor(op.forward(ctx, *[t.data for t in x]))
        ret._ctx = ctx
        return ret
    
# bind apply method of some fxn function to Tensor core
def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# ** OPS **

# https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register("relu", ReLU)

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
register("dot", Dot)

# https://numpy.org/doc/stable/reference/generated/numpy.sum.html
class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register("sum", Sum)

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
register("mul", Mul)

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y
    
    @staticmethod
    def backward(ctx, grad_output):
        # actually its grad_output * 1
        return grad_output, grad_output
register("add", Add)

# https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
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
register("logsoftmax", LogSoftmax)

# https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        cout, cin, H, W = w.shape
        ret = np.zeros((x.shape[0], cout, x.shape[2] - (H - 1), x.shape[3] - (W - 1)), dtype=w.dtype)
        for j in range(H):
            for i in range(W):
                tw = w[:, :, j, i]
                for Y in range(ret.shape[2]):
                    for X in range(ret.shape[3]):
                        ret[:, :, Y, X] += x[:, :, Y + j, X + i].dot(tw.T)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        cout, cin, H, W = w.shape
        for j in range(H):
            for i in range(W):
                tw = w[:, :, j, i]
                for Y in range(grad_output.shape[2]):
                    for X in range(grad_output.shape[3]):
                        gg = grad_output[:, :, Y, X]
                        tx = x[:, :, Y + j, X + i]
                        dx[:, :, Y + j, X + i] += gg.dot(tw)
                        dw[:, :, j, i] += gg.T.dot(tx)
        return dx, dw
        
register('conv2d', Conv2D)