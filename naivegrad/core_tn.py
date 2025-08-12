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
        if data.dtype == np.float64:
            #print("[WARNING ngrad]: float64 tensor constructed - some bugs may appear. Be careful")
            pass
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
            if g is None:
                continue
            if g.shape != t.data.shape:
                print(f"[err]: grad must match with tensor in _ctx={self._ctx}, g.shape={g.shape} t.data.shape={t.data.shape}")
                assert(False)
            t.grad = g
            # recursive for each parent
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1 / self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def zeroes(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    def __repr__(self) -> str:
        return f"(data={self.data}, grad={self.grad})"

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
import naivegrad.ops