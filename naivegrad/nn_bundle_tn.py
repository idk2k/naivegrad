import numpy as np

def initialize_layer(m, h):
    ret = np.random.uniform(-1., 1., size=(m, h)) / np.sqrt(m * h)
    return ret.astype(np.float32)

# SGD optimizer
class SGD:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr
    
    def step(self):
        for i in self.tensors:
            t.data -= self.lr * t.grad