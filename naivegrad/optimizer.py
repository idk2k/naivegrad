# SGD optimizer
class SGD:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr
    
    def step(self):
        for tensor in self.tensors:
            tensor.data -= self.lr * tensor.grad