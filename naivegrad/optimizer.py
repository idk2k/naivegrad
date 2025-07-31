# SGD optimizer
class SGD:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr
    
    def step(self):
        for i in self.tensors:
            t.data -= self.lr * t.grad