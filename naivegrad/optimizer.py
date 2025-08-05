import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params
    
class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGD, self).__init__(params)
        self.lr = lr
    
    def step(self):
        for tensor in self.params:
            tensor.data -= self.lr * tensor.grad

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super(Adam, self).__init__(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(tensor.data) for tensor in self.params]
        self.v = [np.zeros_like(tensor.data) for tensor in self.params]

    # https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    def step(self):
        self.t += 1
        for i, t in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)
            mhat = self.m[i] / (1.0 - self.b1**self.t)
            vhat = self.v[i] / (1.0 - self.b2**self.t)
            t.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

# https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
# https://education.yandex.ru/handbook/ml/article/metody-optimizacii-v-deep-learning
class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
        super(RMSProp, self).__init__(params)
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.v = [np.zeros_like(t.data) for t in self.params]

    def step(self):
        for i, t in enumerate(self.params):
            self.v[i] = self.decay * self.v[i] + (1 - self.decay) * np.square(t.grad)
            t.data -= (self.lr / (np.sqrt(self.v[i]) + self.eps)) * t.grad
