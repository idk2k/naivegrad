#!/usr/bin/env python
import numpy as np
import os
from tqdm import trange
import unittest

from naivegrad.core_tn import Tensor
from naivegrad.utils import fetch_mnist, initialize_layer_uniform
import naivegrad.optimizer as optim

# CONFIG VAR
CONV = 1

np.random.seed(1337)

# load ds
X_train, Y_train, X_test, Y_test = fetch_mnist()

class NaiveNet:
    '''
        Naive model for MNIST with 128 hidden layer
    '''
    def __init__(self):
        self.l1 = Tensor(initialize_layer_uniform(784, 128))
        self.l2 = Tensor(initialize_layer_uniform(128, 10))

    def forward(self, x):
        ret = x.dot(self.l1).relu().dot(self.l2).logsoftmax()
        return ret

class NaiveConvolutionNet:
    def __init__(self) -> None:
        conv = 7
        chans = 8
        self.c1 = Tensor(initialize_layer_uniform(chans, 1, conv, conv))
        self.l1 = Tensor(initialize_layer_uniform(((28 - conv + 1)**2) * chans, 128))
        self.l2 = Tensor(initialize_layer_uniform(128, 10))
    
    def forward(self, x):
        x.data = x.data.reshape((-1, 1, 28, 28))
        x = x.conv2d(self.c1).relu()
        x = x.reshape(Tensor(np.array((x.shape[0], -1))))
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class TestMNIST(unittest.TestCase):
    def test_mnist(self):
        if CONV == 1:
            model_instance = NaiveConvolutionNet()
            optimizer_instance = optim.Adam([model_instance.c1, model_instance.l1, model_instance.l2], lr=0.001)
            steps = 400
        else:
            model_instance = NaiveNet()
            optimizer_instance = optim.SGD([model_instance.l1, model_instance.l2], lr=0.001)
            steps = 1000
                
        # Train
        batch_size = 128
        losses, accuracies = [], []
        for i in (t := trange(steps)):
            samp = np.random.randint(0, X_train.shape[0], size=(batch_size))

            x = Tensor(X_train[samp].reshape((-1, 28 * 28)).astype(np.float32))
            Y = Y_train[samp]
            y = np.zeros((len(samp), 10), np.float32)
            # torch nll return one per row
            y[range(y.shape[0]), Y] = -10.0
            y = Tensor(y)

            out = model_instance.forward(x)

            # NLL (Negative Log-Likelihood) loss
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
            loss = out.mul(y).mean()
            loss.backward()

            # SGD step
            #sgd_optimizer.step()
        
            # Adam step: BUT IMPLEMENTATION IS BAD PROBABLY: prediction 0.6
            optimizer_instance.step()

            accuracy = (np.argmax(out.data, axis=1) == Y).mean() 

            loss = loss.data
            losses.append(loss)
            accuracies.append(accuracy)
            t.set_description(f"loss={loss.item():.2f}, accuracy={accuracy.item():.2f}")

        def predict():
            Y_test_predict_out = model_instance.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
            Y_test_predict = np.argmax(Y_test_predict_out.data, axis=1)
            return (Y_test == Y_test_predict).mean()

        # Prediction
        accuracy = predict()
        print("Prediction: ", accuracy)
        assert accuracy > 0.952101

if __name__ == '__main__':
    unittest.main()
