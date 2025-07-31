#!/usr/bin/env python

import numpy as np
from tqdm import trange

from naivegrad.core_tn import Tensor
from naivegrad.utils import fetch_mnist
import naivegrad.optimizer as optim
# load dataset for static version (TO REMOVE)
# def fetch_ds(url):
#     import gzip, requests
#     with open(url, "rb") as src:
#         dat = src.read()
#     return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
# X_train = fetch_ds("train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
# Y_train = fetch_ds("train-labels-idx1-ubyte.gz")[8:]
# X_test = fetch_ds("t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
# Y_test = fetch_ds("t10k-labels-idx1-ubyte.gz")[8:]

def initialize_layer(m, h):
    ret = np.random.uniform(-1., 1., size=(m, h)) / np.sqrt(m * h)
    return ret.astype(np.float32)

# load ds
X_train, Y_train, X_test, Y_test = fetch_mnist()

class NaiveNet:
    '''
        Naive model for MNIST with 128 hidden layer
    '''
    def __init__(self):
        self.l1 = Tensor(initialize_layer(784, 128))
        self.l2 = Tensor(initialize_layer(128, 10))

    def forward(self, x):
        ret = x.dot(self.l1).relu().dot(self.l2).logsoftmax()
        return ret

model_instance = NaiveNet()
sgd_optimizer = optim.SGD([model_instance.l1, model_instance.l2], lr=0.01)

# Weights tensor. Why hidden layer is 128 ?
# http://vbystricky.ru/2017/10/mnist_cnn.htmls
l1 = Tensor(initialize_layer(784, 128))
l2 = Tensor(initialize_layer(128, 10))

lr = 0.01
batch_size = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))

    x = Tensor(X_train[samp].reshape((-1, 28*28)))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    y[range(y.shape[0]), Y] = -1.0
    y = Tensor(y)

    outs = model_instance.forward(x)

    # NLL loss
    loss = outs.mul(y).mean()
    loss.backward()

    # SGD step
    sgd_optimizer.step()

    accuracy = (cat == Y).mean() 

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def predict():
    Y_test_predict_out = model_instance.forward(Tensor(X_test.reshape((-1, 28*28))))
    Y_test_predict = np.argmax(Y_test_predict_out.data, axis=1)
    return (Y_test == Y_test_predict).mean()

# Prediction
accuracy = predict()
print("Prediction: ", accuracy)

assert accuracy > 0.952101