#!/usr/bin/env python

import numpy as np
from naivegrad.core_tn import Tensor
from tqdm import trange

# load dataset for static version (TO REMOVE)
# def fetch_ds(url):
#     import gzip, requests
#     # [TODO]: check if folder exist, add proper error handling
#     with open(url, "rb") as src:
#         dat = src.read()
#     return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

# # Yann LeCunn datasets links now not works good
# # yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# X_train = fetch_ds("train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
# Y_train = fetch_ds("train-labels-idx1-ubyte.gz")[8:]
# X_test = fetch_ds("t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
# Y_test = fetch_ds("t10k-labels-idx1-ubyte.gz")[8:]

# load dataset
def fetch_ds(url):
    import gzip
    # [TODO]: check if folder exist, add proper error handling
    with open(test.dat, "wb") as f:
        raw_dataset = requests.get(url).content
        f.write(raw_dataset)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

# Yann LeCunn datasets links now not works good
# yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
X_train = fetch_ds("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
Y_train = fetch_ds("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch_ds("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
Y_test = fetch_ds("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# training the model

def initialize_layer(m, h):
    ret = np.random.uniform(-1., 1., size=(m, h))/np.sqrt(m*h)
    return ret.astype(np.float32)

# big weight Tensors
# why hidden layer is 128 ? : http://vbystricky.ru/2017/10/mnist_cnn.htmls
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

    x = x.dot(l1)
    x = x.relu()
    x = x_l2 = x.dot(l2)
    x = x.logsoftmax()
    x = x.mul(y)
    x = x.mean()
    x.backward()

    loss = x.data
    cat = np.argmax(x_l2.data, axis=1)
    accuracy = (cat == Y).mean()

    # sgd
    l1.data = l1.data - lr * l1.grad
    l2.data = l2.data - lr * l2.grad

    losses.append(loss)
    accuracies.append(accuracy)
    #t.set_description(f"loss {loss:.2f} accuracy {accuracy:.2f}")
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

# forward pass
def forward(x):
    x = x.dot(l1.data)
    x = np.maximum(x, 0)
    x = x.dot(l2.data)
    return x

def predict():
    Y_test_predict_out = forward(X_test.reshape((-1, 28*28)))
    Y_test_predict = np.argmax(Y_test_predict_out, axis=1)
    return (Y_test == Y_test_predict).mean()

# Prediction:  ~0.9585
print("Prediction: ", predict())