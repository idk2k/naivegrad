#!/usr/bin/env python
import numpy as np
from tqdm import trange

from naivegrad.core_tn import Tensor
from naivegrad.utils import fetch_mnist, initialize_layer_uniform
import naivegrad.optimizer as optim

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

model_instance = NaiveNet()

# Optimizer
#sgd_optimizer = optim.SGD([model_instance.l1, model_instance.l2], lr=0.001)
adam_optimizer = optim.Adam([model_instance.l1, model_instance.l2], lr=0.001)

# Train
lr = 0.01
batch_size = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))

    x = Tensor(X_train[samp].reshape((-1, 28*28)))
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
    adam_optimizer.step()

    accuracy = (np.argmax(out.data, axis=1) == Y).mean() 

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description(f"loss={loss.item():.2f}, accuracy={accuracy.item():.2f}")

def predict():
    Y_test_predict_out = model_instance.forward(Tensor(X_test.reshape((-1, 28*28))))
    Y_test_predict = np.argmax(Y_test_predict_out.data, axis=1)
    return (Y_test == Y_test_predict).mean()

# Prediction
accuracy = predict()
print("Prediction: ", accuracy)

assert accuracy > 0.952101