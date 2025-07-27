from naivegrad.core_sc import Scalar
from naivegrad.nn_bundle_sc import Neuron, Layer, MLP
import random

#import numpy as np
#import matplotlib.pyplot as plt

# prepare MLP: 2 inputs and 1 output, 16 middle-layers
# and ReLU for all except last
model = MLP(2, [16, 16, 1])
print(f"#parameters: {len(model.parameters())}")

# [Important note about the model (MLP)]:
# Note that model() (i.e after execution) is tuple
# of resulted Scalars object. For each layer item - own Scalar, that
# contains all history back to inputs!
# So we can apply .backward() to any of the item from Layer, including last layer
# (output layer)
# I output Layer contains only 1 item (Scalar), then it contains all history of the NN,
# back to origin inputs!

np.random.seed(1337)
random.seed(1337)

# dataset

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # fit in range [-1, 1]
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

# loss
def loss(batch_size=None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Scalar, xrow)) for xrow in Xb]

    scores = list(map(model, inputs))

    # max-margin loss
    losses = [(1 + -yi*scorei).relu() for scorei, yi  in zip(scores, yb)]
    data_loss = sum(losses) * (1.0 / len(losses))

    # L2
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

total_loss, acc = loss()
print(total_loss, acc)

# optimizer
for k in range(100):
    # forward pass
    total_loss, acc = loss()

    # backward pass
    model.zero_grad()
    total_loss.backward() # updates the grads

    # SGD optimizer
    lr = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p -= lr * p.grad

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")
