from naivegrad.core_sc import Scalar
from naivegrad.nn_bundle_sc import Neuron, Layer, MLP
import random

# prepare mlp
model = MLP(2, [16, 16, 1]) # 2-l nn
print(model)
print(f"#parameters: {len(model.parameters())}")

# forming loss function

#def loss(batch_size=None):
    # todo