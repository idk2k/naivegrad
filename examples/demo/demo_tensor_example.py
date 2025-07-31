import numpy as np
from naivegrad.core_tn import Tensor

def naivegrad_example() -> None:
    x = Tensor(np.eye(3))
    y = Tensor(np.array([[2.0, 0, -2.0]]))
    z = y.dot(x).sum()
    z.backward()

    print(f"x.grad={x.grad} y.grad={y.grad}")

# PyTorch analogous
def torch_example() -> None:
    import torch

    x = torch.eye(3, requires_grad=True)
    y = torch.tensor([[2.0, 0, -2.0]], requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

# run example
naivegrad_example()
