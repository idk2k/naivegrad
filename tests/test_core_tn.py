import numpy as np
import torch
from naivegrad.core_tn import Tensor

# initialization
x_0 = np.random.randn(1, 3).astype(np.float32)
W_0 = np.random.randn(3, 3).astype(np.float32)
m_0 = np.random.randn(1, 3).astype(np.float32)

def test_tensor_core() -> None:
    x = Tensor(x_0)
    W = Tensor(W_0)
    m = Tensor(m_0)
    # matrix multiplication or dot
    out = x.dot(W)
    outr = out.relu()
    outl = outr.logsoftmax()
    # element-wise multiplication
    outm = outl.mul(m)
    outx = outm.sum()
    outx.backward()
    print(f"{outx.data} {x.grad} {W.grad}")
    