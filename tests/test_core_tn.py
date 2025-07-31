# pytorch required
# pylint fix required

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
    out = x.dot(W).relu()
    out = out.logsoftmax()
    out = out.mul(m).add(m).sum()
    out.backward()
    #print(f"outx.data={out.data} >>> x.grad={x.grad} >>> W.grad={W.grad};")
    return out.data, x.grad, W.grad

def test_pytorch_core() -> None:
    x = torch.tensor(x_0, requires_grad=True)
    W = torch.tensor(W_0, requires_grad=True)
    m = torch.tensor(m_0)
    out = x.matmul(W).relu()
    out = torch.nn.functional.log_softmax(out, dim=1)
    out = out.mul(m).add(m).sum()
    out.backward()
    return out.detach(), x.grad, W.grad

# perform test of core and comparison to pytorch
for x, y in zip(test_tensor_core(), test_pytorch_core()):
    print(x, y)
    np.testing.assert_allclose(x, y, atol=1e-5)
