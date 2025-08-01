# pytorch required
# pylint fix required

import numpy as np
import torch
import unittest
from naivegrad.core_tn import Tensor, Conv2D

# initialization
x_0 = np.random.randn(1, 3).astype(np.float32)
W_0 = np.random.randn(3, 3).astype(np.float32)
m_0 = np.random.randn(1, 3).astype(np.float32)

class TestNaivegrad(unittest.TestCase):
    def test_backward_pass(self):
        def test_naivegrad():
            x = Tensor(x_0)
            W = Tensor(W_0)
            m = Tensor(m_0)
            
            out = x.dot(W).relu()
            out = out.logsoftmax()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.data, x.grad, W.grad
        
        def test_pytorch():
            x = torch.tensor(x_0, requires_grad=True)
            W = torch.tensor(W_0, requires_grad=True)
            m = torch.tensor(m_0)
            
            out = x.matmul(W).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad
        
        for x, y in zip(test_naivegrad(), test_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
        
    def test_conv2d(self):
        x = torch.randn((5, 2, 10, 7))
        w = torch.randn((4, 2, 3, 3))

        out = torch.nn.functional.conv2d(x, w)
        #ret = Conv2D.apply(Conv2D, Tensor(x.numpy()), Tensor(w.numpy()))
        ret = Tensor(x.numpy()).conv2d(Tensor(w.numpy()))
        np.testing.assert_allclose(ret.data, out.numpy(), atol=1e-5)

if __name__ == '__main__':
    unittest.main()