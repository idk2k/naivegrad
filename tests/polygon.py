from naivegrad.core_tn import Tensor
import torch
import numpy as np

t = Tensor(np.array([[1,2,3]]))
t.reshape((1, 3))
#print(torch.randn((4, 2, 3, 3)).numpy().shape)