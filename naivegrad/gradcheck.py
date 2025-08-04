import numpy as np

from naivegrad.utils import mask_like
from naivegrad.core_tn import Tensor

def jacobian(func, input):
    output = func(input)

    ji = input.data.reshape(-1).shape[-1]
    jo = output.data.reshape(-1).shape[-1]
    J = np.zeros((jo, ji))

    for o in range(jo):
        o_scalar = Tensor(mask_like(output.data, o, 1.)).mul(output).sum()
        o_scalar.backward()
    
        for i, grad in enumerate(input.grad.reshape(-1)):
            J[o, i] = grad
    return J

def numerical_jacobian(func, input, eps=1e-6):
    output = func(input)

    ji = input.data.reshape(-1).shape[-1]
    jo = output.data.reshape(-1).shape[-1]
    NJ = np.zeros((jo, ji))

    for o in range(jo):
        for i in range(ji):
            eps_perturb = mask_like(input.data, i, mask_value = eps)
            output_perturb_add = func(Tensor(input.data + eps_perturb)).data.reshape(-1)[o]
            output_perturb_sub = func(Tensor(input.data - eps_perturb)).data.reshape(-1)[o]

            grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2 * eps)

            NJ[o, i] = grad_approx

    return NJ

def gradcheck(func, input, eps=1e-6, atol=1e-5, rtol=0.001):
    NJ = numerical_jacobian(func, input, eps)
    J = jacobian(func, input)
    return np.allclose(J, NJ, atol=atol, rtol=rtol)
