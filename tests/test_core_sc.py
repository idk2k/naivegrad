# pytorch required
# pylint fix required (pylint: disable=missing-docstring)

import torch
from naivegrad.core_sc import Scalar

def test_sanity_check() -> None:
    x: Scalar = Scalar(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xng, yng = x, y
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass check
    assert yng.data == ypt.data.item()
    # backward pass check
    assert xng.grad == ypt.grad.item()

def test_more_ops() -> None:
    a: Scalar = Scalar(-4.0)
    b: Scalar = Scalar(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    ang, bng, gng = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass check
    assert abs(gng.data - gpt.data.item()) < tol
    # backward pass check
    assert abs(ang.grad - apt.grad.item()) < tol
    assert abs(bng.grad - bpt.grad.item()) < tol
