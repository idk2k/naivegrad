'''
    Simplest demo static-fire test, without third-party libs required.
'''

from naivegrad.core_sc import Scalar

def static_fire() -> None:
    x: Scalar = Scalar(4.0)
    y: Scalar = Scalar(3.0)
    z: Scalar = x * y
    z.backward()
    assert x.grad == 3.0 and y.grad == 4.0