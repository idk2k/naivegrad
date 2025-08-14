#!/usr/bin/env python

# one variant: is use line_profiler
try:
    import line_profiler
    prof = line_profiler.LineProfiler()
    import builtins
    builtins.__dict__['profile'] = prof
except ImportError:
    prof = None

import time
import cProfile
import pstats
import unittest
from naivegrad.core_tn import Tensor

def profile_conv(bs, chans, conv, cnt=10):
    img = Tensor.zeroes(bs, 1, 28, 28)
    conv = Tensor.randn(chans, 1, conv, conv)
    fpt, bpt = 0.0, 0.0
    for i in range(cnt):
        et0 = time.time()
        out = img.conv2d(conv)
        et1 = time.time()
        g = out.mean().backward()
        et2 = time.time()
        fpt += (et1 - et0)
        bpt += (et2 - et1)
    return fpt / cnt, bpt / cnt

class TestConvSpeed(unittest.TestCase):
    def test_forward_backward_3x3(self):
        pr = cProfile.Profile(timer=lambda: int(time.time() * 1e9), timeunit=1e-6)
        pr.enable()
        fpt, bpt = profile_conv(128, 16, 3)
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats('cumtime')
        ps.print_stats(0.3)

        if prof is not None:
            prof.print_stats()

        print(f"Forward pass: {fpt * 1000:.3f}")
        print(f"Backward pass: {bpt * 1000:.3f}")

if __name__ == '__main__':
    unittest.main()