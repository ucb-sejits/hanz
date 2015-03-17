import unittest
from cstructures.array import Array, transpose
from hanz import compose
import numpy as np

dot = Array.dot


class TestArrayOps(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    @unittest.skip("")
    def test_blas(self):
        A = Array.rand(256, 256).astype(np.float32)
        x = Array.rand(256, 256).astype(np.float32)
        b = Array.rand(256, 256).astype(np.float32)

        def axb(A, x, b):
            return dot(A, x) - b

        composed = compose(axb)
        self._check(composed(A, x, b), axb(A, x, b))

    def test_pl(self):
        A = Array.rand(256, 256).astype(np.float32)
        b = Array.rand(256, 256).astype(np.float32)
        alpha = .1

        def pL(y):
            z = y - alpha*(transpose(A)*(A*y - b))
            # z[z<0] = 0
            return z

        y = Array.rand(256, 256).astype(np.float32)

        composed = compose(pL)
        self._check(composed(y), pL(y))
