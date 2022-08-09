import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../professad'))
from functional_tools import field_dependent_convolution, wavevecs


class TestSpline(unittest.TestCase):

    def test_spline(self):
        shape = (20, 20, 20)
        box_vecs = 2 * torch.eye(3, dtype=torch.double)

        xf, yf, zf = np.meshgrid(np.arange(shape[0]) / shape[0], np.arange(shape[1]) / shape[1],
                                 np.arange(shape[2]) / shape[2], indexing='ij')
        x = box_vecs[0, 0] * xf + box_vecs[1, 0] * yf + box_vecs[2, 0] * zf
        y = box_vecs[0, 1] * xf + box_vecs[1, 1] * yf + box_vecs[2, 1] * zf
        z = box_vecs[0, 2] * xf + box_vecs[1, 2] * yf + box_vecs[2, 2] * zf
        r = np.sqrt(x * x + y * y + z * z)

        kx, ky, kz, k2 = wavevecs(box_vecs, shape)

        # Yukawa potential in Fourier space, K(q,ξ) = 4π/(q²+ξ²)
        def K_tilde(k2, xi_sparse):
            return 4 * np.pi / (k2.unsqueeze(3).expand((-1, -1, -1, len(xi_sparse))) + xi_sparse.pow(2))

        # ξ(r) = cos²(r) + 1
        xis = torch.cos(r).pow(2) + 1
        # g(r') = [ξ(r')]^(1/3)
        g = xis.pow(1 / 3)
        # Computes u(r) = ∫d³r' K(|r-r'|,ξ(r)) g(r')
        u_spline = field_dependent_convolution(k2, K_tilde, g, xis, kappa=0.01)

        u_naive = torch.empty(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    K = 4 * np.pi / (k2 + xis[i, j, k].pow(2))
                    u_naive[i, j, k] = torch.fft.irfftn(torch.fft.rfftn(g) * K, xis.shape)[i, j, k]

        self.assertTrue(np.allclose(u_spline, u_naive, atol=1e-10))
        print('Spline based field-dependent convolution works as expected.')


if __name__ == '__main__':
    unittest.main()
