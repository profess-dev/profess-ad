import numpy as np
import torch
import unittest

from professad.functionals import G_inv_lindhard, WangTeter, G_inv_gap, KGAP, YukawaGGA
from professad.functional_tools import wavevectors, get_inv_G


class TestLinearResponse(unittest.TestCase):

    def test_linear_response(self):
        shape = (61, 61, 61)
        box_vecs = 8 * torch.eye(3, dtype=torch.double)
        kxyz = wavevectors(box_vecs, shape)
        den = torch.ones(shape, dtype=torch.double)

        # Wang-Teter
        eta, G_inv_lind = G_inv_lindhard(den, kxyz)
        eta, G_inv_WT = get_inv_G(box_vecs, den, WangTeter)
        self.assertTrue(np.allclose(G_inv_WT, G_inv_lind, atol=1e-10))

        # KGAP
        E_gap = 1.17
        eta, G_inv_KGAP = get_inv_G(box_vecs, den, lambda bv, n: KGAP(bv, n, E_gap))
        eta, G_inv_jgap = G_inv_gap(den, kxyz, E_gap)
        self.assertTrue(np.allclose(G_inv_KGAP[eta != 0], G_inv_jgap[eta != 0], atol=1e-10))

        # Yukawa GGA
        yGGA = YukawaGGA()
        yGGA.mode = 'arithmetic'
        yGGA.kappa = 0.001
        yGGA.set_yuk1()
        eta, G_inv_yuk1 = get_inv_G(box_vecs, den, yGGA.forward)
        G_inv_yuk1_th = 1 / (3 * eta.pow(2) + (-16 * eta.pow(4) + 40 * eta.pow(2) + 5)
                             / (80 * eta.pow(4) + 40 * eta.pow(2) + 5))

        # lower acceptance threshold because spline method's accuracy decreases for higher derivatives
        self.assertTrue(np.allclose(G_inv_yuk1, G_inv_yuk1_th, atol=1e-3))

        yGGA.set_yuk2()
        eta, G_inv_yuk2 = get_inv_G(box_vecs, den, yGGA.forward)

        yGGA.set_yuk3()
        eta, G_inv_yuk3 = get_inv_G(box_vecs, den, yGGA.forward)

        yGGA.set_yuk4()
        eta, G_inv_yuk4 = get_inv_G(box_vecs, den, yGGA.forward)

        G_inv_yuk2_th = 1 / (3 * eta.pow(2) + ((-160 / 3 * yGGA.alpha.pow(2) - 16) * eta.pow(4)
                                               + (- 40 / 3 * yGGA.alpha.pow(4) + 40 * yGGA.alpha.pow(2)) * eta.pow(2)
                                               + 5 * yGGA.alpha.pow(4))
                             / (80 * eta.pow(4) + 40 * eta.pow(2) * yGGA.alpha.pow(2) + 5 * yGGA.alpha.pow(4)))

        # lower acceptance threshold because spline method's accuracy decreases for higher derivatives
        self.assertTrue(np.allclose(G_inv_yuk2, G_inv_yuk2_th, atol=1e-3))
        self.assertTrue(np.allclose(G_inv_yuk3, G_inv_yuk2_th, atol=1e-3))
        self.assertTrue(np.allclose(G_inv_yuk4, G_inv_yuk2_th, atol=1e-3))

        print('Autograd linear response works as expected.')


if __name__ == '__main__':
    unittest.main()
