import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from system import System
from functionals import IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof
from ion_utils import cardinal_b_spline_values, exponential_spline_b, structure_factor_spline, structure_factor
from scipy.signal import bspline


class TestPME(unittest.TestCase):

    def test1_cardinal_b_spline_values(self):
        m = 11                 # grid points in [0,1)
        for n in range(2, 31):  # spline order
            p = n - 1            # spline max degree
            spl = torch.zeros((m * n,), dtype=torch.double)

            i_over_m = torch.arange(m, dtype=torch.double) / m
            array = cardinal_b_spline_values(i_over_m, n)
            for i in range(m):
                for j in range(n):
                    spl[i + j * m] = array[j, i]
            x = np.linspace(0, n, m * n, endpoint=False)
            self.assertTrue(np.allclose(spl, bspline(x - (p + 1) / 2, p)))

    def test2_exponential_spline_b(self):
        order = 20
        m = 3  # accuracy degrades for m>3
        N = 9
        x = np.linspace(0, 8, 20, endpoint=False)
        f = np.exp(1j * 2 * np.pi * m / N * x)
        s = np.zeros(x.size, dtype=complex)
        for i in range(x.size):
            for k in range(-50, 50):
                if x[i] - k <= 0 or x[i] - k >= order:
                    continue
                M = cardinal_b_spline_values(torch.as_tensor([x[i] - k - np.floor(x[i] - k)]), order)
                s[i] += M[int(np.floor(x[i] - k))] * np.exp(1j * 2 * np.pi * m / N * k)
        s *= exponential_spline_b(torch.as_tensor([m]), N, order).numpy()
        self.assertTrue(np.allclose(f, s))

    def test3_structure_factors(self):
        shape = (35, 36, 37)
        box_vecs = torch.tensor([[4.9, 0.1, 0.2],
                                 [-0.2, 5.0, 0.3],
                                 [0.3, -0.1, 5.1]], dtype=torch.double)
        cart_ion_coords = torch.tensor([[0, 0, 0],
                                        [2, 0.1, 0.2],
                                        [0.3, 1, 2]], dtype=torch.double)

        str_fac = structure_factor(box_vecs, shape, cart_ion_coords)
        str_fac_spline = structure_factor_spline(box_vecs, shape, cart_ion_coords, 20)

        t = 10
        self.assertTrue(
              np.allclose(str_fac[:t, :t, :t].resolve_conj(), str_fac_spline[:t, :t, :t].resolve_conj())
              * np.allclose(str_fac[:t, -t:, :t].resolve_conj(), str_fac_spline[:t, -t:, :t].resolve_conj())
              * np.allclose(str_fac[-t:, :t, :t].resolve_conj(), str_fac_spline[-t:, :t, :t].resolve_conj())
              * np.allclose(str_fac[-t:, -t:, :t].resolve_conj(), str_fac_spline[-t:, -t:, :t].resolve_conj()))

    def test4_pme_den_force_stress(self):
        # this test checks that the PME structure factors agree with the naive structure factors to
        # the extent that they result in the same optimized densities, energies, auto-differentiated
        # forces and stresses.
        shape = (25, 25, 25)
        box_len = 6.96
        box_vecs = box_len * torch.eye(3, dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', box_len * torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]).double()]]
        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        # naive structure factor
        system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')
        system.optimize_density()
        E1 = system.energy('eV'); den1 = system.density(); force1 = system.forces(); stress1 = system.stress()
        # PME structure factor
        system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional', pme_order=20)
        system.optimize_density()
        E2 = system.energy('eV'); den2 = system.density(); force2 = system.forces(); stress2 = system.stress()

        self.assertTrue(np.allclose(E1, E2))
        self.assertTrue(np.allclose(den1, den2))
        self.assertTrue(np.allclose(force1, force2))
        self.assertTrue(np.allclose(stress1, stress2))
        print('Particle-mesh Ewald scheme works as expected.')


if __name__ == '__main__':
    unittest.main()
