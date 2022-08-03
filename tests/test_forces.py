import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from system import System
from functionals import IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof


class TestForces(unittest.TestCase):

    def test_forces(self):

        # create system and compute ground state energy
        box_vecs = torch.tensor([[3.54, -0.13, 0.25],
                                 [-0.33, 3.82, 0.24],
                                 [0.55, 0.04, 3.45]], dtype=torch.double)
        shape = System.ecut2shape(1600, box_vecs)
        frac_ion_coords = torch.tensor([[0, 0, 0], [0.35, 0.65, 0.45]], dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', frac_ion_coords]]
        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        system.optimize_density(ntol=1e-8)
        cart_ion_coords = system.cartesian_ionic_coordinates('a')
        autograd_forces = system.forces('eV/a')

        # peturb ion coordinates and measure energy
        E_plus, E_minus = torch.empty((2, 3), dtype=torch.double), torch.empty((2, 3), dtype=torch.double)
        eps = 1e-4
        for ion in range(2):
            for i in range(3):
                perturbation = torch.zeros((2, 3), dtype=torch.double)
                perturbation[ion, i] += eps
                system.place_ions(cart_ion_coords + perturbation, units='a')
                system.optimize_density(ntol=1e-8)
                E_plus[ion, i] = system.energy('eV')
                system.place_ions(cart_ion_coords - perturbation, units='a')
                system.optimize_density(ntol=1e-8)
                E_minus[ion, i] = system.energy('eV')
        finite_diff_forces = - (E_plus - E_minus) / (2 * eps)

        self.assertTrue(np.allclose(autograd_forces, finite_diff_forces, atol=1e-4))
        print('Autograd forces work as expected.')


if __name__ == '__main__':
    unittest.main()
