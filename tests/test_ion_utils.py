import numpy as np
import torch
import unittest

from professad.system import System
from professad.functionals import IonIon
from professad.ion_utils import ion_interaction_sum


class TestStress(unittest.TestCase):

    def test1_ion_ion_interaction(self):
        # NaCl test (see https://link.springer.com/article/10.1007/s10910-016-0705-9)
        h_max = np.sqrt(4 / 3)
        r_d_hat = 3.0
        Rc = 3.0 * r_d_hat**2 * h_max
        Rd = r_d_hat * h_max

        box_vecs = torch.tensor([[1, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 1]], dtype=torch.float64)

        # compute FCC energy
        coords = torch.zeros((1, 3), dtype=torch.float64)
        charges = torch.ones(coords.shape[0], dtype=torch.float64)
        E_FCC = ion_interaction_sum(box_vecs, coords, charges, Rc, Rd)

        # compute second term
        coords = torch.tensor([[0, 0, 0],
                               [1, 1, 1]], dtype=torch.float64)
        charges = torch.ones(coords.shape[0], dtype=torch.float64)
        E_2 = ion_interaction_sum(box_vecs, coords, charges, Rc, Rd)
        assert np.allclose(4 * E_FCC - E_2, -1.747564594633)

    def test2_ion_ion_stress(self):

        # Use finite difference method to test ion-ion contribution to stress
        shape = (25, 25, 25)
        box_vecs = torch.tensor([[6.5, -0.13, 0.25],
                                 [-0.33, 7.21, 0.24],
                                 [0.55, 0.04, 6.78]], dtype=torch.double)
        frac_ion_coords = torch.tensor([[0, 0, 0], [0.35, 0.65, 0.45]], dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', frac_ion_coords]]
        terms = [IonIon]
        system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')

        volume = system.volume('a3')
        lattice_vectors = system.lattice_vectors('a')
        autograd_stress = system.stress('eV/a3')

        # perturb lattice vectors and measure energy to compute finite difference stress
        E_plus, E_minus = torch.empty((3, 3), dtype=torch.double), torch.empty((3, 3), dtype=torch.double)
        eps = 1e-5
        for i in range(3):
            for j in range(3):
                strain = torch.zeros((3, 3), dtype=torch.double)
                strain[i, j] += 0.5 * eps
                strain[j, i] += 0.5 * eps
                lat_vec_plus = lattice_vectors + (lattice_vectors @ strain)
                system.set_lattice(lat_vec_plus, units='a')
                E_plus[i, j] = system.energy('eV')
                lat_vec_minus = lattice_vectors - (lattice_vectors @ strain)
                system.set_lattice(lat_vec_minus, units='a')
                E_minus[i, j] = system.energy('eV')
        finite_diff_stress = (E_plus - E_minus).numpy() / (2 * eps * volume)
        self.assertTrue(np.allclose(autograd_stress, finite_diff_stress, atol=1e-9))


if __name__ == '__main__':
    unittest.main()
