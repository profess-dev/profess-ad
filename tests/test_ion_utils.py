import numpy as np
import torch
import unittest

from professad.system import System
from professad.functionals import IonIon
from professad.ion_utils import ion_interaction_sum


class TestStress(unittest.TestCase):

    def test1_ion_ion_interaction(self):
        # tests based on those of
        # https://github.com/wcwitt/real-space-electrostatic-sum/tree/master
        # where the references energies were obtained with CASTEP

        # 1) Al
        box_vecs = torch.tensor([[5.41141973394663, 0.00000000000000, 0.00000000000000],
                                 [2.70570986697332, 4.68642696013821, 0.00000000000000],
                                 [2.70570986697332, 1.56214232004608, 4.41840571073226]],
                                 dtype=torch.float64)
        coords = torch.zeros((1, 3), dtype=torch.float64)
        charges = 3.0 * torch.ones((coords.shape[0],), dtype=torch.float64)

        h_max = 4.42
        Rc = 12 * h_max
        Rd = 2 * h_max
        E = ion_interaction_sum(box_vecs, coords, charges, Rc, Rd)
        assert np.abs(E.item() - -2.69595457432924945) < 1e-10

        # 2) Si
        box_vecs = torch.tensor([[7.25654832321381, 0.00000000000000, 0.00000000000000],
                                 [3.62827416160690, 6.28435519169252, 0.00000000000000],
                                 [3.62827416160690, 2.09478506389751, 5.92494689524090]],
                                 dtype=torch.float64)
        coords = torch.tensor([[0.00, 0.00, 0.00],
                               [0.25, 0.25, 0.25]],
                               dtype=torch.float64) @ box_vecs
        charges = 4.0 * torch.ones(coords.shape[0], dtype=torch.float64)

        h_max = 5.92
        Rc = 12 * h_max
        Rd = 2 * h_max
        E = ion_interaction_sum(box_vecs, coords, charges, Rc, Rd)
        assert np.abs(E.item() - -8.39857465282205418) / coords.shape[0] < 1e-10

        # 3) SiO2
        box_vecs = torch.tensor([[9.28422445623683, 0.00000000000000, 0.00000000000000],
                                 [-4.64211222811842, 8.04037423353787, 0.00000000000000],
                                 [0.00000000000000, 0.00000000000000, 10.2139697101486]],
                                 dtype=torch.float64)

        coords = torch.tensor([[0.41500, 0.27200, 0.21300],
                               [0.72800, 0.14300, 0.54633],
                               [0.85700, 0.58500, 0.87967],
                               [0.27200, 0.41500, 0.78700],
                               [0.14300, 0.72800, 0.45367],
                               [0.58500, 0.85700, 0.12033],
                               [0.46500, 0.00000, 0.33333],
                               [0.00000, 0.46500, 0.66667],
                               [0.53500, 0.53500, 0.00000]],
                               dtype=torch.float64) @ box_vecs

        charges = 6.0 * torch.ones(coords.shape[0], dtype=torch.float64)  # most are O
        charges[6:] = 4.0  # three are Si

        h_max = 10.21
        Rc = 12 * h_max
        Rd = 2 * h_max
        E = ion_interaction_sum(box_vecs, coords, charges, Rc, Rd)
        assert np.abs(E.item() - -69.48809871723248932) / coords.shape[0] < 1e-10

        # 4) Al2SiO5
        box_vecs = torch.tensor([[14.7289033699982, 0.00000000000000, 0.00000000000000],
                                 [0.00000000000000, 14.9260018049230, 0.00000000000000],
                                 [0.00000000000000, 0.00000000000000, 10.5049875335275]],
                                 dtype=torch.float64)
        coords = torch.tensor([[0.23030, 0.13430, 0.23900],
                               [0.76970, 0.86570, 0.23900],
                               [0.26970, 0.63430, 0.26100],
                               [0.73030, 0.36570, 0.26100],
                               [0.76970, 0.86570, 0.76100],
                               [0.23030, 0.13430, 0.76100],
                               [0.73030, 0.36570, 0.73900],
                               [0.26970, 0.63430, 0.73900],
                               [0.00000, 0.00000, 0.24220],
                               [0.50000, 0.50000, 0.25780],
                               [0.00000, 0.00000, 0.75780],
                               [0.50000, 0.50000, 0.74220],
                               [0.37080, 0.13870, 0.50000],
                               [0.42320, 0.36270, 0.50000],
                               [0.62920, 0.86130, 0.50000],
                               [0.57680, 0.63730, 0.50000],
                               [0.12920, 0.63870, 0.00000],
                               [0.07680, 0.86270, 0.00000],
                               [0.87080, 0.36130, 0.00000],
                               [0.92320, 0.13730, 0.00000],
                               [0.24620, 0.25290, 0.00000],
                               [0.42400, 0.36290, 0.00000],
                               [0.10380, 0.40130, 0.00000],
                               [0.75380, 0.74710, 0.00000],
                               [0.57600, 0.63710, 0.00000],
                               [0.89620, 0.59870, 0.00000],
                               [0.25380, 0.75290, 0.50000],
                               [0.07600, 0.86290, 0.50000],
                               [0.39620, 0.90130, 0.50000],
                               [0.74620, 0.24710, 0.50000],
                               [0.92400, 0.13710, 0.50000],
                               [0.60380, 0.09870, 0.50000]],
                               dtype=torch.float64) @ box_vecs

        charges = 6.0 * torch.ones((coords.shape[0],), dtype=torch.float64)  # most are O
        charges[8:13] = 3.0  # eight are Al
        charges[14] = 3.0
        charges[16] = 3.0
        charges[18] = 3.0
        charges[20] = 4.0    # four are Si
        charges[23] = 4.0
        charges[26] = 4.0
        charges[29] = 4.0

        h_max = 14.93
        Rc = 12 * h_max
        Rd = 2 * h_max
        E = ion_interaction_sum(box_vecs, coords, charges, Rc, Rd)
        assert np.abs(E.item() - -244.05500850908111943) / coords.shape[0] < 1e-10

        # 5) NaCl test (see https://link.springer.com/article/10.1007/s10910-016-0705-9)
        h_max = np.sqrt(4 / 3)
        Rc = 12 * h_max
        Rd = 2 * h_max

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
        assert np.abs((4 * E_FCC - E_2).item() - -1.747564594633) < 1e-10

    def test2_ion_ion_derivatives(self):

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
