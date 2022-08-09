import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../professad'))
from system import System
from functionals import IonIon, IonElectron, Hartree, Weizsaecker, LuoKarasievTrickey, WangTeter, PerdewBurkeErnzerhof
from crystal_tools import get_cell


class TestDenOpt(unittest.TestCase):

    def test1_exact_cases(self):

        # using a large box to test non-interacting single orbital systems
        L = 20.0
        box_vecs = L * torch.eye(3, dtype=torch.double)
        shape = System.ecut2shape(250, box_vecs)

        # hydrogen atom
        ions = [['H', 'potentials/H.coulomb-kcut-15.recpot', torch.tensor([[0.5, 0.5, 0.5]]).double()]]
        terms = [IonElectron, Weizsaecker]
        system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')
        system.set_electron_number(1)

        system.optimize_density(ntol=1e-4)
        self.assertAlmostEqual(system.energy('Ha'), -0.5, places=2)

        # quantum harmonic oscillator
        k = 10
        xf, yf, zf = np.meshgrid(np.arange(shape[0]) / shape[0], np.arange(shape[1]) / shape[1],
                                 np.arange(shape[2]) / shape[2], indexing='ij')
        x = box_vecs[0, 0] * xf + box_vecs[1, 0] * yf + box_vecs[2, 0] * zf
        y = box_vecs[0, 1] * xf + box_vecs[1, 1] * yf + box_vecs[2, 1] * zf
        z = box_vecs[0, 2] * xf + box_vecs[1, 2] * yf + box_vecs[2, 2] * zf
        qho_pot = 0.5 * k * ((x - L / 2).pow(2) + (y - L / 2).pow(2) + (z - L / 2).pow(2))

        system.set_potential(torch.as_tensor(qho_pot).double())
        system.initialize_density()
        system.optimize_density(ntol=1e-4)
        self.assertAlmostEqual(system.energy('Ha'), 3 / 2 * np.sqrt(k), places=5)

    def test2_compare_optimizers(self):
        terms = [IonIon, IonElectron, Hartree, LuoKarasievTrickey, PerdewBurkeErnzerhof]
        box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.8, coord_type='fractional')
        ions = [['Al', 'potentials/al.gga.recpot', frac_ion_coords]]
        shape = System.ecut2shape(1600, box_vecs)
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        # optimize density with LBFGS
        system.optimize_density(ntol=1e-4, n_method='LBFGS')
        E1 = system.energy('eV')
        system.initialize_density()
        # optimize density with TPGD
        system.optimize_density(ntol=1e-4, n_conv_cond_count=5, n_method='TPGD')
        E2 = system.energy('eV')
        self.assertAlmostEqual(E1, E2, places=3)

    def test3_check_convergence_measures(self):
        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.8, coord_type='fractional')
        ions = [['Al', 'potentials/al.gga.recpot', frac_ion_coords]]
        shape = System.ecut2shape(1600, box_vecs)
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        system.optimize_density(ntol=1e-4)

        dEdchi = system.check_density_convergence()

        dEdn = system.functional_derivative('density')
        chi = torch.sqrt(system.density())
        N_tilde = torch.mean(chi.pow(2)) * system.volume()
        dEdchi_from_dEdn = (system.electron_count() / N_tilde) * 2 * chi * \
                           (dEdn - torch.mean(dEdn * system.density()) * system.volume() / system.electron_count())

        self.assertTrue(np.allclose(dEdchi, torch.max(torch.abs(dEdchi_from_dEdn)).item(), rtol=1e-10))
        print('Density optimization works as expected.')


if __name__ == '__main__':
    unittest.main()
