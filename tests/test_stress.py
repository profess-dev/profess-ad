import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../professad'))
from system import System
from functionals import *

from tools_for_tests import *
from functional_tools import get_stress, get_pressure


class TestStress(unittest.TestCase):

    def test1_stress(self):

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
        eps = 1e-4
        for i in range(3):
            for j in range(3):
                perturbation = torch.zeros((3, 3), dtype=torch.double)
                perturbation[i, j] += eps
                lat_vec_plus = torch.matmul((torch.eye(3, dtype=torch.double)
                                             + perturbation), lattice_vectors.T).T
                system.set_lattice(lat_vec_plus, units='a')
                E_plus[i, j] = system.energy('eV')
                lat_vec_minus = torch.matmul((torch.eye(3, dtype=torch.double)
                                              - perturbation), lattice_vectors.T).T
                system.set_lattice(lat_vec_minus, units='a')
                E_minus[i, j] = system.energy('eV')
        finite_diff_stress = (E_plus - E_minus).numpy() / (2 * eps * volume)

        self.assertTrue(np.allclose(autograd_stress, finite_diff_stress, atol=1e-3))

        # ------- test functional contribution to stress -------

        # perform a density optimization to obtain a density profile for testing

        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')
        system.optimize_density(ntol=1e-8)
        den = system.density()
        box_vecs = system.lattice_vectors()

        # hartree
        ag_stress = get_stress(box_vecs, den, Hartree)
        th_stress = hartree_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # ------- Kinetic Functionals -------
        # Thomas-Fermi
        ag_stress = get_stress(box_vecs, den, ThomasFermi)
        th_stress = TF_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Weizsaecker
        ag_stress = get_stress(box_vecs, den, Weizsaecker)
        th_stress = vW_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Wang-Teter
        ag_stress = get_stress(box_vecs, den, WangTeter)
        th_stress = non_local_KEF_stress(box_vecs, den, alpha=5 / 6, beta=5 / 6)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        p = get_pressure(box_vecs, den, WangTeter).item()
        p_from_stress = - torch.trace(ag_stress).item() / 3
        self.assertTrue(np.allclose(p, p_from_stress, rtol=1e-10))

        # exponential stabilized Wang-Teter
        WTexp = WangTeterStyleFunctional((5 / 6, 5 / 6, lambda x: torch.exp(x)))
        ag_stress = get_stress(box_vecs, den, WTexp.forward)
        th_stress = pauli_stabilized_stress(box_vecs, den, alpha=5 / 6, beta=5 / 6,
                                            f=lambda x: torch.exp(x), fprime=lambda x: torch.exp(x))
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        p = get_pressure(box_vecs, den, WTexp.forward).item()
        p_from_stress = - torch.trace(ag_stress).item() / 3
        self.assertTrue(np.allclose(p, p_from_stress, rtol=1e-10))

        # Perrot
        ag_stress = get_stress(box_vecs, den, Perrot)
        th_stress = non_local_KEF_stress(box_vecs, den, alpha=1, beta=1)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Smargiassi-Madden
        ag_stress = get_stress(box_vecs, den, SmargiassiMadden)
        th_stress = non_local_KEF_stress(box_vecs, den, alpha=0.5, beta=0.5)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Wang-Govind-Carter 98
        ag_stress = get_stress(box_vecs, den, WangGovindCarter98)
        th_stress = non_local_KEF_stress(box_vecs, den, alpha=(5 + np.sqrt(5)) / 6, beta=(5 - np.sqrt(5)) / 6)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # exponential stabilized Wang-Govind-Carter 98
        WGC98exp = WangTeterStyleFunctional(((5 + np.sqrt(5)) / 6, (5 - np.sqrt(5)) / 6, lambda x: torch.exp(x)))
        ag_stress = get_stress(box_vecs, den, WGC98exp.forward)
        th_stress = pauli_stabilized_stress(box_vecs, den, alpha=(5 + np.sqrt(5)) / 6, beta=(5 - np.sqrt(5)) / 6,
                                            f=lambda x: torch.exp(x), fprime=lambda x: torch.exp(x))
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # ------- XC Functionals -------
        # LDA exchange
        ag_stress = get_stress(box_vecs, den, lda_exchange)
        th_stress = lda_exchange_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Perdew-Zunger correlation
        ag_stress = get_stress(box_vecs, den, perdew_zunger_correlation)
        th_stress = perdew_zunger_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Perdew-Wang correlation
        ag_stress = get_stress(box_vecs, den, perdew_wang_correlation)
        th_stress = perdew_wang_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Chachiyo correlation
        ag_stress = get_stress(box_vecs, den, chachiyo_correlation)
        th_stress = chachiyo_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # PBE exchange
        ag_stress = get_stress(box_vecs, den, pbe_exchange)
        th_stress = pbe_exchange_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        p = get_pressure(box_vecs, den, pbe_exchange).item()
        p_from_stress = - torch.trace(ag_stress).item() / 3
        self.assertTrue(np.allclose(p, p_from_stress, rtol=1e-10))

        # PBE correlation
        ag_stress = get_stress(box_vecs, den, pbe_correlation)
        th_stress = pbe_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        p = get_pressure(box_vecs, den, pbe_correlation).item()
        p_from_stress = - torch.trace(ag_stress).item() / 3
        self.assertTrue(np.allclose(p, p_from_stress, rtol=1e-10))

        print('Autograd stresses work as expected.')

    def test2_pressure_stress(self):
        box_vecs = torch.tensor([[3.54, -0.13, 0.25],
                                 [-0.33, 3.82, 0.24],
                                 [0.55, 0.04, 3.45]], dtype=torch.double)
        shape = System.ecut2shape(1000, box_vecs)
        frac_ion_coords = torch.tensor([[0, 0, 0], [0.35, 0.65, 0.45]], dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', frac_ion_coords]]
        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        system.optimize_density()

        pressure = system.pressure('GPa')
        stress = system.stress('GPa')
        pressure_from_stress = - torch.trace(stress).item() / 3

        self.assertAlmostEqual(pressure, pressure_from_stress, places=6)


if __name__ == '__main__':
    unittest.main()
