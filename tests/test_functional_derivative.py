import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from system import System
from functionals import *

from tools_for_tests import *
from functional_tools import get_functional_derivative
from crystal_tools import get_cell


class TestFunctionalDerivatives(unittest.TestCase):

    def test1_functional_derivatives(self):
        # perform a density optimization to obtain a density profile for testing
        shape = (25, 25, 25)
        box_len = 6.96
        box_vecs = box_len * torch.eye(3, dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', box_len * torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]).double()]]
        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        system = System(box_vecs, shape, ions, terms, units='b')
        system.optimize_density()
        box_vecs = system.lattice_vectors()
        den = system.density()
        v_ext = system.ionic_potential()

        # compute autograd (ag) and theoretical (th) functional derivatives (fd) for comparison

        # ion-electron interaction
        ag_fd = get_functional_derivative(box_vecs, den, lambda bv, n: IonElectron(bv, n, v_ext))
        th_fd = v_ext
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Hartree energy
        ag_fd = get_functional_derivative(box_vecs, den, Hartree)
        th_fd = hartree_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # ------- Kinetic Functionals -------
        # Thomas-Fermi
        ag_fd = get_functional_derivative(box_vecs, den, ThomasFermi)
        th_fd = TF_kp(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Weizsaecker
        ag_fd = get_functional_derivative(box_vecs, den, Weizsaecker)
        th_fd = vW_kp(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Wang-Teter
        ag_fd = get_functional_derivative(box_vecs, den, WangTeter)
        th_fd = non_local_KEFD(box_vecs, den, alpha=5 / 6, beta=5 / 6)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Perrot
        ag_fd = get_functional_derivative(box_vecs, den, Perrot)
        th_fd = non_local_KEFD(box_vecs, den, alpha=1, beta=1)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Smargiassi-Madden
        ag_fd = get_functional_derivative(box_vecs, den, SmargiassiMadden)
        th_fd = non_local_KEFD(box_vecs, den, alpha=0.5, beta=0.5)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Wang-Govind-Carter 98
        ag_fd = get_functional_derivative(box_vecs, den, WangGovindCarter98)
        th_fd = non_local_KEFD(box_vecs, den, alpha=(5 + np.sqrt(5)) / 6, beta=(5 - np.sqrt(5)) / 6)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # LKT
        ag_fd = get_functional_derivative(box_vecs, den, LuoKarasievTrickey)
        th_fd = LKT_kp(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        pg = PauliGaussian()
        # PG1
        pg.set_PG1()
        ag_fd = get_functional_derivative(box_vecs, den, pg.forward)
        th_fd = PG1_kp(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # PGSL0.25
        pg.set_PGSL025()
        ag_fd = get_functional_derivative(box_vecs, den, pg.forward)
        th_fd = PGSL_kp(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # ------- XC Functionals -------
        # LDA exchange
        ag_fd = get_functional_derivative(box_vecs, den, lda_exchange)
        th_fd = lda_exchange_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Perdew-Zunger correlation
        ag_fd = get_functional_derivative(box_vecs, den, perdew_zunger_correlation)
        th_fd = perdew_zunger_correlation_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Perdew-Wang correlation
        ag_fd = get_functional_derivative(box_vecs, den, perdew_wang_correlation)
        th_fd = perdew_wang_correlation_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # Chachiyo correlation
        ag_fd = get_functional_derivative(box_vecs, den, chachiyo_correlation)
        th_fd = chachiyo_correlation_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # PBE exchange
        ag_fd = get_functional_derivative(box_vecs, den, pbe_exchange)
        th_fd = pbe_exchange_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

        # PBE correlation
        ag_fd = get_functional_derivative(box_vecs, den, pbe_correlation)
        th_fd = pbe_correlation_potential(box_vecs, den)
        self.assertTrue(np.allclose(ag_fd, th_fd, rtol=1e-10))

    def test2_density_optimization(self):
        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.8, coord_type='fractional')
        ions = [['Al', 'potentials/al.gga.recpot', frac_ion_coords]]
        shape = System.ecut2shape(1600, box_vecs)
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        system.optimize_density()
        E1, den1 = system.energy(), system.density()
        system.initialize_density()

        def dEdn(bv, n):
            return system.ionic_potential() + hartree_potential(bv, n) \
                   + non_local_KEFD(bv, n, alpha=5 / 6, beta=5 / 6) \
                   + pbe_exchange_potential(bv, n) + pbe_correlation_potential(bv, n)

        system.optimize_density(potentials=dEdn)
        E2, den2 = system.energy(), system.density()
        self.assertTrue(np.allclose(E1, E2, rtol=1e-7))
        self.assertTrue(np.allclose(den1, den2, atol=1e-5))
        print('Autograd functional derivatives work as expected.')


if __name__ == '__main__':
    unittest.main()
