import numpy as np
import torch
import unittest

from professad.system import System
from professad.functionals import *
from professad.functionals import LocalExchange, _perdew_zunger_correlation, _perdew_wang_correlation, \
    _chachiyo_correlation, PerdewBurkeErnzerhof

from tools_for_tests import *
from professad.functional_tools import get_stress, get_pressure


class TestStress(unittest.TestCase):

    def test1_stress(self):
        # ------- test functional contribution to stress -------
        shape = (25, 25, 25)
        box_vecs = torch.tensor([[6.5, -0.13, 0.25],
                                 [-0.33, 7.21, 0.24],
                                 [0.55, 0.04, 6.78]], dtype=torch.double)
        frac_ion_coords = torch.tensor([[0, 0, 0], [0.35, 0.65, 0.45]], dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', frac_ion_coords]]
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
        #print(ag_stress - th_stress)
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
        ag_stress = get_stress(box_vecs, den, LocalExchange)
        th_stress = lda_exchange_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Perdew-Zunger correlation
        ag_stress = get_stress(box_vecs, den, _perdew_zunger_correlation)
        th_stress = perdew_zunger_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Perdew-Wang correlation
        ag_stress = get_stress(box_vecs, den, _perdew_wang_correlation)
        th_stress = perdew_wang_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # Chachiyo correlation
        ag_stress = get_stress(box_vecs, den, _chachiyo_correlation)
        th_stress = chachiyo_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        # PBE XC
        ag_stress = get_stress(box_vecs, den, PerdewBurkeErnzerhof)
        th_stress = pbe_exchange_stress(box_vecs, den) + pbe_correlation_stress(box_vecs, den)
        self.assertTrue(np.allclose(ag_stress, th_stress, rtol=1e-10))

        p = get_pressure(box_vecs, den, PerdewBurkeErnzerhof).item()
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
