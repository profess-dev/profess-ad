import numpy as np
import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from system import System
from functionals import IonIon, IonElectron, Hartree, WangTeter, SmargiassiMadden, PerdewBurkeErnzerhof


class TestProfess(unittest.TestCase):

    def test_fcc_aluminium_against_profess4(self):
        shape = (18, 18, 18)
        box_vecs = 4.050 * torch.tensor([[0.5, 0.5, 0.0],
                                         [0.0, 0.5, 0.5],
                                         [0.5, 0.0, 0.5]], dtype=torch.double)
        frac_ion_coords = torch.tensor([[0, 0, 0]], dtype=torch.double)
        ions = [['Al', 'potentials/al.gga.recpot', frac_ion_coords]]

        terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        system.optimize_density(ntol=1e-7)
        energy = system.energy('eV')
        self.assertTrue(np.allclose(energy, -57.183329401794985, atol=1e-4))

    def test_bcc_lithium_against_profess4(self):
        shape = (18, 18, 18)
        box_vecs = 3.48 * torch.eye(3, dtype=torch.double)
        frac_ion_coords = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=torch.double)
        ions = [['Li', 'potentials/li.gga.recpot', frac_ion_coords]]

        terms = [IonIon, IonElectron, Hartree, SmargiassiMadden, PerdewBurkeErnzerhof]
        system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
        system.optimize_density(ntol=1e-7)
        energy = system.energy('eV')
        self.assertTrue(np.allclose(energy, -14.741886997024537, atol=1e-4))

        print('Single point energies match that of PROFESS 4.0.')


if __name__ == '__main__':
    unittest.main()
