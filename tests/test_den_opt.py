import numpy as np
import torch
import unittest

from professad.system import System
from professad.functionals import IonElectron, Hartree, Weizsaecker, WangTeter, PerdewBurkeErnzerhof
from professad.ion_utils import PickardSum
from professad.crystal_tools import get_cell


class TestDenOpt(unittest.TestCase):
    def test1_exact_cases(self):
        # using a large box to test non-interacting single orbital systems
        L = 20.0
        box_vecs = L * torch.eye(3, dtype=torch.double)
        shape = System.ecut2shape(250, box_vecs)

        # hydrogen atom
        ions = [["H", "potentials/H.coulomb-kcut-15.recpot", torch.tensor([[0.5, 0.5, 0.5]]).double()]]
        terms = [IonElectron(), Weizsaecker()]
        system = System(box_vecs, shape, ions, terms, units="b", coord_type="fractional")
        system.set_electron_number(1)

        system.optimize_density(ntol=1e-4)
        self.assertAlmostEqual(system.energy("Ha"), -0.5, places=2)

        # quantum harmonic oscillator
        k = 10
        frac = np.stack(
            np.meshgrid(
                np.arange(shape[0]) / shape[0],
                np.arange(shape[1]) / shape[1],
                np.arange(shape[2]) / shape[2],
                indexing="ij",
            ),
            axis=-1,
        )  # (N0, N1, N2, 3)
        cart = np.einsum("...i,ij->...j", frac, box_vecs.numpy())  # (N0, N1, N2, 3)
        qho_pot = 0.5 * k * np.sum((cart - L / 2) ** 2, axis=-1)

        system.set_potential(torch.as_tensor(qho_pot).double())
        system.initialize_density()
        system.optimize_density(ntol=1e-4)
        self.assertAlmostEqual(system.energy("Ha"), 3 / 2 * np.sqrt(k), places=5)

    def test2_check_convergence_measures(self):
        terms = [PickardSum(), IonElectron(), Hartree(), WangTeter(), PerdewBurkeErnzerhof()]
        box_vecs, frac_ion_coords = get_cell("fcc", vol_per_atom=16.8, coord_type="fractional")
        ions = [["Al", "potentials/al.gga.recpot", frac_ion_coords]]
        shape = System.ecut2shape(1600, box_vecs)
        system = System(box_vecs, shape, ions, terms, units="a", coord_type="fractional")
        system.optimize_density(ntol=1e-4)

        dEdchi = system.check_density_convergence()

        dEdn = system.functional_derivative("density")
        chi = torch.sqrt(system.density())
        N_tilde = torch.mean(chi.pow(2)) * system.volume()
        dEdchi_from_dEdn = (
            (system.n_elec / N_tilde)
            * 2
            * chi
            * (dEdn - torch.mean(dEdn * system.density()) * system.volume() / system.n_elec)
        )

        self.assertTrue(np.allclose(dEdchi, torch.max(torch.abs(dEdchi_from_dEdn)).item(), rtol=1e-10))
        print("Density optimization works as expected.")


if __name__ == "__main__":
    unittest.main()
