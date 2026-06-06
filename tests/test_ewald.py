import numpy as np
import torch
import unittest

from professad.ion_utils import EwaldSum


def _ewald_params(box_vecs):
    """Compute reasonable Ewald parameters for a given cell."""
    interplanar_dist = 1 / torch.sqrt(torch.sum(torch.linalg.inv(box_vecs.T).pow(2), 1))
    h_max = torch.max(interplanar_dist).item()
    alpha = np.sqrt(np.pi) / h_max
    Rc = 5.25 / alpha
    k_cutoff = 10.5 * alpha
    return alpha, Rc, k_cutoff


class TestEwald(unittest.TestCase):
    def _make_systems(self):
        """Return the 5 test systems (same as test_ion_utils) with CASTEP reference energies."""
        systems = []

        # Al
        box_vecs = torch.tensor(
            [
                [5.41141973394663, 0.00000000000000, 0.00000000000000],
                [2.70570986697332, 4.68642696013821, 0.00000000000000],
                [2.70570986697332, 1.56214232004608, 4.41840571073226],
            ],
            dtype=torch.float64,
        )
        coords = torch.zeros((1, 3), dtype=torch.float64)
        charges = 3.0 * torch.ones((1,), dtype=torch.float64)
        systems.append(("Al", box_vecs, coords, charges, -2.69595457432924945))

        # Si
        box_vecs = torch.tensor(
            [
                [7.25654832321381, 0.00000000000000, 0.00000000000000],
                [3.62827416160690, 6.28435519169252, 0.00000000000000],
                [3.62827416160690, 2.09478506389751, 5.92494689524090],
            ],
            dtype=torch.float64,
        )
        coords = torch.tensor([[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]], dtype=torch.float64) @ box_vecs
        charges = 4.0 * torch.ones(2, dtype=torch.float64)
        systems.append(("Si", box_vecs, coords, charges, -8.39857465282205418))

        # SiO2
        box_vecs = torch.tensor(
            [
                [9.28422445623683, 0.00000000000000, 0.00000000000000],
                [-4.64211222811842, 8.04037423353787, 0.00000000000000],
                [0.00000000000000, 0.00000000000000, 10.2139697101486],
            ],
            dtype=torch.float64,
        )
        coords = (
            torch.tensor(
                [
                    [0.41500, 0.27200, 0.21300],
                    [0.72800, 0.14300, 0.54633],
                    [0.85700, 0.58500, 0.87967],
                    [0.27200, 0.41500, 0.78700],
                    [0.14300, 0.72800, 0.45367],
                    [0.58500, 0.85700, 0.12033],
                    [0.46500, 0.00000, 0.33333],
                    [0.00000, 0.46500, 0.66667],
                    [0.53500, 0.53500, 0.00000],
                ],
                dtype=torch.float64,
            )
            @ box_vecs
        )
        charges = 6.0 * torch.ones(9, dtype=torch.float64)
        charges[6:] = 4.0
        systems.append(("SiO2", box_vecs, coords, charges, -69.48809871723248932))

        # Al2SiO5
        box_vecs = torch.tensor(
            [
                [14.7289033699982, 0.00000000000000, 0.00000000000000],
                [0.00000000000000, 14.9260018049230, 0.00000000000000],
                [0.00000000000000, 0.00000000000000, 10.5049875335275],
            ],
            dtype=torch.float64,
        )
        coords = (
            torch.tensor(
                [
                    [0.23030, 0.13430, 0.23900],
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
                    [0.60380, 0.09870, 0.50000],
                ],
                dtype=torch.float64,
            )
            @ box_vecs
        )
        charges = 6.0 * torch.ones(32, dtype=torch.float64)
        charges[8:13] = 3.0
        charges[14] = 3.0
        charges[16] = 3.0
        charges[18] = 3.0
        charges[20] = 4.0
        charges[23] = 4.0
        charges[26] = 4.0
        charges[29] = 4.0
        systems.append(("Al2SiO5", box_vecs, coords, charges, -244.05500850908111943))

        # NaCl (Madelung constant test)
        box_vecs = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float64)
        systems.append(("NaCl", box_vecs, None, None, -1.747564594633))

        return systems

    def test_vanilla_ewald_vs_castep(self):
        """Vanilla Ewald matches CASTEP reference energies."""
        systems = self._make_systems()

        for name, box_vecs, coords, charges, E_ref in systems:
            with self.subTest(system=name):
                alpha, Rc, k_cutoff = _ewald_params(box_vecs)
                ewald = EwaldSum(alpha=alpha, Rc=Rc, k_cutoff=k_cutoff)
                if name == "NaCl":
                    coords_fcc = torch.zeros((1, 3), dtype=torch.float64)
                    charges_fcc = torch.ones(1, dtype=torch.float64)
                    E_FCC = ewald(box_vecs, coords_fcc, charges_fcc)

                    coords_2 = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64)
                    charges_2 = torch.ones(2, dtype=torch.float64)
                    E_2 = ewald(box_vecs, coords_2, charges_2)

                    self.assertTrue(
                        np.abs((4 * E_FCC - E_2).item() - E_ref) < 1e-10,
                        f"NaCl Madelung: {(4 * E_FCC - E_2).item()} vs {E_ref}",
                    )
                else:
                    E = ewald(box_vecs, coords, charges)
                    err = np.abs(E.item() - E_ref) / coords.shape[0]
                    self.assertTrue(err < 1e-10, f"{name}: per-atom error {err:.2e}")

    def test_pme_ewald_vs_vanilla(self):
        """PME Ewald matches vanilla Ewald on all systems."""
        systems = self._make_systems()

        for name, box_vecs, coords, charges, E_ref in systems:
            with self.subTest(system=name):
                alpha, Rc, k_cutoff = _ewald_params(box_vecs)
                if name == "NaCl":
                    coords = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64)
                    charges = torch.ones(2, dtype=torch.float64)

                E_vanilla = EwaldSum(alpha=alpha, Rc=Rc, k_cutoff=k_cutoff)(box_vecs, coords, charges)
                shape = (30, 30, 30)
                E_pme = EwaldSum(alpha=alpha, Rc=Rc, shape=shape, pme_order=20)(box_vecs, coords, charges)
                err = np.abs(E_vanilla.item() - E_pme.item()) / coords.shape[0]
                self.assertTrue(err < 1e-8, f"{name}: per-atom PME vs vanilla error {err:.2e}")


if __name__ == "__main__":
    unittest.main()
