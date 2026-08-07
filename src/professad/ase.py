import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from professad.system import System


class ProfessAD(Calculator):
    """
    PROFESS-AD ASE calculator.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, system: System, ntol=1e-12):
        Calculator.__init__(self)
        self.results = {}
        self.system = system
        self.ntol = ntol

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)

        # lattice must be set first, as place_ions maps Cartesian coordinates
        # to fractional ones using the current lattice vectors
        self.system.set_lattice(torch.as_tensor(np.asarray(atoms.cell), dtype=torch.double))
        self.system.place_ions(torch.as_tensor(atoms.positions, dtype=torch.double))
        self.system.optimize_density(ntol=self.ntol)

        self.results["energy"] = self.system.energy("eV")
        self.results["forces"] = self.system.forces("eV/a").cpu().numpy()
        self.results["stress"] = self.system.stress(units="eV/a3", voigt=True).cpu().numpy()
