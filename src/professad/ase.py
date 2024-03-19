import torch
from ase.calculators.calculator import Calculator, all_changes
from professad.system import System


class ProfessAD(Calculator):
    """
    PROFESS-AD ASE calculator.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, system: System):
        Calculator.__init__(self)
        self.results = {}
        self.system = system

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):

        Calculator.calculate(self, atoms)

        self.system.set_lattice(torch.as_tensor(atoms.cell))
        self.system.place_ions(torch.as_tensor(atoms.positions))
        self.system.optimize_density(ntol=1e-12, n_verbose=False)

        self.results['energy'] = self.system.energy('eV')
        self.results['forces'] = self.system.forces('eV/a').cpu().numpy()
        self.results['stress'] = self.system.stress(units='eV/a3', voigt=True).cpu().numpy()
