import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../professad'))
from system import System
from functionals import IonIon, IonElectron, Hartree, WangTeterStyleFunctional, PerdewBurkeErnzerhof


# create system and compute ground state energy
box_len = 3.48
box_vecs = box_len * torch.eye(3, dtype=torch.double)
shape = System.ecut2shape(800, box_vecs)
ions = [['Li', 'li.gga.recpot', box_len * torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]).double()]]
WTexp = WangTeterStyleFunctional((5 / 6, 5 / 6, lambda x: torch.exp(x)))
terms = [IonIon, IonElectron, Hartree, WTexp.forward, PerdewBurkeErnzerhof]
system = System(box_vecs, shape, ions, terms, units='a')

system.optimize_density(1e-10)
energy = system.energy('eV')
print('Initial Energy = {:.4f} eV per atom'.format(energy / system.ion_count()))

# OPTIMIZE IONIC POSITIONS / MINIMIZE FORCES

# peturb ions
print('Perturbing ions ...')
system.place_ions(box_len * torch.tensor([[0.0, 0.1, 0.0], [0.6, 0.4, 0.6]], dtype=torch.double), units='a')
system.optimize_density(1e-10)
print('Perturbed energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

# restore optimal ionic positions by minimizing forces, keeping lattice fixed
print('Performing force minimization ...')
system.optimize_geometry(stol=None, ftol=1e-4, g_method='LBFGSlinesearch', g_verbose=True)
print('Optimized Energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

# OPTIMIZE LATTICE / MINIMIZE STRESS

# predict relaxed energy by fitting to murnaghan equation
print('\nPerforming EOS fit for equilibrium volume ...')
params, err = system.eos_fit(N=5)
relaxed_energy = system.ion_count() * params[2]
print('Equilibrium energy = {:.4f} eV per atom'.format(relaxed_energy / system.ion_count()))

# distort lattice
print('Deforming lattice ...')
tm = torch.tensor([[0.94, -0.03, 0.05],
                   [-0.03, 0.98, 0.04],
                   [0.05, 0.04, 1.05]], dtype=torch.double)
system.set_lattice(torch.matmul(tm, system.lattice_vectors('a').T).T, units='a')
system.optimize_density(1e-10)
print('Perturbed energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

# relax by minimizing stress, keeping box coordinates of ions fixed
print('Performing stress minimization ...')
system.optimize_geometry(ftol=None, stol=1e-4, g_method='LBFGSlinesearch', g_verbose=True)
print('Optimized Energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

# OPTIMIZE GEOMETRY / MINIMIZE FORCES AND STRESS

print('\nPerturbing overall geometry ...')
# peturb ions and distort lattice
tm = torch.tensor([[0.94, -0.03, 0.05],
                   [-0.03, 0.98, 0.04],
                   [0.05, 0.04, 1.05]], dtype=torch.double)
system.place_ions(torch.matmul(tm, system.cartesian_ionic_coordinates('a').T).T, units='a')
system.set_lattice(torch.matmul(tm, system.lattice_vectors('a').T).T, units='a')
system.optimize_density(1e-10)
print('Perturbed energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

# restore optimal geometry by minimizing forces and stress
print('Performing geometry optimization ...')
system.optimize_geometry(stol=1e-4, ftol=1e-4, g_method='LBFGSlinesearch', g_verbose=True)
print('Optimized Energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))
