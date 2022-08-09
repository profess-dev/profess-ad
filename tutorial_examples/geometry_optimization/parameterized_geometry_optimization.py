import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../professad'))
from system import System
from functionals import IonIon, IonElectron, Hartree, WangTeterStyleFunctional, PerdewBurkeErnzerhof


params = torch.tensor([23.1 / System.A_per_b**3, 1.6], dtype=torch.double).requires_grad_()
print('Initial Guess: Volume per atom = {:.5f} Å³, c/a = {:.5f}'
      .format(params[0].item() * System.A_per_b**3, params[1].item()))


# define the lattice vectors and fractional ionic coordinates as a function of the parameters
def parameterized_geometry(params):
    vol_per_atom, c_over_a = params
    a = ((2 * vol_per_atom) / (np.sqrt(3) / 2 * c_over_a)).pow(1 / 3)
    box_vecs = torch.tensor([[1.0, 0.0, 0.0],
                             [-0.5, np.sqrt(3) / 2, 0.0],
                             [0.0, 0.0, 0.0]], dtype=torch.double)
    box_vecs[2, 2] = c_over_a
    box_vecs = a * box_vecs
    frac_ion_coords = torch.tensor([[1 / 3, 2 / 3, 3 / 4],
                                    [2 / 3, 1 / 3, 1 / 4]], dtype=torch.double)
    return box_vecs, frac_ion_coords


box_vecs, frac_ion_coords = parameterized_geometry(params)

# construct the system object
WTexp = WangTeterStyleFunctional((5 / 6, 5 / 6, lambda x: torch.exp(x)))
terms = [IonIon, IonElectron, Hartree, WTexp.forward, PerdewBurkeErnzerhof]
ions = [['Mg', 'mg.gga.recpot', frac_ion_coords.detach()]]
shape = System.ecut2shape(800, box_vecs.detach())
system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')


# define a print statement to track how the parameters evolve over the optimization
def param_string(params):
    return '{:.5f} {:.5f}'.format(params[0].item() * System.A_per_b**3, params[1].item())


system.optimize_parameterized_geometry(params, parameterized_geometry, g_method='LBFGSlinesearch',
                                       g_verbose=True, param_string=param_string)
print('Optimized Results: Volume per atom = {:.5f} Å³, c/a = {:.5f}\n'
      .format(params[0].item() * System.A_per_b**3, params[1].item()))
