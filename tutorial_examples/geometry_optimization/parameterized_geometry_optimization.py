import numpy as np
import torch

from professad.system import System
from professad.functionals import IonIon, IonElectron, Hartree, WangTeterStyleFunctional, PerdewBurkeErnzerhof

# use GPU if available else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = torch.tensor([24 / System.A_per_b**3, 1.5], dtype=torch.double, device=device).requires_grad_()
print('Initial Guess: Volume per atom = {:.5f} Å³, c/a = {:.5f}'
      .format(params[0].item() * System.A_per_b**3, params[1].item()))


# define the lattice vectors and fractional ionic coordinates as a function of the parameters
def parameterized_geometry(params):
    vol_per_atom, c_over_a = params
    a = ((2 * torch.abs(vol_per_atom)) / (np.sqrt(3) / 2 * c_over_a)).pow(1 / 3)
    box_vecs = torch.tensor([[1.0, 0.0, 0.0],
                             [-0.5, np.sqrt(3) / 2, 0.0],
                             [0.0, 0.0, 0.0]], dtype=torch.double, device=device)
    box_vecs[2, 2] = torch.abs(c_over_a)
    box_vecs = a * box_vecs
    frac_ion_coords = torch.tensor([[1 / 3, 2 / 3, 3 / 4],
                                    [2 / 3, 1 / 3, 1 / 4]], dtype=torch.double, device=device)
    return box_vecs, frac_ion_coords


box_vecs, frac_ion_coords = parameterized_geometry(params)

WTexp = WangTeterStyleFunctional((5 / 6, 5 / 6, lambda x: torch.exp(x)))
# required for GPU usage with functionals that inherit from the KineticFunctional class
WTexp.set_device(device)
terms = [IonIon, IonElectron, Hartree, WTexp.forward, PerdewBurkeErnzerhof]

# construct the system object
ions = [['Mg', 'mg.gga.recpot', frac_ion_coords.detach()]]
# lattice vectors must be in angstroms for ecut2shape
shape = System.ecut2shape(2000, box_vecs.detach() * System.A_per_b)
system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional', device=device)


# define a print statement to track how the parameters evolve over the optimization
def param_string(params):
    return '{:.5f} {:.5f}'.format(params[0].item() * System.A_per_b**3, params[1].item())


system.optimize_parameterized_geometry(params, parameterized_geometry, g_method='LBFGSlinesearch',
                                       g_verbose=True, param_string=param_string, ftol=1e-3, stol=1e-3)
print('Optimized Results: Volume per atom = {:.5f} Å³, c/a = {:.5f}\n'
      .format(params[0].item() * System.A_per_b**3, params[1].item()))
