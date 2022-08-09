import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../professad'))
from system import System
from functionals import IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof
from crystal_tools import get_cell

# set energy terms and functionals to be used
terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]

# use "get cell" to get the lattice vectors and fractional ionic coordinates of
# a face-centred cubic (fcc) lattice
box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=24.8, coord_type='fractional')

# defining the ions in the system
ions = [['Al', 'al.gga.recpot', frac_ion_coords]]

# set plane-wave cutoff at 2000 eV
shape = System.ecut2shape(2000, box_vecs)

# create an fcc-aluminium system object
system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')

# perform density optimization (by default n_verbose is False, but we want to
# display the progress of the density optimization)
system.optimize_density(ntol=1e-7, conv_target='dE', n_method='LBFGS', n_verbose=True)

# check the measures of convergence
dEdchi_max = system.check_density_convergence('dEdchi')
mu_minus_dEdn_max = system.check_density_convergence('euler')

print('Convergence check:')
print('Max |𝛿E/𝛿χ| = {:.4g}'.format(dEdchi_max))
print('Max |µ-𝛿E/𝛿n| = {:.4g}'.format(mu_minus_dEdn_max))
