from professad.system import System
from professad.functionals import IonIon, IonElectron, Hartree, vWGTF1, PerdewBurkeErnzerhof
from professad.crystal_tools import get_cell

# define the system at a close estimate to the equilibrium volume
terms = [IonIon, IonElectron, Hartree, vWGTF1, PerdewBurkeErnzerhof]
box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.9, coord_type='fractional')
ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
shape = System.ecut2shape(2000, box_vecs)
system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')

# perform the equation of state (EOS) fit

# "f" is the fraction of the equilibrium volume by which the system is stretched or squeezed by
# "N" is the number of energy-volume points used for the EOS fit
# "ntol" is a density optimization tolerace argument
# "eos" specifies the equation of state used for the fit, it can either be 'bm' for Birch-Murnaghan
# or 'm' for Murnaghan
params, err = system.eos_fit(f=0.05, N=11, ntol=1e-7, eos='bm')
K0, K0prime, E0, V0 = params  # unpack params

print('Bulk modulus, K₀ = {:.5g} GPa'.format(K0))
print('Bulk modulus derivative (wrt pressure), K₀\' = {:.5g}'.format(K0prime))
print('Equilibrium energy, E₀ = {:.5g} eV per atom'.format(E0))
print('Equilibrium volume, V₀ = {:.5g} A³ per atom'.format(V0))
