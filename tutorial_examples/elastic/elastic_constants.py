from professad.system import System
from professad.functionals import IonIon, IonElectron, Hartree, XuWangMa, PerdewBurkeErnzerhof
from professad.crystal_tools import get_cell
from professad.elastic_tools import shear_average, poissons_ratio


# define system
terms = [IonIon, IonElectron, Hartree, XuWangMa, PerdewBurkeErnzerhof]
box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.52, coord_type='fractional')
ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
shape = System.ecut2shape(2000, box_vecs)
system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')


# perform Birch-Murnaghan fit to determine equilibrium volume and bulk modulus
params, err = system.eos_fit(f=0.05, N=11, ntol=1e-10, eos='bm')
K0, K0prime, E0, V0 = params

print('Birch-Murnaghan fit results:')
print('Equilibrium volume = {:.5g} Å³'.format(V0))
print('Equilibrium bulk modulus = {:.5g} GPa'.format(K0))

# set system to equilibrium volume
box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=V0, coord_type='fractional')
system.set_lattice(box_vecs, units='a')

# use higher density optimization tolerance for second-derivative calculations
system.optimize_density(ntol=1e-10)

# check if pressure is zero
pressure = system.pressure('GPa')
print('Pressure = {:.5g} GPa (expect zero pressure at equilibrium volume)'.format(pressure))

# get auto-differentiated elastic constants
Cs = system.elastic_constants('GPa')

print('\nElastic constants from auto-differentiation:')
print('C11 = {:.5g} GPa'.format(Cs[0, 0].item()))
print('C12 = {:.5g} GPa'.format(Cs[0, 1].item()))
print('C44 = {:.5g} GPa'.format(Cs[3, 3].item()))

# compute bulk modulus from elastic constants
# for cubic systems, K = (C11 + 2 * C12) / 3
K_ec = (Cs[0, 0].item() + 2 * Cs[0, 1].item()) / 3

# get auto-differentiated bulk modulus
K_ad = system.bulk_modulus('GPa')

print('\nCompare bulk moduli:')
print('EOS bulk modulus = {:.5g} GPa'.format(K0))
print('Auto-differentiation bulk modulus = {:.5g} GPa'.format(K_ad))
print('Bulk modulus from elastic constants = {:.5g} GPa'.format(K_ec))

# post-process elastic constants matrix to shear modulus and poisson's ratio
G = shear_average(Cs, mean_type='arithmetic')
v = poissons_ratio(K_ec, G)

print('\nPost-processed quantities:')
print('Shear modulus = {:.5g}'.format(G))
print('Poisson\'s ratio = {:.5g}'.format(v))
