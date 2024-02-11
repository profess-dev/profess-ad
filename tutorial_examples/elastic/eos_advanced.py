from professad.system import System
from professad.functionals import IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof
from professad.crystal_tools import get_cell

f = 0.05
N = 11
energy_cutoff = 2000  # eV
tol = 1e-7

print('Performing aluminium calculations with Wang-Teter KE and PBE XC functionals.')
print('Energy cut-off = {} eV with density optimization tolerance {}'.format(energy_cutoff, tol))
print('Stretched up to ±{}% over {} points and fitted with the {} EOS\n'
      .format(f * 100, N, 'Birch-Murnaghan'))

terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
crystal_predv0s = [('fcc', 16.8), ('hcp', 16.9), ('bcc', 17.2), ('sc', 19.9), ('dc', 28.8)]

print('{:^8} {:^17} {:^17} {:^14} {:^14}'.format('Crystal', 'V₀/A³ per atom', 'E₀/eV per atom', 'ΔE₀/meV', 'K₀/GPa'))
for crystal, pred_v0 in crystal_predv0s:
    box_vecs, frac_ion_coords = get_cell(crystal, vol_per_atom=pred_v0, c_over_a=1.66, coord_type='fractional')
    ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
    shape = System.ecut2shape(energy_cutoff, box_vecs)
    system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
    params, err = system.eos_fit(f=f, N=N, ntol=tol, eos='bm')
    K0, K0prime, E0, V0 = params
    if crystal == 'fcc':
        E_fcc = E0
    print('{:^8} {:^17.5f} {:^17.5f} {:^14.2f} {:^14.5f}'.format(crystal, V0, E0, (E0 - E_fcc) * 1e3, K0))
