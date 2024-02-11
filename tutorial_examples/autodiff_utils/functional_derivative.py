import numpy as np
import matplotlib.pyplot as plt

from professad.system import System
from professad.functionals import IonIon, IonElectron, Hartree, WangTeter, ThomasFermi, \
    Weizsaecker, PauliGaussian, PerdewBurkeErnzerhof
from professad.crystal_tools import get_cell
from professad.functional_tools import get_functional_derivative

# generate an optimized density to be used
terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.9, coord_type='fractional')
ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
shape = System.ecut2shape(3500, box_vecs)
system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
system.optimize_density(ntol=1e-10)

# extract optimized density and lattice vectors (in atomic units)
den = system.density()
box_vecs = system.lattice_vectors('b')

# compute functional derivatives (or kinetic potentials)
WT_kp = get_functional_derivative(box_vecs, den, WangTeter)

TFvW = lambda bv, n: ThomasFermi(bv, n) + 1 / 9 * Weizsaecker(bv, n)
TFvW_kp = get_functional_derivative(box_vecs, den, TFvW)

pg = PauliGaussian()
pg.set_PGS()
PG_kp = get_functional_derivative(box_vecs, den, pg.forward)

# make plot
plt.rc('font', family='serif')
fig, axs = plt.subplots(figsize=(5, 5), nrows=2, sharex=True, gridspec_kw={'hspace': 0})

r = np.linspace(0, 1, shape[0])
axs[0].plot(r, [den[i, i, i] for i in range(den.shape[0])], '-k')

axs[1].plot(r, [WT_kp[i, i, i] for i in range(den.shape[0])], '-b')
axs[1].plot(r, [TFvW_kp[i, i, i] for i in range(den.shape[0])], '--r')
axs[1].plot(r, [PG_kp[i, i, i] for i in range(den.shape[0])], ':g', linewidth=2)

axs[0].set_xlim([0, 1])
axs[1].set_xlim([0, 1])

axs[0].set_ylabel('Electron Density (a.u.)')
axs[1].set_ylabel('Kinetic Potential (a.u.)')
axs[1].set_xlabel(r'[111] direction ($a_0 \sqrt{3}$)')

labels = ['WT', 'TF(1/9)vW', 'PGS']
plt.legend(labels=labels, loc="lower center", borderaxespad=0.4, ncol=1, prop={'size': 10})

plt.tight_layout()
plt.show()
