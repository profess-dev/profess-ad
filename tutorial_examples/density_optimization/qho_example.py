import numpy as np
import matplotlib.pyplot as plt
import torch

from professad.system import System
from professad.functionals import IonElectron, Weizsaecker

# the single electron quantum harmonic oscillator (QHO) is a non-interacting
# single-orbital system - hence, it can be modelled well with just the
# ion-electron interaction and Weizsaecker terms
terms = [IonElectron, Weizsaecker]

# use a large box to simulate such localied systems with periodic
# boundary conditions so that the electron density will approach zero
# at the box boundaries
L = 20.0
box_vecs = L * torch.eye(3, dtype=torch.double)

# set low energy cutoff of 300 eV
shape = System.ecut2shape(300, box_vecs)

# as we will set the external potential ourselves later, we just need to
# submit a dummy "ions" parameter (the recpot file and ionic coordinates
# are arbitrary for this example)
ions = [['-', 'al.gga.recpot', torch.tensor([[0.5, 0.5, 0.5]]).double()]]

# create system object
system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')

# as we have used an arbitrary recpot file, we need to set the electron number explicitly
system.set_electron_number(1)

# QHO quadratic potential
k = 10
xf, yf, zf = np.meshgrid(np.arange(shape[0]) / shape[0], np.arange(shape[1]) / shape[1],
                         np.arange(shape[2]) / shape[2], indexing='ij')
x = box_vecs[0, 0] * xf + box_vecs[1, 0] * yf + box_vecs[2, 0] * zf
y = box_vecs[0, 1] * xf + box_vecs[1, 1] * yf + box_vecs[2, 1] * zf
z = box_vecs[0, 2] * xf + box_vecs[1, 2] * yf + box_vecs[2, 2] * zf
r = np.sqrt(x * x + y * y + z * z)
qho_pot = 0.5 * k * ((x - L / 2).pow(2) + (y - L / 2).pow(2) + (z - L / 2).pow(2))

# set external potential to QHO potential
system.set_potential(torch.as_tensor(qho_pot).double())

# perform density optimization
system.optimize_density(ntol=1e-7, n_verbose=True)

# compare optimized energy and the ones expected from elementary quantum mechanics
print('Optimized energy = {:.8f} Ha'.format(system.energy('Ha')))
print('Expected energy = {:.8f} Ha'.format(3 / 2 * np.sqrt(k)))

# check measures of convergence
dEdchi_max = system.check_density_convergence('dEdchi')
mu_minus_dEdn_max = system.check_density_convergence('euler')
print('\nConvergence check:')
print('Max |𝛿E/𝛿χ| = {:.4g}'.format(dEdchi_max))
print('Max |µ-𝛿E/𝛿n| = {:.4g}'.format(mu_minus_dEdn_max))

den = system.density()
pot = system.ionic_potential()


def den_QHO_0(k, x):
    return np.pi**(-1 / 2) * k**(1 / 4) * torch.exp(-k**(1 / 2) * x * x)


den_th = den_QHO_0(k, x - L / 2) * den_QHO_0(k, y - L / 2) * den_QHO_0(k, z - L / 2)

_, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 6), sharex=True)

r_100 = [r[i, 0, 0] for i in range(shape[0])]
r_010 = [r[0, i, 0] for i in range(shape[1])]
r_001 = [r[0, 0, i] for i in range(shape[2])]
r_110 = [r[i, i, 0] for i in range(shape[0])]
r_111 = [r[i, i, i] for i in range(shape[0])]

ax[0][0].plot(r_100, [pot[i, int(shape[1] / 2), int(shape[2] / 2)] for i in range(shape[0])], '-b')
ax[1][0].plot(r_010, [pot[int(shape[0] / 2), i, int(shape[2] / 2)] for i in range(shape[1])], '-b')
ax[2][0].plot(r_001, [pot[int(shape[0] / 2), int(shape[1] / 2), i] for i in range(shape[2])], '-b')
ax[3][0].plot(r_110, [pot[i, i, int(shape[2] / 2)] for i in range(shape[0])], '-b')
ax[4][0].plot(r_111, [pot[i, i, i] for i in range(shape[0])], '-b')

ax[0][1].plot(r_100, [den_th[i, int(shape[1] / 2), int(shape[2] / 2)] for i in range(shape[0])], '-b')
ax[1][1].plot(r_010, [den_th[int(shape[0] / 2), i, int(shape[2] / 2)] for i in range(shape[1])], '-b')
ax[2][1].plot(r_001, [den_th[int(shape[0] / 2), int(shape[1] / 2), i] for i in range(shape[2])], '-b')
ax[3][1].plot(r_110, [den_th[i, i, int(shape[2] / 2)] for i in range(shape[0])], '-b')
ax[4][1].plot(r_111, [den_th[i, i, i] for i in range(shape[0])], '-b')

ax[0][1].plot(r_100, [den[i, int(shape[1] / 2), int(shape[2] / 2)] for i in range(shape[0])], '--r')
ax[1][1].plot(r_010, [den[int(shape[0] / 2), i, int(shape[2] / 2)] for i in range(shape[1])], '--r')
ax[2][1].plot(r_001, [den[int(shape[0] / 2), int(shape[1] / 2), i] for i in range(shape[2])], '--r')
ax[3][1].plot(r_110, [den[i, i, int(shape[2] / 2)] for i in range(shape[0])], '--r')
ax[4][1].plot(r_111, [den[i, i, i] for i in range(shape[0])], '--r')

ax[0][0].set_ylabel('[100]')
ax[1][0].set_ylabel('[010]')
ax[2][0].set_ylabel('[001]')
ax[3][0].set_ylabel('[110]')
ax[4][0].set_ylabel('[111]')

ax[0][0].set_title('Potential')
ax[0][1].set_title('Density')

plt.show()
