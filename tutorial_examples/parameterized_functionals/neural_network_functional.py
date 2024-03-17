import numpy as np
import matplotlib.pyplot as plt
import torch

from professad.system import System
from professad.functionals import KineticFunctional, Weizsaecker, IonIon, IonElectron, Hartree, \
    WangTeter, PerdewBurkeErnzerhof
from professad.functional_tools import get_functional_derivative, wavevectors, reduced_gradient, reduced_laplacian
from professad.crystal_tools import get_cell


# create semi-local neural network functional
class NeuralNetworkFunctional(KineticFunctional):

    def __init__(self, inner_layer_sizes):

        super().__init__()
        self.init_args = inner_layer_sizes
        layer_sizes = [2] + self.init_args + [1]

        self.nn = torch.nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.nn.add_module('Linear_{}'.format(i), torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1],
                               dtype=torch.double))
            if i != len(layer_sizes) - 2:
                self.nn.add_module('Activation_{}'.format(i), torch.nn.SiLU())
        self.nn.add_module('Activation_-1', torch.nn.Softplus())
        self.initialize()

    def forward(self, box_vecs, den):
        # getting descriptors
        kxyz = wavevectors(box_vecs, den.shape)
        s = reduced_gradient(kxyz, den)
        q = reduced_laplacian(kxyz.square().sum(-1), den)

        # compute Pauli enhancement factor
        Fenh = torch.squeeze(self.nn(torch.cat((s.unsqueeze(-1), q.unsqueeze(-1)), dim=-1)))

        # assembling the terms
        vol = torch.abs(torch.linalg.det(box_vecs))
        TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den.pow(5 / 3)
        Pauli_T = torch.mean(Fenh * TF_ked) * vol
        return Weizsaecker(box_vecs, den) + Pauli_T


# generate an optimized density to be used
terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.9, coord_type='fractional')
ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
shape = System.ecut2shape(2000, box_vecs)
system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
system.optimize_density(ntol=1e-10)

# extract optimized density and lattice vectors (in atomic units)
den = system.density()
box_vecs = system.lattice_vectors('b')

# as a toy example, let's try to fit the neural network functional's kinetic potential
# to that of the Wang-Teter functional for this optimized density

# the "target"
WT_kp = get_functional_derivative(box_vecs, den, WangTeter)

# create the neural network functional with 3 hidden layers having 8 nodes each
model = NeuralNetworkFunctional([8, 8, 8])

# explicitly defining the optimizer and its parameters
model.optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, step_sizes=(1e-8, 50))

model.param_grad(True)

for iter in range(201):
    NN_kp = get_functional_derivative(box_vecs, den, model.forward, requires_grad=True)
    loss = model.grid_error(WT_kp, NN_kp)
    model.update_params(loss)
    if iter % 20 == 0:
        print('Iteration {}, RMSE = {:.5g}'.format(iter, torch.sqrt(loss).item()))

model.param_grad(False)

NN_kp = get_functional_derivative(box_vecs, den, model.forward)

# make plot
plt.rc('font', family='serif')
fig, axs = plt.subplots(figsize=(5, 5), nrows=2, sharex=True, gridspec_kw={'hspace': 0})

r = np.linspace(0, 1, shape[0])
axs[0].plot(r, [den[i, i, i] for i in range(den.shape[0])], '-k')

axs[1].plot(r, [WT_kp[i, i, i] for i in range(den.shape[0])], '-b')
axs[1].plot(r, [NN_kp[i, i, i] for i in range(den.shape[0])], '--r')

axs[0].set_xlim([0, 1])
axs[1].set_xlim([0, 1])

axs[0].set_ylabel('Electron Density (a.u.)')
axs[1].set_ylabel('Kinetic Potential (a.u.)')
axs[1].set_xlabel(r'[111] direction ($a_0 \sqrt{3}$)')

labels = ['WT', 'NN']
plt.legend(labels=labels, loc="lower center", borderaxespad=0.4, ncol=1, prop={'size': 10})

plt.tight_layout()
plt.show()
