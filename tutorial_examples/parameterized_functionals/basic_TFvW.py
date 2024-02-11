import matplotlib.pyplot as plt
import torch

from professad.functionals import KineticFunctional, Weizsaecker, ThomasFermi, G_inv_lindhard
from professad.functional_tools import get_inv_G


# define the µ TF + λ vW functional
class TFvW(KineticFunctional):

    def __init__(self, init_args=None):
        super().__init__()
        # init_args is a tuple
        if init_args is None:
            mu, lamb = 1, 1  # defauts to TF + vW
        else:
            mu, lamb = init_args
        # make µ and λ trainable parameters
        self.mu = torch.nn.Parameter(torch.tensor([mu], dtype=torch.double, device=self.device))
        self.lamb = torch.nn.Parameter(torch.tensor([lamb], dtype=torch.double, device=self.device))
        self.initialize()

    def forward(self, box_vecs, den):
        return self.mu * Weizsaecker(box_vecs, den) + self.lamb * ThomasFermi(box_vecs, den)


# train the µ TF + λ vW functional response to fit the Lindhard response

shape = (61, 61, 61)
box_vecs = 8 * torch.eye(3, dtype=torch.double)
den = torch.ones(shape, dtype=torch.double)

# compute Lindhard response function
eta, G_inv_lind = G_inv_lindhard(box_vecs, den)

# initialize µ TF + λ vW functional
TFvW_train = TFvW()
# set its parameters to have requires_grad = True to make them trainable
TFvW_train.param_grad(True)

print('Initial (µ, λ) = ({:.5g}, {:.5g})\n'.format(TFvW_train.mu.item(), TFvW_train.lamb.item()))

for i in range(20):
    eta, G_inv = get_inv_G(box_vecs, den, TFvW_train.forward, requires_grad=True)
    # computes the loss function and performs optimization to minimize it
    loss = TFvW_train.grid_error(G_inv_lind, G_inv)  # these are inherited helper functions from the
    TFvW_train.update_params(loss)                           # KineticFunctional class to facilitate training
    print('Epoch = {}, Loss = {:.5g}'.format(i, loss.item()))

TFvW_train.param_grad(False)

print('\nOptimized (µ, λ) = ({:.5g}, {:.5g})'.format(TFvW_train.mu.item(), TFvW_train.lamb.item()))

eta, G_inv_opt = get_inv_G(box_vecs, den, TFvW_train.forward)

# make plot to compare against Lindhard response
plt.rc('font', family='serif')
plt.subplots(figsize=(5, 4))

plt.plot(eta[0, 0, :], G_inv_lind[0, 0, :], '-k')
plt.plot(eta[0, 0, :], G_inv_opt[0, 0, :], '--r')

plt.xlim([0, eta[0, 0, -1]])
plt.ylim([0, 1.05])

plt.xlabel(r'$\eta$', fontsize=12)
plt.ylabel(r'$G^{-1}(\eta)$', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

labels = ['Lindhard', r'Optimized $\mu$TF + $\lambda$vW']
plt.legend(labels=labels, loc="upper right", borderaxespad=0.4, ncol=1, prop={'size': 12})

plt.tight_layout()
plt.show()
