import matplotlib.pyplot as plt
import torch

from professad.functionals import G_inv_lindhard, WangTeter, LuoKarasievTrickey, \
    PauliGaussian, Weizsaecker, ThomasFermi
from professad.functional_tools import get_inv_G

shape = (61, 61, 61)
box_vecs = 8 * torch.eye(3, dtype=torch.double)
den = torch.ones(shape, dtype=torch.double)

plt.rc('font', family='serif')
plt.subplots(figsize=(5, 4))

eta, lind = G_inv_lindhard(box_vecs, den)
plt.plot(eta[0, 0, :], lind[0, 0, :], '-k')

eta, F = get_inv_G(box_vecs, den, WangTeter)
plt.plot(eta[0, 0, :], F[0, 0, :], 'r', ls=(0, (5, 5)))

eta, F = get_inv_G(box_vecs, den, LuoKarasievTrickey)
plt.plot(eta[0, 0, :], F[0, 0, :], 'b', ls=(0, (1, 1)))

pg = PauliGaussian()
pg.set_PGSL025()
eta, F = get_inv_G(box_vecs, den, pg.forward)
plt.plot(eta[0, 0, :], F[0, 0, :], '--m')

eta, F = get_inv_G(box_vecs, den, lambda bv, den: 0.6 * Weizsaecker(bv, den) + ThomasFermi(bv, den))
plt.plot(eta[0, 0, :], F[0, 0, :], '-.g')

plt.xlim([0, eta[0, 0, -1]])
plt.ylim([0, 1.05])

plt.xlabel(r'$\eta$', fontsize=10)
plt.ylabel(r'$G^{-1}(\eta)$', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

labels = ['Lindhard', 'WT', 'LKT', 'PGSL0.25', 'TF(0.6)vW']
plt.legend(labels=labels, loc="upper right", borderaxespad=0.4, ncol=1, prop={'size': 12})

plt.tight_layout()
plt.show()
