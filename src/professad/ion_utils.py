import numpy as np
import torch
from matscipy.neighbours import neighbour_list as matscipy_nl
from professad.functional_tools import wavevecs, interpolate

# ----------------------------------------------------------------------------
# This script contains auxiliary functions for computing the ion related
# terms in a periodic system.
# ----------------------------------------------------------------------------

bohr = 0.529177208607388
hartree_to_ev = 27.2113834279111
pot_conv_factor = 1 / (bohr * bohr * bohr * hartree_to_ev)

# -----------------------------------------------------------------------------------------------------------------------
#                                       Utility Functions to Read Recpot Files
# -----------------------------------------------------------------------------------------------------------------------


def get_ion_charge(path):
    """
    Reads recpot file to extract ion charge.

    Args:
      path (string) : Path to recpot file from current directory

    Returns:
      int: Ion charge
    """
    total_lines = 0
    with open(path, "r") as f:
        for line in f:
            total_lines += 1
    comment_lines = 0
    with open(path, "r") as f:
        for line in f:
            comment_lines += 1
            if "END COMMENT" in line:
                break
        f.readline()  # read the '3     5' line
        k_max = float(f.readline()) * bohr
        pot012 = np.asarray(f.readline().split(), dtype=np.float64) * pot_conv_factor
    num_ks = 3 * (total_lines - comment_lines - 3)
    dk = k_max / (num_ks - 1)
    z = round((pot012[1] - pot012[0]) * dk * dk / (-4 * np.pi))
    return z


def interpolate_recpot(path, ks_interp):
    """
    Reads recpot file to get interpolated ion potential.

    Args:
      path (string)            : Path to recpot file from current directory
      ks_interp (torch.Tensor) : Reciprocal space grid for which the ionic potential is
                                 interpolated over.

    Returns:
      torch.Tensor: Reciprocal space ionic potential
    """
    pot_ft = []
    with open(path, "r") as f:
        for line in f:
            if "END COMMENT" in line:
                break
        f.readline()  # read the '3     5' line
        k_max = float(f.readline()) * bohr
        for line in f:
            if len(line.split()) == 3:
                pot_ft += line.split()
    pot_ft = np.asarray(pot_ft, dtype=np.float64, order="C") * pot_conv_factor
    ks, dk = np.linspace(0, k_max, pot_ft.size, retstep=True)
    z = round((pot_ft[1] - pot_ft[0]) * dk * dk / (-4 * np.pi))
    pot_ft[1:] += 4 * np.pi * z / (ks[1:] * ks[1:])
    ks_torch = torch.as_tensor(ks, dtype=torch.double, device=ks_interp.device)
    pot_ft_torch = torch.as_tensor(pot_ft, dtype=torch.double, device=ks_interp.device)
    pot_ft_interp = interpolate(ks_torch, pot_ft_torch, torch.minimum(ks_interp, ks_torch[-1]))
    pot_ft_aux = torch.empty(ks_interp.shape, dtype=torch.double, device=ks_interp.device)
    pot_ft_aux[ks_interp == 0] = pot_ft_interp[ks_interp == 0]
    pot_ft_aux[ks_interp != 0] = pot_ft_interp[ks_interp != 0] - 4 * np.pi * z / ks_interp[ks_interp != 0].pow(2)
    return pot_ft_aux


# -----------------------------------------------------------------------------------------------------------------------
#                                     Utility Functions for Lattice Convolution
# -----------------------------------------------------------------------------------------------------------------------


def lattice_sum(box_vecs, shape, cart_ion_coords, f_tilde, order=None):
    r"""
    Performs convolution of lattice with function :math:`f` using FFTs and the convolution theorem.
    Effectively, :math:`F(\mathbf{r})` is computed by taking the inverse FFT of

    .. math:: \tilde{F}(\mathbf{q}) = \frac{1}{\Omega} S(\mathbf{q}) \tilde{f}(\mathbf{q}),

    where :math:`\tilde{f}` is the Fourier transform of :math:`f` and :math:`S(\mathbf{q})`
    is the structure factor given by

    .. math:: S(\mathbf{q}) = \sum_{i=1}^N e^{-i\mathbf{q}\cdot\mathbf{r}}.

    Args:
      box_vecs (torch.Tensor)        : Lattice vectors
      shape (torch.Size or iterable) : Real-space grid shape
      cart_ion_coords (Torch.tensor) : Cartesian ionic coordinates
      f_tilde (torch.Tensor)         : Fourier transform of function to be convolved
      order (None or int)            : Order of approximation for structure factor via the
                                       Particle-Mesh Ewald scheme, which must be an even integer
                                       more than or equal to 2. ``None`` indicates that the exact
                                       quadratic scaling method is used.

    Returns:
      torch.Tensor: Convolution of the lattice with function :math:`f`
    """
    if order is None:
        S = structure_factor(box_vecs, shape, cart_ion_coords)
    else:
        assert (order % 2 == 0) & (order >= 2), "Requires even order n ≥ 2"
        S = structure_factor_spline(box_vecs, shape, cart_ion_coords, order)
    return torch.fft.irfftn(S * f_tilde, shape, norm="forward") / torch.abs(torch.linalg.det(box_vecs))


def structure_factor(box_vecs, shape, cart_ion_coords, weights=None):
    r"""
    Computes, using the direct :math:`\mathcal{O}(N^2)` method, the structure factor given by

    .. math:: S(\mathbf{q}) = \sum_{i=1}^N w_i\, e^{-i\mathbf{q}\cdot\mathbf{r}_i},

    where :math:`w_i` are optional per-ion weights (default 1).

    Args:
      box_vecs (torch.Tensor)        : Lattice vectors
      shape (torch.Size or iterable) : Real-space grid shape
      cart_ion_coords (torch.Tensor) : Cartesian ionic coordinates
      weights (torch.Tensor or None) : Per-ion weights of shape ``(N,)``. Default ``None`` uses
                                       unit weights.

    Returns:
      torch.Tensor: Exact structure factor
    """
    kx, ky, kz, k2 = wavevecs(box_vecs, shape)
    kr = torch.einsum("xyza, ia -> xyzi", torch.stack([kx, ky, kz], dim=-1), cart_ion_coords)
    phases = torch.exp(-1j * kr)
    if weights is not None:
        phases = weights.reshape(1, 1, 1, -1) * phases
    return torch.sum(phases, axis=3)


def cardinal_b_spline_values(x, order):
    r"""
    This is a helper function for Particle-Mesh Ewald scheme. For :math:`x \in [0,1)` and
    order :math:`n \geq 2`, this function returns

    [:math:`M_n(x+i)` for :math:`i=0,1,\ldots,n-1`]

    The basic formula is

    .. math::  M_n[i] = \frac{x+i}{n-1} M_{n-1}[i] + \frac{n-x-i}{n-1} M_{n-1}[i-1].

    Note that while a much simpler implementation, ::

      def cardinal_b_spline_values(x, order):
        M = torch.zeros((order,) + x.shape, dtype = torch.double, device = x.device)
        M[0] = x
        M[1] = 1 - x

        for n in range(3, order+1):
          for i in np.flip(np.arange(1, n)):
            M[i] = ((x+i)*M[i] + (n-x-i)*M[i-1]) / (n-1)
          M[0] = x/(n-1) * M[0]
        return M

    is possible, we use a more complicated implementation to avoid in-place operations for this
    function to be auto-differentiable (necessary for auto-differentiated forces and stresses) .

    Args:
      x (torch.Tensor) : :math:`x \in [0,1)`
      order (int)      : Order :math:`n \geq 2`

    Returns:
      torch.Tensor: Cardinal b-spline values
    """
    assert torch.all(x >= 0.0) and torch.all(x < 1.0), "Requires 0 ≤ x < 1"
    assert order >= 2, "Requires order n ≥ 2"

    M1 = [
        [torch.zeros(x.shape, dtype=torch.double, device=x.device) for _ in range(order)] for _ in range(int(order / 2))
    ]
    M2 = [
        [torch.zeros(x.shape, dtype=torch.double, device=x.device) for _ in range(order)]
        for _ in range(int((order - 1) / 2))
    ]

    M1[0][0][:] = x  # M2(x)   = x
    M1[0][1][:] = 1 - x  # M2(x+1) = 2-(x+1)

    for n in range(3, order + 1):
        j = int(n / 2) - 1
        for i in np.flip(np.arange(1, n)):
            if n % 2 == 1:
                M2[j][i][:] = ((x + i) * M1[j][i][:] + (n - x - i) * M1[j][i - 1][:]) / (n - 1)
            else:
                M1[j][i][:] = ((x + i) * M2[j - 1][i][:] + (n - x - i) * M2[j - 1][i - 1][:]) / (n - 1)
        if n % 2 == 1:
            M2[j][0][:] = x / (n - 1) * M1[j][0][:]
        else:
            M1[j][0][:] = x / (n - 1) * M2[j - 1][0][:]

    M_return = torch.zeros((order,) + x.shape, dtype=torch.double, device=x.device)

    for i in range(order):
        if order % 2 == 0:
            M_return[i, :] = M1[-1][i][:]
        else:
            M_return[i, :] = M2[-1][i][:]
    return M_return


def exponential_spline_b(m, N, order):
    """
    This is a helper function for Particle-Mesh Ewald scheme.
    """
    zero = torch.zeros(m.shape, dtype=torch.double, device=m.device)
    M = cardinal_b_spline_values(zero, order)
    i = torch.arange(0, order, dtype=torch.double, device=m.device).unsqueeze(1).expand((-1,) + m.shape)
    b = torch.sum(M * torch.exp(1j * 2 * np.pi * m * (i - 1) / N), axis=0)
    return torch.exp(1j * 2 * np.pi * m * (order - 1) / N) / b


def structure_factor_spline(box_vecs, shape, cart_ion_coords, order, weights=None):
    r"""
    Computes an approximate structure factor using the Particle-Mesh Ewald scheme based on cardinal B-splines,
    which is an :math:`\mathcal{O}(N\log N)` scaling method. For comprehensive details, see the following
    references.

    * `Essmann et al., J. Chem. Phys. 103, 8577 (1995) <https://doi.org/10.1063/1.470117>`_

    * `Choly and Kaxiras, Phys. Rev. B 67, 155101 (2003) <https://doi.org/10.1103/PhysRevB.67.155101>`_

    * `Hung and Carter, Chem. Phys. Lett. 475, 163 (2009) <https://doi.org/10.1016/j.cplett.2009.04.059>`_

    Args:
      box_vecs (torch.Tensor)        : Lattice vectors
      shape (torch.Size or iterable) : Real-space grid shape
      cart_ion_coords (torch.Tensor) : Cartesian ionic coordinates
      order (int)                    : Order of approximation for structure factor
      weights (torch.Tensor or None) : Per-ion weights of shape ``(N,)``. Default ``None`` uses
                                       unit weights.

    Returns:
      torch.Tensor: Approximate structure factor
    """
    N0, N1, N2 = shape
    frac_ion_coords = torch.matmul(cart_ion_coords, torch.linalg.inv(box_vecs))
    # make sure fractional coordinates lie in [0,1) to prevent tensor size incompatibility
    # operation done twice because of special case, e.g. if original frac_ion_coords has
    # element -1e-16, first operation makes it 1.0, second operation makes it 0.0
    frac_ion_coords = frac_ion_coords - torch.floor(frac_ion_coords)
    frac_ion_coords = frac_ion_coords - torch.floor(frac_ion_coords)
    assert torch.all(frac_ion_coords >= 0) and torch.all(frac_ion_coords < 1), (
        "Fractional ionic coordinates don't all lie in [0,1)"
    )

    # getting Q(l1, l2, l3)
    Q = torch.zeros(shape, dtype=torch.double, device=box_vecs.device)

    u0 = frac_ion_coords[:, 0] * N0
    u1 = frac_ion_coords[:, 1] * N1
    u2 = frac_ion_coords[:, 2] * N2

    floor0 = torch.floor(u0).to(torch.int64)
    floor1 = torch.floor(u1).to(torch.int64)
    floor2 = torch.floor(u2).to(torch.int64)

    M0 = cardinal_b_spline_values(u0 - floor0, order)
    M1 = cardinal_b_spline_values(u1 - floor1, order)
    M2 = cardinal_b_spline_values(u2 - floor2, order)

    orders = (
        torch.arange(0, order, dtype=torch.int64, device=box_vecs.device)
        .unsqueeze(1)
        .expand((-1, frac_ion_coords.shape[0]))
    )
    l0 = torch.fmod(orders - floor0, N0) + (orders < floor0) * N0
    l1 = torch.fmod(orders - floor1, N1) + (orders < floor1) * N1
    l2 = torch.fmod(orders - floor2, N2) + (orders < floor2) * N2

    for i in range(frac_ion_coords.shape[0]):
        M0i, M1i, M2i = torch.meshgrid(M0[:, i], M1[:, i], M2[:, i], indexing="ij")
        l0i, l1i, l2i = torch.meshgrid(l0[:, i], l1[:, i], l2[:, i], indexing="ij")
        w_i = weights[i] if weights is not None else 1
        Q[l0i, l1i, l2i] += w_i * M0i * M1i * M2i

    Q_ft = torch.fft.rfftn(Q)

    # getting B(m1, m2, m3) = b(m1) b(m2) b(m3)
    n0 = torch.arange(0, Q_ft.shape[0], dtype=torch.double, device=box_vecs.device)
    b0 = exponential_spline_b(n0, N0, order)
    n1 = torch.arange(0, Q_ft.shape[1], dtype=torch.double, device=box_vecs.device)
    b1 = exponential_spline_b(n1, N1, order)
    n2 = torch.arange(0, Q_ft.shape[2], dtype=torch.double, device=box_vecs.device)
    b2 = exponential_spline_b(n2, N2, order)
    b0, b1, b2 = torch.meshgrid(b0, b1, b2, indexing="ij")

    return torch.conj(b0 * b1 * b2 * Q_ft)


# -----------------------------------------------------------------------------------------------------------------------
#                                     Ion-ion Interaction Classes
# -----------------------------------------------------------------------------------------------------------------------


class IonIon(torch.nn.Module):
    r"""
    Base class for ion-ion interaction energy terms. Subclasses must implement
    :meth:`forward`, which takes ``(box_vecs, coords, charges)`` and returns a scalar energy.
    """

    def forward(self, box_vecs, coords, charges):
        raise NotImplementedError


def _ewald_real_space(box_vecs, coords, charges, alpha, Rc):
    r"""
    Computes the real-space part of the Ewald sum:

    .. math:: E_\text{real} = \frac{1}{2} \sum_{i \neq j} q_i q_j
              \frac{\text{erfc}(\alpha r_{ij})}{r_{ij}}.
    """
    i_idx, j_idx, shifts_idx = matscipy_nl(
        "ijS",
        pbc=(True, True, True),
        cell=box_vecs.detach().cpu().numpy(),
        positions=coords.detach().cpu().numpy(),
        cutoff=float(Rc),
    )
    i_idx = torch.as_tensor(i_idx, dtype=torch.long, device=coords.device)
    j_idx = torch.as_tensor(j_idx, dtype=torch.long, device=coords.device)
    shifts_idx = torch.as_tensor(shifts_idx, dtype=box_vecs.dtype, device=coords.device)
    Zi = torch.index_select(charges, 0, i_idx)
    Zj = torch.index_select(charges, 0, j_idx)
    r_ij = (
        torch.index_select(coords, 0, j_idx) + torch.matmul(shifts_idx, box_vecs) - torch.index_select(coords, 0, i_idx)
    ).norm(p=2, dim=1)
    return 0.5 * torch.sum(Zi * Zj * torch.erfc(alpha * r_ij) / r_ij)


class EwaldSum(IonIon):
    r"""
    Ion-ion interaction via Ewald summation. When ``pme_order`` is ``None``, the reciprocal sum
    uses explicit :math:`\mathcal{O}(N^2)` G-vector enumeration (requires ``k_cutoff``). When
    ``pme_order`` is given, the Particle-Mesh Ewald scheme is used for
    :math:`\mathcal{O}(N \log N)` scaling (requires ``shape``).

    Args:
      alpha (float)            : Ewald splitting parameter
      Rc (float)               : Real-space cutoff
      k_cutoff (float or None) : Reciprocal-space cutoff (required for vanilla Ewald)
      shape (tuple or None)    : FFT grid shape (required for PME)
      pme_order (int or None)  : PME spline order (even int >= 2), or ``None`` for vanilla
    """

    def __init__(self, alpha, Rc, k_cutoff=None, shape=None, pme_order=None):
        super().__init__()
        self.alpha = alpha
        self.Rc = Rc
        self.k_cutoff = k_cutoff
        self.shape = shape
        self.pme_order = pme_order

    def forward(self, box_vecs, coords, charges):
        alpha, Rc = self.alpha, self.Rc
        vol = torch.abs(torch.linalg.det(box_vecs))

        # real-space sum
        E_real = _ewald_real_space(box_vecs, coords, charges, alpha, Rc)

        # reciprocal-space sum
        if self.pme_order is not None:
            assert self.shape is not None, "shape is required for PME Ewald"
            assert (self.pme_order % 2 == 0) & (self.pme_order >= 2), "requires even pme_order >= 2"
            S_q = structure_factor_spline(box_vecs, self.shape, coords, self.pme_order, weights=charges)
            kx, ky, kz, k2 = wavevecs(box_vecs, self.shape)
            kernel = torch.zeros(k2.shape, dtype=torch.double, device=coords.device)
            nonzero = k2 > 0
            kernel[nonzero] = torch.exp(-k2[nonzero] / (4 * alpha * alpha)) / k2[nonzero]
            S_q_abs2 = S_q.real.pow(2) + S_q.imag.pow(2)
            # rfft stores half the last dimension; interior columns represent two conjugate pairs
            weight = 2 * torch.ones(k2.shape, dtype=torch.double, device=coords.device)
            weight[:, :, 0] = 1
            if self.shape[2] % 2 == 0:
                weight[:, :, -1] = 1
            E_recip = 2 * np.pi / vol * torch.sum(weight * S_q_abs2 * kernel)
        else:
            assert self.k_cutoff is not None, "k_cutoff is required for vanilla Ewald"
            b = 2 * np.pi * torch.linalg.inv(box_vecs.T)
            b_lengths = torch.sqrt(torch.sum(b.detach().pow(2), dim=1))
            n_max = torch.ceil(self.k_cutoff / b_lengths).int()
            n0 = torch.arange(-n_max[0], n_max[0] + 1, device=coords.device)
            n1 = torch.arange(-n_max[1], n_max[1] + 1, device=coords.device)
            n2 = torch.arange(-n_max[2], n_max[2] + 1, device=coords.device)
            nA, nB, nC = torch.meshgrid(n0, n1, n2, indexing="ij")
            nA, nB, nC = nA.reshape(-1), nB.reshape(-1), nC.reshape(-1)
            G = nA.unsqueeze(1).double() * b[0] + nB.unsqueeze(1).double() * b[1] + nC.unsqueeze(1).double() * b[2]
            G2 = torch.sum(G.pow(2), dim=1)
            G_norm = torch.sqrt(G2)
            mask = (G_norm > 0) & (G_norm <= self.k_cutoff)
            G, G2 = G[mask], G2[mask]
            G_dot_r = torch.matmul(G, coords.T)
            cos_phase = torch.cos(G_dot_r)
            sin_phase = torch.sin(G_dot_r)
            S_real = torch.matmul(cos_phase, charges)
            S_imag = -torch.matmul(sin_phase, charges)
            S_q_abs2 = S_real.pow(2) + S_imag.pow(2)
            kernel = torch.exp(-G2 / (4 * alpha * alpha)) / G2
            E_recip = 2 * np.pi / vol * torch.sum(S_q_abs2 * kernel)

        E_self = -alpha / np.sqrt(np.pi) * torch.sum(charges.pow(2))
        E_bg = -np.pi / (2 * vol * alpha * alpha) * torch.sum(charges).pow(2)
        return E_real + E_recip + E_self + E_bg


class PickardSum(IonIon):
    r"""
    Ion-ion interaction via the Pickard real-space pairwise electrostatic sum.

    Based on `Phys. Rev. Materials 2, 013806 <https://doi.org/10.1103/PhysRevMaterials.2.013806>`_.

    Args:
      Rc (float) : Cutoff radius
      Rd (float) : Damping parameter
    """

    def __init__(self, Rc=None, Rd=None):
        super().__init__()
        self.Rc = Rc
        self.Rd = Rd

    def forward(self, box_vecs, coords, charges):
        Rc, Rd = self.Rc, self.Rd
        if Rc is None or Rd is None:
            interplanar_dist = 1 / torch.sqrt(torch.sum(torch.linalg.inv(box_vecs.detach().T).pow(2), 1))
            h_max = torch.max(interplanar_dist)
            if Rc is None:
                Rd = 2 * h_max
                Rc = 3 * Rd * Rd / h_max
            else:
                Rd = torch.sqrt(h_max * Rc / 3)
        # neighbor list
        i_idx, j_idx, shifts_idx = matscipy_nl(
            "ijS",
            pbc=(True, True, True),
            cell=box_vecs.detach().cpu().numpy(),
            positions=coords.detach().cpu().numpy(),
            cutoff=float(Rc),
        )
        i_idx = torch.as_tensor(i_idx, dtype=torch.long, device=coords.device)
        j_idx = torch.as_tensor(j_idx, dtype=torch.long, device=coords.device)
        shifts_idx = torch.as_tensor(shifts_idx, dtype=box_vecs.dtype, device=coords.device)
        # charge terms
        rho = torch.sum(charges) / torch.abs(torch.linalg.det(box_vecs))
        Zi = torch.index_select(charges, 0, i_idx)
        Zj = torch.index_select(charges, 0, j_idx)
        Qi = torch.scatter_add(charges, 0, i_idx, Zj)
        aux = (0.75 / np.pi) * Qi / rho
        Ra = aux.sign() * aux.abs().pow(1 / 3)  # a.pow(n<1) gives nans for a<0
        # pairwise distances
        r_ij = (
            torch.index_select(coords, 0, j_idx)
            + torch.matmul(shifts_idx, box_vecs)
            - torch.index_select(coords, 0, i_idx)
        ).norm(p=2, dim=1)
        # energy terms
        E_local = torch.sum(0.5 * Zi * Zj * torch.erfc(r_ij.div(Rd)).div(r_ij))
        E_corr = torch.sum(
            -np.pi * charges * rho * Ra.square()
            + np.pi * charges * rho * (Ra.square() - 0.5 * Rd * Rd) * torch.erf(Ra.div(Rd))
            + np.sqrt(np.pi) * charges * rho * Ra * Rd * torch.exp(-Ra.square().div(Rd * Rd))
            - charges.square() / np.sqrt(np.pi) / Rd
        )
        return E_local + E_corr
