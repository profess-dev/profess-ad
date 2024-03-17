import numpy as np
from math import pi, sqrt
import torch
from professad.functional_tools import wavevectors, interpolate
from torch_nl import compute_neighborlist

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
    with open(path, 'r') as f:
        for line in f:
            total_lines += 1
    comment_lines = 0
    with open(path, 'r') as f:
        for line in f:
            comment_lines += 1
            if 'END COMMENT' in line:
                break
        f.readline()  # read the '3     5' line
        k_max = float(f.readline()) * bohr
        pot012 = np.asarray(f.readline().split(), dtype=np.float64) * pot_conv_factor
    num_ks = 3 * (total_lines - comment_lines - 3)
    dk = k_max / (num_ks - 1)
    z = round((pot012[1] - pot012[0]) * dk * dk / (-4 * pi))
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
    with open(path, 'r') as f:
        for line in f:
            if 'END COMMENT' in line:
                break
        f.readline()  # read the '3     5' line
        k_max = float(f.readline()) * bohr
        for line in f:
            if len(line.split()) == 3:
                pot_ft += line.split()
    pot_ft = np.asarray(pot_ft, dtype=np.float64, order='C') * pot_conv_factor
    ks, dk = np.linspace(0, k_max, pot_ft.size, retstep=True)
    z = round((pot_ft[1] - pot_ft[0]) * dk * dk / (-4 * pi))
    pot_ft[1:] += 4 * pi * z / (ks[1:] * ks[1:])
    ks_torch = torch.as_tensor(ks, dtype=ks_interp.dtype, device=ks_interp.device)
    pot_ft_torch = torch.as_tensor(pot_ft, dtype=ks_interp.dtype, device=ks_interp.device)
    pot_ft_interp = interpolate(ks_torch, pot_ft_torch, torch.minimum(ks_interp, ks_torch[-1]))
    pot_ft_aux = torch.empty(ks_interp.shape, dtype=ks_interp.dtype, device=ks_interp.device)
    pot_ft_aux[ks_interp == 0] = pot_ft_interp[ks_interp == 0]
    pot_ft_aux[ks_interp != 0] = pot_ft_interp[ks_interp != 0] - 4 * pi * z / ks_interp[ks_interp != 0].square()
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
        assert (order % 2 == 0) & (order >= 2), 'Requires even order n ≥ 2'
        S = structure_factor_spline(box_vecs, shape, cart_ion_coords, order)
    return torch.fft.irfftn(S * f_tilde, shape, norm='forward') / torch.abs(torch.linalg.det(box_vecs))


def structure_factor(box_vecs, shape, cart_ion_coords):
    r"""
    Computes, using the direct :math:`\mathcal{O}(N^2)` method, the structure factor given by

    .. math:: S(\mathbf{q}) = \sum_{i=1}^N e^{-i\mathbf{q}\cdot\mathbf{r}}.

    Args:
      box_vecs (torch.Tensor)        : Lattice vectors
      shape (torch.Size or iterable) : Real-space grid shape
      cart_ion_coords (torch.Tensor) : Cartesian ionic coordinates

    Returns:
      torch.Tensor: Exact structure factor
    """
    return torch.sum(torch.exp(-1j * torch.einsum('xyza, ia -> xyzi',
                                                   wavevectors(box_vecs, shape),
                                                   cart_ion_coords)), axis=3)


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
    assert torch.all(x >= 0.0) and torch.all(x < 1.0), 'Requires 0 ≤ x < 1'
    assert order >= 2, 'Requires order n ≥ 2'

    M1 = [[torch.zeros(x.shape, dtype=x.dtype, device=x.device) for _ in range(order)]
          for _ in range(int(order / 2))]
    M2 = [[torch.zeros(x.shape, dtype=x.dtype, device=x.device) for _ in range(order)]
          for _ in range(int((order - 1) / 2))]

    M1[0][0][:] = x        # M2(x)   = x
    M1[0][1][:] = 1 - x    # M2(x+1) = 2-(x+1)

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

    M_return = torch.zeros((order,) + x.shape, dtype=x.dtype, device=x.device)

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
    b = torch.sum(M * torch.exp(1j * 2 * pi * m * (i - 1) / N), axis=0)
    return torch.exp(1j * 2 * pi * m * (order - 1) / N) / b


def structure_factor_spline(box_vecs, shape, cart_ion_coords, order):
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
    assert torch.all(frac_ion_coords >= 0) and torch.all(frac_ion_coords < 1), \
           'Fractional ionic coordinates don\'t all lie in [0,1)'

    # getting Q(l1, l2, l3)
    Q = torch.zeros(shape, dtype=box_vecs.dtype, device=box_vecs.device)

    u0 = frac_ion_coords[:, 0] * N0
    u1 = frac_ion_coords[:, 1] * N1
    u2 = frac_ion_coords[:, 2] * N2

    floor0 = torch.floor(u0).to(torch.long)
    floor1 = torch.floor(u1).to(torch.long)
    floor2 = torch.floor(u2).to(torch.long)

    M0 = cardinal_b_spline_values(u0 - floor0, order)
    M1 = cardinal_b_spline_values(u1 - floor1, order)
    M2 = cardinal_b_spline_values(u2 - floor2, order)

    orders = torch.arange(0, order, dtype=torch.long, device=box_vecs.device) \
             .unsqueeze(1).expand((-1, frac_ion_coords.shape[0]))
    l0 = (torch.fmod(orders - floor0, N0) + (orders < floor0) * N0)
    l1 = (torch.fmod(orders - floor1, N1) + (orders < floor1) * N1)
    l2 = (torch.fmod(orders - floor2, N2) + (orders < floor2) * N2)

    for i in range(frac_ion_coords.shape[0]):
        M0i, M1i, M2i = torch.meshgrid(M0[:, i], M1[:, i], M2[:, i], indexing='ij')
        l0i, l1i, l2i = torch.meshgrid(l0[:, i], l1[:, i], l2[:, i], indexing='ij')
        Q[l0i, l1i, l2i] += M0i * M1i * M2i

    Q_ft = torch.fft.rfftn(Q)

    # getting B(m1, m2, m3) = b(m1) b(m2) b(m3)
    n0 = torch.arange(0, Q_ft.shape[0], dtype=box_vecs.dtype, device=box_vecs.device)
    b0 = exponential_spline_b(n0, N0, order)
    n1 = torch.arange(0, Q_ft.shape[1], dtype=box_vecs.dtype, device=box_vecs.device)
    b1 = exponential_spline_b(n1, N1, order)
    n2 = torch.arange(0, Q_ft.shape[2], dtype=box_vecs.dtype, device=box_vecs.device)
    b2 = exponential_spline_b(n2, N2, order)
    b0, b1, b2 = torch.meshgrid(b0, b1, b2, indexing='ij')

    return torch.conj(b0 * b1 * b2 * Q_ft)

# -----------------------------------------------------------------------------------------------------------------------
#                                  Utility Functions for Ion-ion Interaction Term
# -----------------------------------------------------------------------------------------------------------------------


def ion_interaction_sum(box_vecs, coords, charges, Rc, Rd, neighborlist=None):
    r"""
    Computes the ion-ion interaction energy using a real-space pairwise electrostatic summation
    in a uniform neutralizing background. Key parameters are :math:`R_c`, the cut-off radius, and
    :math:`R_d`, a damping parameter. It was recommended that :math:`R_d = 2 h_\text{max}` and
    :math:`R_c = 3 R_d^2 / h_\text{max}`, where :math:`h_\text{max}` is the maximum interplanar
    distance.

    More details can be found in `Phys. Rev. Materials 2, 013806 <https://doi.org/10.1103/PhysRevMaterials.2.013806>`_.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      coords (torch.Tensor)   : Cartesian ionic coordinates
      charges (torch.Tensor)  : Charges of all ions in the simulation cell
      Rc, Rd  (float)         : Parameters for the electrtrostatic sum (see paper)
      neighborlist (list)     : Optional neighborlist

    Returns:
      torch.Tensor: Ion-ion interaction energy
    """
    # get neighbor list
    if neighborlist is None:
        mapping, batch_mapping, shifts_idx = compute_neighborlist(
            Rc,
            coords.detach().clone(),
            box_vecs.detach().clone(),
            torch.ones((3,), device=coords.device).bool(),
            torch.zeros((coords.shape[0], ), dtype=torch.long, device=coords.device),
            self_interaction=False
        )
    else:
        mapping, shifts_idx = neighborlist

    # charge terms
    rho = torch.sum(charges) / torch.abs(torch.linalg.det(box_vecs))
    Zi = torch.index_select(charges, 0, mapping[0])  # (PQ,)
    Zj = torch.index_select(charges, 0, mapping[1])  # (PQ,)
    Qi = torch.scatter_add(charges, 0, mapping[0], Zj)  # (P,)
    aux = (0.75 / pi) * Qi / rho  # (P,)
    Ra = aux.sign() * aux.abs().pow(1 / 3)  # (P,) | a.pow(n<1) gives nans for a<0
    Ra2 = Ra.square()
    # pairwise distances
    r_ij = (torch.index_select(coords, 0, mapping[1]) + torch.matmul(shifts_idx, box_vecs)
            - torch.index_select(coords, 0, mapping[0])).norm(p=2, dim=1)  # (PQ,)
    # energy terms
    E_local = torch.sum(0.5 * Zi * Zj * torch.erfc(r_ij.div(Rd)).div(r_ij))  # sum over i,j
    E_corr = torch.sum(- pi * charges * rho * Ra2
                       + pi * charges * rho * (Ra2 - 0.5 * Rd * Rd) * torch.erf(Ra.div(Rd))
                       + sqrt(pi) * charges * rho * Ra * Rd * torch.exp(- Ra2.div(Rd * Rd))
                       - charges.square() / sqrt(pi) / Rd)  # sum over i
    return E_local + E_corr
