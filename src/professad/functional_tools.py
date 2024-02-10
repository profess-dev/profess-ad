import numpy as np
import torch

# ---------------------------------------------------------------------------------------------------------------------
#                            Auto-differentiation Tools for Derivatives of Functionals
# ---------------------------------------------------------------------------------------------------------------------


def get_functional_derivative(box_vecs, den, functional, requires_grad=False):
    r""" Computes functional derivative

    This is a utility function that computes the functional derivative

    .. math:: \frac{\delta F}{\delta n(\mathbf{r})}

    of a given density functional :math:`F[n]` via autograd. The functional
    derivative computed can be used for further derivatives if ``requires_grad = True``.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density
      functional (function)   : Density functional that takes in arguments ``box_vecs`` and ``den``
      requires_grad (bool)    : Whether the fuctional derivative returned has ``requires_grad = True``

    Returns:
      torch.Tensor: Functional derivative
    """
    den.requires_grad = True
    functional_derivative = torch.autograd.grad(functional(box_vecs, den), den, create_graph=requires_grad)[0]
    den.requires_grad = False
    return functional_derivative / (torch.abs(torch.linalg.det(box_vecs)) / den.numel())


def get_inv_G(box_vecs, den, kinetic_functional, requires_grad=False):
    r""" Computes linear response function

    This is a utility function that computes the linear response function :math:`G^{-1}(\eta)` where

    .. math:: G^{-1}(\eta) = \frac{\pi^2}{k_F} \left( \hat{\mathcal{F}}  \left. \left\{
                                  \frac{\delta^2 T_\text{S}}{\delta n(\mathbf{r}) \delta n(\mathbf{r}')}
                                  \right\} \right|_{n_0,n_0} \right)^{-1}

    of a given density functional :math:`F[n]` via autograd. The response function computed can be
    used for further derivatives if ``requires_grad = True``. This utility function can
    be used for inspection purposes or even to fit the linear response of a kinetic functional to be
    closer to the Lindhard response, for example.

    Args:
      box_vecs (torch.Tensor)         : Lattice vectors
      den      (torch.Tensor)         : Electron density
      kinetic_functional (function)   : Kinetic functional that takes in arguments ``box_vecs`` and ``den``
      requires_grad (bool)            : Whether the response function returned has ``requires_grad = True``

    Returns:
      torch.Tensor:  Linear response function
    """
    vol = torch.abs(torch.linalg.det(box_vecs))
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    N_elec = round((torch.mean(den) * vol).detach().item())
    n0 = (N_elec / vol).repeat(den.shape)
    if not vol.requires_grad:
        n0.requires_grad_()
    k_F = (3 * np.pi * np.pi * N_elec / vol)**(1 / 3)
    T = kinetic_functional(box_vecs, n0)
    dTdn = torch.autograd.grad(T, n0, create_graph=True)[0] / (vol / torch.numel(den))
    G_inv = np.pi * np.pi / k_F / torch.fft.rfftn(torch.autograd.grad(dTdn[0, 0, 0], n0,
                                                  create_graph=requires_grad or vol.requires_grad)[0]).real
    eta = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
    eta[k2 != 0] = torch.sqrt(k2[k2 != 0]) / (2 * k_F)
    return eta, G_inv


def get_stress(box_vecs, den, functional, requires_grad=False):
    r""" Computes stress

    This is a utility function that computes the functional contribution to stress

    .. math:: \sigma_{ij} = \frac{1}{\Omega} \left.\frac{\partial F[n]}{\partial \epsilon_{ij}}
              \right|_{\epsilon_{ij} = 0}
              = \frac{1}{\Omega} \sum_k \frac{\partial F[n]}{\partial h_{ik}} h_{jk}

    of a given density functional :math:`F[n]` via autograd. :math:`h_{ij}` are elements
    of a matrix whose columns are lattice vectors and the cell volume is :math:`\Omega`.
    The stress computed can be used for further derivatives if ``requires_grad = True``.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density
      functional (function)   : Density functional that takes in arguments ``box_vecs`` and ``den``
      requires_grad (bool)    : Whether the response function returned has ``requires_grad = True``

    Returns:
      torch.Tensor: Stress tensor (3 by 3)
    """
    box_vecs.requires_grad = True
    vol = torch.abs(torch.linalg.det(box_vecs))
    grad_den = den * vol.detach() / vol
    dEdcell = torch.autograd.grad(functional(box_vecs, grad_den), box_vecs, create_graph=requires_grad)[0].T
    box_vecs.requires_grad = False
    stress = torch.matmul(dEdcell, box_vecs) / vol.detach()
    return stress


def get_pressure(box_vecs, den, functional, requires_grad=False):
    r""" Computes pressure

    This is a utility function that computes the functional contribution to pressure

    .. math:: P_F = - \frac{dF[n]}{d\Omega}

    of a given density functional :math:`F[n]` via autograd, where :math:`\Omega`
    is the volume of the cell. The pressure computed can be used for further derivatives
    if ``requires_grad = True``.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density
      functional (function)   : Density functional that takes in arguments ``box_vecs`` and ``den``
      requires_grad (bool)    : Whether the response function returned has ``requires_grad = True``

    Returns:
      torch.Tensor: Pressure
    """
    vol = torch.abs(torch.linalg.det(box_vecs))
    vol.requires_grad = True
    F = functional(box_vecs * (vol / vol.detach()).pow(1 / 3), den * vol.detach() / vol)
    return - torch.autograd.grad(F, vol, create_graph=requires_grad)[0]


# -----------------------------------------------------------------------------------------------------------------
#                                          Utility Functions for Functionals
# -----------------------------------------------------------------------------------------------------------------

# ----------------------------------------- Wavevectors from Lattice ----------------------------------------------
def wavevecs(box_vecs, shape):
    """ Generates wavevectors

    This is a utility function that generates the wavevectors for a given
    lattice such that the wavevectors are differentiable with respect to
    the lattice vectors.

    Args:
      box_vecs (torch.Tensor)           : Lattice vectors
      shape    (torch.Size or iterable) : Real-space grid shape

    Returns:
      torch.Tensor: Wavevectors consistent with real FFTs in the order :math:`k_x,~k_y,~k_z,~k^2`
    """
    b = 2 * np.pi * torch.linalg.inv(box_vecs.T)  # reciprocal lattice vectors
    assert not torch.any(torch.isnan(b)), 'Lattice vector matrix is not invertible.'
    # k-vector indices (enforcing nyquist > 0 for even lengths)
    j0, j1 = (torch.fft.fftfreq(shape[i], dtype=torch.double, device=box_vecs.device) * shape[i] for i in range(2))
    for f in [j0, j1]:
        f[int(len(f) / 2)] = torch.abs(f[int(len(f) / 2)])
    j2 = torch.fft.rfftfreq(shape[2], dtype=torch.double, device=box_vecs.device) * shape[2]
    nA, nB, nC = torch.meshgrid(j0, j1, j2, indexing='ij')
    # k-vector arrays
    kx = nA * b[0, 0] + nB * b[1, 0] + nC * b[2, 0]
    ky = nA * b[0, 1] + nB * b[1, 1] + nC * b[2, 1]
    kz = nA * b[0, 2] + nB * b[1, 2] + nC * b[2, 2]
    k2 = kx.pow(2) + ky.pow(2) + kz.pow(2)
    return kx, ky, kz, k2


# -------------------------------------------- FFT-based Derivatives ---------------------------------------------
def grad_i(ki, f):
    r""" Computes gradient component

    This is a utility function that computes the gradient component or
    partial spatial derivative

    .. math:: \nabla_i f =  \frac{\partial f}{\partial r_i}

    where :math:`f` is a given function and :math:`r_i \in \{x,y,z\}`.

    Args:
      ki (torch.Tensor)  : :math:`k_i \in \{k_x,k_y,k_z\}`
      f  (torch.Tensor)  : A scalar function

    Returns:
      torch.Tensor: Gradient component
    """
    return torch.fft.irfftn(1j * ki * torch.fft.rfftn(f), f.shape)


def grad_dot_grad(kx, ky, kz, f):
    r""" Computes squared gradient

    This is a utility function that computes the squared gradient or
    the dot product of a gradient with itself

    .. math:: |\nabla f|^2 =  \left(\frac{\partial f}{\partial x}\right)^2
                             + \left(\frac{\partial f}{\partial y}\right)^2
                             + \left(\frac{\partial f}{\partial z}\right)^2

    where :math:`f` is a given function.

    Args:
      ki (torch.Tensor)  : :math:`k_i \in \{k_x,k_y,k_z\}`
      f  (torch.Tensor)  : A scalar function

    Returns:
      torch.Tensor: Dot product of gradient with itself
    """
    grad_x, grad_y, grad_z = grad_i(kx, f), grad_i(ky, f), grad_i(kz, f)
    return grad_x * grad_x + grad_y * grad_y + grad_z * grad_z


def laplacian(k2, f):
    r""" Computes Laplacian

    This is a utility function that computes the Laplacian

    .. math:: \nabla^2 f =  \frac{\partial^2 f}{\partial^2 x}
                           + \frac{\partial^2 f}{\partial^2 y}
                           + \frac{\partial^2 f}{\partial^2 z}

    where :math:`f` is a given function.

    Args:
      k2 (torch.Tensor)  : :math:`k^2 = k_x^2 + k_y^2 + k_z^2`
      f  (torch.Tensor)  : A scalar function

    Returns:
      torch.Tensor: Laplacian
    """
    return torch.fft.irfftn(-k2 * torch.fft.rfftn(f), f.shape)


def reduced_gradient(kx, ky, kz, den):
    r""" Computes reduced gradient

    This is a utility function that computes the reduced gradient

    .. math:: s = \frac{|\nabla n|}{2(3\pi)^{1/3} n^{4/3}}

    where :math:`n` is the electron density.

    Args:
      ki (torch.Tensor)  : :math:`k_i \in \{k_x,k_y,k_z\}`
      den (torch.Tensor) : Electron density, :math:`n`

    Returns:
      torch.Tensor: Reduced gradient
    """
    gdg = grad_dot_grad(kx, ky, kz, den)
    abs_grad = torch.zeros(den.shape, dtype=torch.double, device=den.device)
    abs_grad[gdg != 0] = torch.sqrt(gdg[gdg != 0])
    return 0.5 * (3 * np.pi * np.pi)**(-1 / 3) * abs_grad / den.pow(4 / 3)


def reduced_gradient_squared(kx, ky, kz, den):
    r""" Computes squared reduced gradient

    This is a utility function that computes the reduced gradient

    .. math:: s^2 = \frac{|\nabla n|^2}{4(3\pi)^{2/3} n^{8/3}}

    where :math:`n` is the electron density.

    Args:
      ki (torch.Tensor)  : :math:`k_i \in \{k_x,k_y,k_z\}`
      den (torch.Tensor) : Electron density, :math:`n`

    Returns:
      torch.Tensor: Squared reduced gradient
    """
    return 0.25 * (3 * np.pi * np.pi)**(-2 / 3) * grad_dot_grad(kx, ky, kz, den) / den.pow(8 / 3)


def reduced_laplacian(k2, den):
    r""" Computes reduced Laplacian

    This is a utility function that computes the reduced Laplacian

    .. math:: q = \frac{\nabla^2 n}{4(3\pi)^{2/3} n^{5/3}}

    where :math:`n` is the electron density.

    Args:
      k2 (torch.Tensor)  : :math:`k^2 = k_x^2 + k_y^2 + k_z^2`
      den (torch.Tensor) : Electron density, :math:`n`

    Returns:
      torch.Tensor: Reduced Laplacian
    """
    return 0.25 * (3 * np.pi * np.pi)**(-2 / 3) * laplacian(k2, den) / den.pow(5 / 3)


# ------------------------------------------------ Interpolation Tools ------------------------------------------------

def interpolate(x, y, xs):
    r""" Performs interpolation

    This is a utility function that performs a Pytorch based cubic hermite
    spline interpolation method.
    The implementation is based on this
    `gist <https://gist.github.com/chausies/c453d561310317e7eda598e229aea537>`_
    , and extends it to ``xs`` with any dimensions :math:`N \geq 1`.

    Args:
      x, y (torch.Tensor)  : One-dimensional tensors such that y(x) is the function
                             to be interpolated
      xs (torch.Tensor)    : :math:`N`-dimensional tensor :math:`(N \geq 1)`
                             for which y(xs) is to be interpolated for
    Returns:
      torch.Tensor: Interpolated quantity
    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    t = ((xs - x[idxs]) / dx).unsqueeze(0).expand((4,) + xs.shape)

    a = torch.arange(4, device=xs.device)
    for _ in range(len(xs.shape)):
        a = a.unsqueeze(-1)
    a = a.expand((-1,) + xs.shape)

    tt = torch.ones(a.shape, dtype=torch.double, device=xs.device)
    tt[a != 0] = t[a != 0].pow(a[a != 0])

    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=xs.device)
    for _ in range(len(xs.shape)):
        A = A.unsqueeze(-1)
    A = A.expand((-1, -1) + xs.shape)

    hh = torch.sum(A * tt, axis=1)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx


def interpolate_kernel(xi_sparse, f, xis):
    r""" Performs kernel interpolation

    Consider a function :math:`f(x,y,z,\xi)` where the last argument :math:`\xi(x,y,z)` is
    also a scalar field. Let us have

    1. a list of discrete :math:`\xi` values, e.g. :math:`\xi_i \in \{\xi_1, \xi_2, \xi_3, \ldots\}`,
       represented by the argument ``xi_sparse``
    2. a set of :math:`f(x,y,z,\xi_i)` evaluated for specifc :math:`\xi_i` values,
       where :math:`\xi(x,y,z) = \xi_i` is constant over all space, represented by the argument ``f``
    3. the function :math:`\xi(x,y,z)` that could vary in space, represented by the argument ``xis``

    This function performs a cubic hermite interpolation to compute :math:`f(x,y,z,\xi)` for
    :math:`\xi(x,y,z)` using the set of :math:`f(x,y,z,\xi_i)` values.

    Args:
      xi_sparse (torch.Tensor) : Tensor of :math:`\xi` values with shape :math:`(n_\xi)`
      f (torch.Tensor)         : Tensor of :math:`f(x,y,z,\xi_i)` values with shape :math:`(n_1,n_2,n_3,n_\xi)`
      xis (torch.Tensor)       : Tensor of the spatially varying :math:`\xi(x,y,z)` with shape
                                 :math:`(n_1,n_2,n_3)`

    Returns:
      torch.Tensor: :math:`f(x,y,z,\xi)` with shape :math:`(n_1,n_2,n_3)`
    """
    xiss = xi_sparse.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(f.shape)
    m = (f[:, :, :, 1:] - f[:, :, :, :-1]) / (xiss[:, :, :, 1:] - xiss[:, :, :, :-1])
    m = torch.cat((m[:, :, :, 0].unsqueeze(3), (m[:, :, :, 1:] + m[:, :, :, :-1]) / 2, m[:, :, :, -1].unsqueeze(3)), 3)
    idxs = torch.searchsorted(xi_sparse[1:], xis)
    dx = (xi_sparse[idxs + 1] - xi_sparse[idxs])
    t = (xis - xi_sparse[idxs]) / dx
    tt = t**torch.arange(4, device=xis.device).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((-1,) + t.shape)
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=xis.device).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand((-1, -1) + t.shape)
    hh = torch.sum(A * tt, axis=1)
    return hh[0] * torch.gather(f, 3, idxs.unsqueeze(3))[:, :, :, 0] \
         + hh[1] * torch.gather(m, 3, idxs.unsqueeze(3))[:, :, :, 0] * dx \
         + hh[2] * torch.gather(f, 3, (idxs + 1).unsqueeze(3))[:, :, :, 0] \
         + hh[3] * torch.gather(m, 3, (idxs + 1).unsqueeze(3))[:, :, :, 0] * dx


def field_dependent_convolution(k, f_tilde, g, xis, kappa, mode='arithmetic'):
    r""" Computes a field-dependent convolution

    Computes a "field-dependent convolution", which is an integral quantity

    .. math:: K(\mathbf{r}) = \int d^3\mathbf{r}' f(|\mathbf{r}-\mathbf{r}'|, \xi(\mathbf{r}))~g(\mathbf{r}')

    Args:

      k (torch.Tensor)    : Wavevectors :math:`k=|\mathbf{k}|` with shape :math:`(m_1,m_2,m_3)`
      f_tilde (function)  : A function :math:`\tilde{f}(k,\xi_i)` which has to be able to broadcast
                            inputs :math:`k` with shape :math:`(m_1,m_2,m_3)` and :math:`\xi` with
                            shape :math:`(n_\xi)` to an output with shape :math:`(m_1,m_2,m_3,n_\xi)`.
                            Represents the fourier transform of
                            :math:`f(|\mathbf{r}-\mathbf{r}'|, \xi = \xi_i)` where
                            :math:`\xi(\mathbf{r}) = \xi_i` is constant over all space.
      g (torch.Tensor)    : :math:`g(\mathbf{r}')` with shape :math:`(n_1,n_2,n_3)`
      xis (torch.Tensor)  : :math:`\xi(x,y,z)` with shape :math:`(n_1,n_2,n_3)`
      kappa (int)         : Interval between the sparse :math:`\xi_i`'s.
      mode (str)          : Whether the intervals of the sparse :math:`\xi_i`'s correspond to an
                            ``arithmetic`` (default) or ``geometric`` progression.

    Returns:
      torch.Tensor: :math:`K(\mathbf{r})` with shape :math:`(n_1,n_2,n_3)`
    """
    if mode == 'arithmetic':
        # generates arithmetic progression based on κ that covers the whole ξ range (and a bit more)
        lower = (np.floor(xis.min().item() / kappa) - 3) * kappa
        upper = (np.ceil(xis.max().item() / kappa) + 3) * kappa
        xi_sparse = torch.arange(lower, upper, kappa, dtype=torch.double, device=xis.device)
        xi_sparse[xi_sparse == 0] = xis.min().item()
    elif mode == 'geometric':
        assert kappa > 1, 'κ > 1 for geometric progression based spline for field_dependent_convolution'
        # generates geometric progression based on κ that covers the whole ξ range (and a bit more)
        lower = kappa**(-(np.ceil(- np.log(xis.min().item()) / np.log(kappa)) + 3))
        N = np.ceil(np.log((xis.max().item() + 1) / lower) / np.log(kappa)) + 3
        xi_sparse = lower * kappa**torch.arange(N, dtype=torch.double, device=xis.device)
    else:
        raise ValueError('Parameter \'mode\' can only be \'arithmetic\' or \'geometric\'')
    # print(xi_sparse.min().item(), xi_sparse.max().item(), len(xi_sparse))
    g_tilde = torch.fft.rfftn(g).unsqueeze(3).expand((-1, -1, -1, len(xi_sparse)))
    conv = torch.fft.irfftn(f_tilde(k, xi_sparse) * g_tilde, s=g.shape, dim=(0, 1, 2))
    return interpolate_kernel(xi_sparse, conv, xis)
