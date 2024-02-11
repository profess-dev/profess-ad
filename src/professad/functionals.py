import numpy as np
import torch
from professad.functional_tools import wavevecs, grad_dot_grad, reduced_gradient, \
    reduced_gradient_squared, laplacian, reduced_laplacian, interpolate, field_dependent_convolution
from xitorch.integrate import solve_ivp

# ----------------------------------------------------------------
# This script contains kinetic and XC functionals for orbtial-free
# density functional theory calculations and a general "trainable"
# kinetic energy functional template.
# ----------------------------------------------------------------

J_per_Ha = 4.3597447222071e-18
eV_per_Ha = J_per_Ha / 1.602176634e-19

##############################################################################################
#                             Ion and Electron Interaction Terms                             #
##############################################################################################


def IonIon():
    """ Ion-ion interaction energy

    This is a dummy function used purely for consistency in how the energy terms are used as input
    during initialization of a system object. Inclusion of this term causes the system object to
    include an ion-ion interaction energy term during the system object's energy computations.
    """
    return None


def IonElectron(box_vecs, den, v_ext):
    r""" Ion-electron energy functional

    The ion-electron energy functional is given by

    .. math:: U_\text{ion-electron}[n] = \int d^3\mathbf{r}~ n(\mathbf{r}) v_\text{ext}(\mathbf{r}).

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density
      v_ext    (torch.Tensor) : Ionic/external potential

    Returns:
      torch.Tensor: Ion-electron interaction energy
    """
    return torch.mean(den * v_ext) * torch.abs(torch.linalg.det(box_vecs))


def Hartree(box_vecs, den):
    r""" Hartree energy functional

    The Hartree energy functional is the classical mean-field electron-electron interaction
    energy. It is given by

    .. math:: U_\text{Hartree}[n] = \frac{1}{2} \int d^3\mathbf{r} d^3\mathbf{r}'~
              \frac{n(\mathbf{r})n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} .

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: Hartree energy
    """
    den_ft = torch.fft.rfftn(den)
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    coloumb_ft = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
    # set k=0 component to zero. appropriate if the density integrates to
    # zero over the box (e.g. if neutralized by a uniform background charge).
    coloumb_ft[k2 != 0] = 4 * np.pi / k2[k2 != 0]
    pot = torch.fft.irfftn(den_ft * coloumb_ft, den.shape)
    return 0.5 * torch.mean(den * pot) * torch.abs(torch.linalg.det(box_vecs))

##############################################################################################
#                                     Kinetic Functionals                                    #
##############################################################################################

# --------------------------------------------------------------------------------------------
#                     Trainable Kinetic Energy Functional Parent Class
# --------------------------------------------------------------------------------------------


class KineticFunctional(torch.nn.Module):
    """ Template kinetic functional class

    This is a class that represents a kinetic functional which can be useful for functionals that
    require an initialized kernel to be saved for repeated use via interpolation or otherwise. This
    class is also useful for functionals with tunable parameters (e.g. machine-learned functionals)
    as it allows for Pytorch-based optimization of those parameters.
    """
    def __init__(self, init_args=None):
        """
        Args:
          init_args (tuple or None): Functional parameters for initialization
        """
        super().__init__()
        self.init_args = init_args
        self.device = torch.device('cpu')
        self.training_curve, self.validation_curve = [], []

    def initialize(self):
        """
        Sets the optimizer (default is Rprop) and make all functional parameters have
        ``requires_grad = False``.
        """
        self.to(self.device)
        self.optimizer = torch.optim.Rprop(self.parameters(), lr=0.1, step_sizes=(1e-8, 50))
        self.param_grad(False)  # default behaviour for parameters.requires_grad is False

    def set_device(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Moves all functional parameter tensors to the specified device. By default, the device
        is set to a GPU if is avaliable, otherwise the device is set as a CPU.

        Args:
          device (torch.device): Device
        """
        self.device = device
        for p in self.parameters():
            p.data = p.data.to(self.device)

    def param_grad(self, requires_grad=True):
        """
        Sets whether the functional parameters' ``requires_grad`` is ``True`` or ``False``.

        Args:
          requires_grad (boo): Whether the functional parameters' ``requires_grad`` is ``True`` or ``False``
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)

    def save(self, PATH):
        """
        Saves the functional parameters in a file.

        Args:
          PATH (string): Path to the file where the parameters are saved
        """
        torch.save([self.init_args, self.state_dict(), self.optimizer.state_dict(),
                    self.training_curve, self.validation_curve], PATH)

    @classmethod
    def load(cls, PATH):
        """
        Loads the functional parameters from a file.

        Args:
          PATH (string): Path to the file where the parameters to be loaded are
        """
        params = torch.load(PATH)
        model = cls(params[0])
        model.load_state_dict(params[1])
        model.optimizer.load_state_dict(params[2])
        model.training_curve = params[3]
        model.validation_curve = params[4]
        return model

    def grid_error(self, target, prediction, norm=False):
        """
        Utility function to compute errors on a grid.

        Args:
          target (torch.Tensor)     : The expected result
          prediction (torch.Tensor) : Prediction from the functional
          norm (boo)                : Whether to normalize the error by the range of
                                      the ``target``
        Returns:
          torch.Tensor: Mean error
        """
        # the normalization is based on the range of the target
        norm_factor = 1 if torch.all(target == 0) else (target.max() - target.min())**2
        if norm:
            return torch.mean((target - prediction).pow(2)) / norm_factor
        else:
            return torch.mean((target - prediction).pow(2))

    def scalar_error(self, target, prediction):
        """
        Utility function to compute scalar errors.

        Args:
          target (torch.Tensor)     : The expected result
          prediction (torch.Tensor) : Prediction from the functional

        Returns:
          torch.Tensor: Relative error
        """
        norm_factor = 1 if target == 0 else target * target
        return (target - prediction)**2 / norm_factor

    def update_params(self, loss):
        """
        Updates the functional parameters based on a loss function to be minimized.

        Args:
          loss (torch.Tensor): Quantity to be minimized
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --------------------------------------------------------------------------------------------
#                                Semi-local Kinetic Functionals
# --------------------------------------------------------------------------------------------


def ThomasFermi(box_vecs, den):
    r""" Thomas-Fermi functional

    The Thomas-Fermi functional is exact for the free electron gas and
    can be considered the local density approximation (LDA) for
    non-interacting kinetic energy functionals. It is given by

    .. math:: T_\text{TF}[n] = \int d^3\mathbf{r} ~\frac{3}{10} (3\pi)^{2/3} n^{5/3}(\mathbf{r})

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: Thomas-Fermi kinetic energy
    """
    ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den.pow(5 / 3)
    return torch.mean(ked) * torch.abs(torch.linalg.det(box_vecs))


def Weizsaecker(box_vecs, den):
    r""" von Weizsaecker functional

    The von Weizsaecker functional is exact for single-orbital systems
    It is given by

    .. math:: T_\text{vW}[n] = \int d^3\mathbf{r} ~\frac{1}{8} \frac{|\nabla n(\mathbf{r})|^2}{n(\mathbf{r})}

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: von Weizsaecker kinetic energy
    """
    sqrt_den = torch.zeros(den.shape, dtype=torch.double, device=den.device)
    sqrt_den[den != 0] = torch.sqrt(den[den != 0])
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    ked = 0.25 * laplacian(k2, den) - 0.5 * sqrt_den * laplacian(k2, sqrt_den)
    return torch.mean(ked) * torch.abs(torch.linalg.det(box_vecs))

# ------------------------------------ vWGTF functionals --------------------------------------


def vWGTF1(box_vecs, den):
    r""" vWGTF1 functional

    The vWGTF1 functional [`Phys. Rev. B 91, 045124 <https://doi.org/10.1103/PhysRevB.91.045124>`_]
    has a Pauli enhancement factor given by

    .. math:: F_\theta^\text{vWGTF1}(d) = 0.9892 \cdot d^{-1.2994}

    where :math:`d = n/n_0` for average density :math:`n_0`.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: vWGTF1 kinetic energy
    """
    vol = torch.abs(torch.linalg.det(box_vecs))
    N_elec = round((torch.mean(den) * vol).detach().item())
    n0 = N_elec / vol
    d = den / n0
    G = 0.9892 * d.pow(-1.2994)
    TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den**(5 / 3)
    return Weizsaecker(box_vecs, den) + torch.mean(G * TF_ked) * vol


def vWGTF2(box_vecs, den):
    r""" vWGTF2 functional

    The vWGTF2 functional [`Phys. Rev. B 91, 045124 <https://doi.org/10.1103/PhysRevB.91.045124>`_]
    has a Pauli enhancement factor given by

    .. math:: F_\theta^\text{vWGTF2}(d) = \sqrt{\frac{1}{\text{ELF}} - 1}

    with the electron localization function (ELF) parameterized as

    .. math:: \text{ELF} = \frac{1}{2} \left(1 + \tanh(5.7001 \cdot d^{0.2563} - 5.7001) \right)

    where :math:`d = n/n_0` for average density :math:`n_0`.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: vWGTF2 kinetic energy
    """
    vol = torch.abs(torch.linalg.det(box_vecs))
    N_elec = round((torch.mean(den) * vol).detach().item())
    n0 = N_elec / vol
    d = den / n0
    ELF = 0.5 * (1 + torch.tanh(5.7001 * d.pow(0.2563) - 5.7001))
    G = torch.sqrt(1 / ELF - 1)
    TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den**(5 / 3)
    return Weizsaecker(box_vecs, den) + torch.mean(G * TF_ked) * vol


# --------------------------- Luo-Karasiev-Trickey (LKT) functional --------------------------
def LuoKarasievTrickey(box_vecs, den):
    r""" Luo-Karasiev-Trickey (LKT) functional

    The Luo-Karasiev-Trickey (LKT) GGA kinetic functional
    [`Phys. Rev. B 98, 041111(R) <https://link.aps.org/doi/10.1103/PhysRevB.98.041111>`_]
    has a Pauli enhancement factor given by

    .. math:: F_\theta^\text{LKT}(s) = \frac{1}{\cosh(1.3 s)}

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: LKT kinetic energy
    """
    TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den.pow(5 / 3)
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    s = reduced_gradient(kx, ky, kz, den)
    # clamp to avoid s from growing too large, which can cause F_pauli -> 0 and get nan derivatives
    F_pauli = 1 / torch.cosh(1.3 * s.clamp(max=100))
    pauli_T = torch.mean(TF_ked * F_pauli) * torch.abs(torch.linalg.det(box_vecs))
    return Weizsaecker(box_vecs, den) + pauli_T

# ----------------------------- Pauli-Gaussian style functionals ------------------------------


class PauliGaussian(KineticFunctional):
    r""" Pauli-Gaussian functional

    The Pauli-Gaussian class of GGA kinetic energy functionals
    [`J. Phys. Chem. Lett. 2018, 9, 15, 4385–4390 <https://doi.org/10.1021/acs.jpclett.8b01926>`_,
    `J. Chem. Theory Comput. 2019, 15, 5, 3044–3055 <https://doi.org/10.1021/acs.jctc.9b00183>`_]
    have Pauli enhancement factors with the form

    .. math:: F_\theta^\text{PG}(s,q) = e^{-\mu s^2} + \beta q^2 - \lambda q s^2 + \sigma s^4
    """
    def __init__(self, init_args=None):
        r"""
        Args:
          init_args (tuple) : :math:`(\mu,~\beta,~\lambda,~\sigma)` where each parameter is a float.
                              These are key parameters of the Pauli-Gaussian functionals.
                              The default parameters are that of the PGSL0.25 functional's,
                              :math:`(\mu,~\beta,~\lambda,~\sigma) = (40/27,~0.25,~0,~0)`.
        """
        super().__init__()
        if init_args is None:
            mu, beta, lamb, sigma = 40 / 27, 0.25, 0.0, 0.0  # default to PGSL0.25
        else:
            mu, beta, lamb, sigma = init_args
        self.mu = torch.nn.Parameter(torch.tensor([mu], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.lamb = torch.nn.Parameter(torch.tensor([lamb], dtype=torch.double, device=self.device))
        self.sigma = torch.nn.Parameter(torch.tensor([sigma], dtype=torch.double, device=self.device))
        self.initialize()

    def set_PG1(self):
        r""" Set :math:`(\mu,~\beta,~\lambda,~\sigma) = (1,~0,~0,~0)` """
        self.mu[0], self.beta[0] = 1.0, 0.0
        self.lamb[0], self.sigma[0] = 0.0, 0.0

    def set_PGS(self):
        r""" Set :math:`(\mu,~\beta,~\lambda,~\sigma) = (40/27,~0,~0,~0)` """
        self.mu[0], self.beta[0] = 40 / 27, 0.0
        self.lamb[0], self.sigma[0] = 0.0, 0.0

    def set_PGSL025(self):
        r""" Set :math:`(\mu,~\beta,~\lambda,~\sigma) = (40/27,~0.25,~0,~0)` """
        self.mu[0], self.beta[0] = 40 / 27, 0.25
        self.lamb[0], self.sigma[0] = 0.0, 0.0

    def set_PGSLr(self):
        r""" Set :math:`(\mu,~\beta,~\lambda,~\sigma) = (40/27,~0.25,~0.4,~0.2)` """
        self.mu[0], self.beta[0] = 40 / 27, 0.25
        self.lamb[0], self.sigma[0] = 0.4, 0.2

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: Pauli-Gaussian kinetic energy
        """
        TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den.pow(5 / 3)
        kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
        s2 = reduced_gradient_squared(kx, ky, kz, den)
        q = reduced_laplacian(k2, den)
        F_enh = torch.exp(- torch.abs(self.mu) * s2) + torch.abs(self.beta) * q.pow(2) \
                - torch.abs(self.lamb) * q * s2 + torch.abs(self.sigma) * s2.pow(2)
        pauli_T = torch.mean(TF_ked * F_enh) * torch.abs(torch.linalg.det(box_vecs))
        return Weizsaecker(box_vecs, den) + pauli_T

# ----------------------------------- Yukawa GGA functionals ----------------------------------


class YukawaGGA(KineticFunctional):
    r""" Yukawa GGA functional

    The Yukawa GGA class of kinetic energy functionals
    [`Phys. Rev. B 103, 155127 <https://link.aps.org/doi/10.1103/PhysRevB.103.155127>`_,
    `Computation 2022, 10(2), 30 <https://doi.org/10.3390/computation10020030>`_]
    have Pauli enhancement factors :math:`F_\theta(y, s^2, q)` that depend on the Yukawa
    potential term,

    .. math:: y_{\alpha \beta}(\mathbf{r}) = \frac{3\pi\alpha^2}{4 k_F(\mathbf{r}) n^{\beta-1}(\mathbf{r})}
              \int d^3\mathbf{r}' \frac{n^\beta(\mathbf{r}')
              e^{-\alpha k_F(\mathbf{r}) |\mathbf{r}-\mathbf{r}'|}}{|\mathbf{r}-\mathbf{r}'|}

    where :math:`k_F(\mathbf{r}) = [3\pi^2 n(\mathbf{r})]^{1/3}`, as an ingredient besides
    the reduced gradient and laplacian.
    """
    def __init__(self, init_args=None):
        r"""
        Args:
          init_args (tuple) : :math:`(\alpha,~\beta,~F_\theta,~\kappa)` where (\alpha,~\beta) are
                              key parameters of the Yukawa GGA functional,
                              :math:`F_\theta` is the Pauli enhancement factor that takes as arguments
                              the Yukawa descriptor, reduced gradient and reduced laplacian,
                              i.e. :math:`F_\theta(y, s^2, q)`,
                              and :math:`\kappa` is a parameter for the spline-based field dependent
                              convolution. Note that a geometric progression based spline is used so
                              :math:`\kappa > 1`
                              The default parameters are that of the Yuk1 functional's,
                              :math:`(\mu,~\beta,~f,~\kappa) = (1,~1,~y,~1.2)`.
        """
        super().__init__()
        if init_args is None:
            alpha, beta, func, kappa = 1, 1, lambda y, s2, q: y, 1.2  # default to yuk1
        else:
            alpha, beta, func, kappa = init_args
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.F_pauli = func
        self.kappa = kappa
        self.debug = False
        self.mode = 'geometric'
        self.spline = True
        self.initialize()

    def yukawa_descriptor(self, k2, den):
        # ξ(r) = α k_F(r)
        k_F = (3 * np.pi * np.pi * den).pow(1 / 3)
        xis = self.alpha * k_F
        # g(r') = [n(r')]^β
        g = den.pow(self.beta)
        if self.spline:  # use spline method
            # Yukawa potential in Fourier space, K(q,ξ) = 4π/(q²+ξ²)
            def K_tilde(k2, xi_sparse):
                return 4 * np.pi / (k2.unsqueeze(3).expand((-1, -1, -1, len(xi_sparse))) + xi_sparse.pow(2))
            # Computes u(r) = ∫d³r' K(|r-r'|,ξ(r)) g(r')
            if self.debug:
                print('ξ_min: {:.6g}, ξ_max: {:.6g}'.format(xis.min().item(), xis.max().item()))
            u = field_dependent_convolution(k2, K_tilde, g, xis, kappa=self.kappa, mode=self.mode)
        else:  # use naive method
            u = torch.empty(den.shape, dtype=torch.double, device=self.device)
            for l in range(den.shape[0]):
                for m in range(den.shape[1]):
                    for n in range(den.shape[2]):
                        K = 4 * np.pi / (k2 + xis[l, m, n].pow(2))
                        u_lmn = torch.fft.irfftn(torch.fft.rfftn(g) * K, xis.shape)[l, m, n]
                        u[l, m, n] = u_lmn

        # y = 3πα²/(4 k_F n^(β-1)) u
        y = 3 * np.pi * self.alpha.pow(2) / (4 * k_F * den.pow(self.beta - 1)) * u
        return y

    def T_a(self, a, x):
        return 1 + (2 / a) * torch.tanh((a / 2) * x)

    def set_yuk1(self):
        r"""
        Set :math:`(\alpha,~\beta) = (1,~1)` and

        .. math:: F_\theta(y,~s^2,~q) = y

        """
        self.alpha[0] = 1; self.beta[0] = 1
        self.F_pauli = lambda y, s2, q: y

    def set_yuk2(self):
        r"""
        Set :math:`(\alpha,~\beta) = (1.36297,~1)` and

        .. math:: F_\theta(y,~s^2,~q) = y~ \left(1 + \frac{40}{27} (q-s^2) \right)

        .. warning::
            Unstable for density optimizations as it violates Pauli positivity
        """
        self.alpha[0] = 1.3629; self.beta[0] = 1
        self.F_pauli = lambda y, s2, q: y * (1 + 40 / 27 * (q - s2))

    def set_yuk3(self, a=4):
        r"""
        Set :math:`(\alpha,~\beta) = (1.36297,~1)` and

        .. math:: F_\theta(y,~s^2,~q) = y~ T_a \left(-\frac{40}{27} (q-s^2) \right)

        where

        .. math:: T_a(x) = \frac{4}{a} \frac{e^{ax}}{e^{ax} + 1} + \frac{a-2}{a}.
        """
        self.alpha[0] = 1.3629; self.beta[0] = 1

        def func(y, s2, q):
            x = 40 / 27 * (q - s2)
            return y * self.T_a(a, x)
        self.F_pauli = func

    def set_yuk4(self, a=3.3):
        r"""
        Set :math:`(\alpha,~\beta) = (1.36297,~1)` and

        .. math:: F_\theta(y,~s^2,~q) = y~ T_a \left(-\frac{40}{27} s^2 \right)
                  T_2\left(-\frac{40}{27} q \right)

        where

        .. math:: T_a(x) = \frac{4}{a} \frac{e^{ax}}{e^{ax} + 1} + \frac{a-2}{a}.
        """
        self.alpha[0] = 1.3629; self.beta[0] = 1

        def func(y, s2, q):
            xq = 40 / 27 * q
            xp = - 40 / 27 * s2
            return y * self.T_a(a, xp) * self.T_a(2, xq)
        self.F_pauli = func

    def set_yuk2beta(self, alpha, beta):
        r"""
        Set :math:`(\alpha,~\beta)` according to user input and

        .. math:: F_\theta(y,~s^2,~q) = 1 - G_0 + y(G_0 + G)

        where

        .. math:: G_0 = \frac{\alpha^2 (\alpha^2 - 60)}{108 \beta (9\beta - 10)}

        and

        .. math:: G = \left(\frac{40}{27\beta} - \frac{4}{\alpha^2} (\beta-1) G_0\right) (q - \beta s^2).

        .. warning::
            Unstable for density optimizations as it violates Pauli positivity
        """
        self.alpha[0] = alpha; self.beta[0] = beta

        def func(y, s2, q):
            G0 = self.alpha.pow(2) * (self.alpha.pow(2) - 60) \
                 / (108 * self.beta * (9 * self.beta - 10))
            G = (40 / 27 / self.beta - 4 / self.alpha.pow(2) * (self.beta - 1) * G0) \
                * (q - self.beta * s2)
            return 1 - G0 + y * (G0 + G)
        self.F_pauli = func

    def set_yuk3beta(self, alpha, beta, a=2):
        r"""
        Set :math:`(\alpha,~\beta)` according to user input and

        .. math:: F_\theta(y,~s^2,~q) = T_a\left( - G_0 + y(G_0 + G) \right)

        where

        .. math:: G_0 = \frac{\alpha^2 (\alpha^2 - 60)}{108 \beta (9\beta - 10)},

        .. math:: G = \left(\frac{40}{27\beta} - \frac{4}{\alpha^2} (\beta-1) G_0\right) (q - \beta s^2)

        and

        .. math:: T_a(x) = \frac{4}{a} \frac{e^{ax}}{e^{ax} + 1} + \frac{a-2}{a}.
        """
        self.alpha[0] = alpha; self.beta[0] = beta

        def func(y, s2, q):
            G0 = self.alpha.pow(2) * (self.alpha.pow(2) - 60) \
                 / (108 * self.beta * (9 * self.beta - 10))
            G = (40 / 27 / self.beta - 4 / self.alpha.pow(2) * (self.beta - 1) * G0) \
                * (q - self.beta * s2)
            return self.T_a(a, - G0 + y * (G0 + G))
        self.F_pauli = func

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: Yukawa GGA kinetic energy
        """
        vol = torch.abs(torch.linalg.det(box_vecs))
        kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
        y = self.yukawa_descriptor(k2, den)
        s2 = reduced_gradient_squared(kx, ky, kz, den)
        q = reduced_laplacian(k2, den)
        F_pauli = self.F_pauli(y, s2, q)
        TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den**(5 / 3)
        return Weizsaecker(box_vecs, den) + torch.mean(TF_ked * F_pauli) * vol


# --------------------------------------------------------------------------------------------
#                                 Non-local Kinetic Functionals
# --------------------------------------------------------------------------------------------

# -------------------------- Auxiliary Lindhard Response Functions ---------------------------


def G_inv_lind_analytical(eta):
    return 0.5 + ((1 - eta.pow(2)) / (4 * eta)) * torch.log(torch.abs((1 + eta) / (1 - eta)))


def G_inv_lind(eta):
    G_inv_lind = G_inv_lind_analytical(eta)

    G_inv = torch.empty(eta.shape, dtype=torch.double, device=eta.device)
    G_inv[eta == 0.0] = 1.0
    G_inv[eta == 1.0] = 0.5
    G_inv[(eta != 0.0) & (eta != 1.0)] = G_inv_lind[(eta != 0.0) & (eta != 1.0)]
    return G_inv


def G_inv_lindhard(box_vecs, den):
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    # important to do it this way instead of just n0 = torch.mean(den)
    N_elec = (torch.mean(den) * torch.abs(torch.linalg.det(box_vecs))).item()
    n0 = N_elec / (torch.abs(torch.linalg.det(box_vecs)))
    k_F = (3 * np.pi * np.pi * n0).pow(1 / 3)
    eta = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
    eta[k2 != 0] = torch.sqrt(k2[k2 != 0]) / (2 * k_F)
    return eta, G_inv_lind(eta)

# ------------------------------ Wang-Teter style functionals -------------------------------


def non_local_KEF(box_vecs, den, alpha, beta):
    vol = torch.abs(torch.linalg.det(box_vecs))
    N_elec = (torch.mean(den) * vol).item()
    n0 = N_elec / vol
    eta, G_inv = G_inv_lindhard(box_vecs, den)
    kernel = 5 / (9 * alpha * beta * n0.pow(alpha + beta - 5 / 3)) * (1 / G_inv - 3 * eta.pow(2) - 1)
    conv = torch.fft.irfftn(kernel * torch.fft.rfftn(den.pow(beta) - n0.pow(beta)), den.shape)
    NL = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * torch.mean((den.pow(alpha) - n0.pow(alpha)) * conv) * vol
    return NL


def WangTeter(box_vecs, den):
    r""" Wang-Teter (WT) functional

    The Wang-Teter (WT)  functional [`Phys. Rev. B 45, 13196 <https://doi.org/10.1103/PhysRevB.45.13196>`_]
    is a Wang-Teter style non-local kinetic functional with a density-independent kernel and
    parameters :math:`(\alpha,~\beta) = (5/6,~5/6)`.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: WT kinetic energy
    """
    vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
    return vW + TF + non_local_KEF(box_vecs, den, alpha=5 / 6, beta=5 / 6)


def Perrot(box_vecs, den):
    r""" Perrot functional

    The Perrot functional [`J. Phys.: Condens. Matter 6 431
    <https://iopscience.iop.org/article/10.1088/0953-8984/6/2/014>`_]
    is a Wang-Teter style non-local kinetic functional with a density-independent kernel and
    parameters :math:`(\alpha,~\beta) = (1,~1)`.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: Perrot kinetic energy
    """
    vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
    return vW + TF + non_local_KEF(box_vecs, den, alpha=1, beta=1)


def SmargiassiMadden(box_vecs, den):
    r""" Smargiassi-Madden (SM) functional

    The Smargiassi-Madden (SM) functional [`Phys. Rev. B 49, 5220 <https://doi.org/10.1103/PhysRevB.49.5220>`_]
    is a non-local kinetic Wang-Teter style functional with a density-independent kernel and
    parameters :math:`(\alpha,~\beta) = (1/2,~1/2)`.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: SM kinetic energy
    """
    vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
    return vW + TF + non_local_KEF(box_vecs, den, alpha=0.5, beta=0.5)


def WangGovindCarter98(box_vecs, den):
    r""" Wang-Govind-Carter 98 (WGC98) functional

    The Wang-Govind-Carter 98 (WGC98) functional [`Phys. Rev. B 58, 13465 <https://doi.org/10.1103/PhysRevB.58.13465>`_]
    is a non-local kinetic Wang-Teter style functional with a density-independent kernel and
    parameters :math:`(\alpha,~\beta) = ((5+\sqrt{5})/6,~(5-\sqrt{5})/6)`.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: WGC98 kinetic energy
    """
    vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
    return vW + TF + non_local_KEF(box_vecs, den, alpha=(5 + np.sqrt(5)) / 6, beta=(5 - np.sqrt(5)) / 6)


class WangTeterStyleFunctional(KineticFunctional):
    r""" Wang-Teter style functional

    This class represents a general Wang-Teter style non-local kinetic energy functional with user-chosen
    :math:`(\alpha,~\beta)` parameters. Conventional choices of these parameters include

    * :math:`(\alpha,~\beta)=(5/6,5/6)` in the Wang-Teter functional
      [`Phys. Rev. B 45, 13196 <https://doi.org/10.1103/PhysRevB.45.13196>`_]

    * :math:`(\alpha,~\beta)=(1,1)` in the Perrot functional
      [`J. Phys.: Condens. Matter 6 431 <https://iopscience.iop.org/article/10.1088/0953-8984/6/2/014>`_]

    * :math:`(\alpha,~\beta)=(1/2,1/2)` in the Smargiassi-Madden functional
      [`Phys. Rev. B 49, 5220 <https://doi.org/10.1103/PhysRevB.49.5220>`_]

    * :math:`(\alpha,~\beta) = ((5+\sqrt{5})/6,~(5-\sqrt{5})/6)` in the Wang-Govind-Carter 98 functional
      [`Phys. Rev. B 58, 13465 <https://doi.org/10.1103/PhysRevB.58.13465>`_]

    This class also allows for the use of a Pauli-positivity stabilization function
    [`J. Phys. Chem. A 2021, 125, 7, 1650–1660 <https://doi.org/10.1021/acs.jpca.0c11030>`_].
    """
    def __init__(self, init_args=None):
        r"""
        Args:
          init_args (tuple) : :math:`(\alpha,~\beta,~f)` where :math:`(\alpha,~\beta)` are floats and
                              :math:`f` is a function. :math:`(\alpha,~\beta)` are key parameters of the
                              Wang-Teter style functionals while :math:`f` is the Pauli-positivity stabilization
                              function, which must obey :math:`f(0) = 1`. The default parameters are
                              :math:`(\alpha,~\beta,~f) = (5/6,~5/6,~f(x) = 1 +x)`.
        """
        super().__init__()
        if init_args is None:
            alpha, beta, f = 5 / 6, 5 / 6, lambda x: 1 + x  # defaults to WT
        else:
            alpha, beta, f = init_args
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.f = f
        zero = torch.zeros((1,), dtype=torch.double, device=self.device, requires_grad=True)
        assert self.f(zero).item() == 1.0, 'Requires f(0) = 1'
        self.fprime0 = torch.autograd.grad(self.f(zero), zero)[0].item()
        self.initialize()

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: WT-style functional kinetic energy
        """
        vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
        T_NL = non_local_KEF(box_vecs, den, self.alpha, self.beta) / self.fprime0
        return vW + TF * self.f(T_NL / TF)

# --------------------------- Wang-Govind-Carter 99 functional ----------------------------


class WangGovindCarter99(KineticFunctional):
    """ Wang-Govind-Carter 99 (WGC99) functional

    The Wang-Govind-Carter 99 (WGC99) functional [`Phys. Rev. B 60, 16350 <https://doi.org/10.1103/PhysRevB.60.16350>`_]
    is non-local kinetic functional that extends the Wang-Govind-Carter 98 functional to have a density-dependent
    kernel. In practice however, a Taylor-expansion is used to avoid the unattractive computational scaling resulting
    from the density-dependent kernel.
    """
    def __init__(self, init_args=None):
        r"""
        Args:
          init_args (tuple) : :math:`(\alpha,~\beta,~\gamma,~\kappa)` where each parameter is a float.
                              :math:`\alpha,~\beta,~\gamma` are key parameters of the WGC99 functional,
                              while :math:`\kappa` is an enhancement factor for the reference uniform
                              density such that :math:`n^* = κ n_0` The default parameters are
                              :math:`(\alpha,~\beta,~\gamma,~\kappa) = ((5+\sqrt{5})/6,~(5-\sqrt{5})/6,~2.7,~1)`
        """
        super().__init__()
        if init_args is None:
            alpha, beta, gamma, kappa = (5 + np.sqrt(5)) / 6, (5 - np.sqrt(5)) / 6, 2.7, 1.0
        else:
            alpha, beta, gamma, kappa = init_args
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.gamma = torch.nn.Parameter(torch.tensor([gamma], dtype=torch.double, device=self.device))
        self.kappa = torch.nn.Parameter(torch.tensor([kappa], dtype=torch.double, device=self.device))
        self.initialize()
        self.kernel = None
        self.eta = None

    def __get_Ai(self, num_terms):
        ai = torch.zeros((num_terms + 1,), dtype=torch.double, device=self.device)
        for index in range(num_terms + 1):
            i = index - 1
            if i == -1:
                ai[index] = 3
            else:
                for j in range(-1, i):
                    ai[index] += -3 * ai[j + 1] / (4 * (i - j + 1)**2 - 1)
        Ai = torch.empty((num_terms,), dtype=torch.double, device=self.device)
        Ai[0] = ai[1] - 1.0
        Ai[1:] = ai[2:]
        return Ai

    def __get_Bi(self, num_terms):
        bi = torch.zeros((num_terms,), dtype=torch.double, device=self.device)
        for i in range(num_terms):
            if i == 0:
                bi[i] = 1
            else:
                for j in range(0, i):
                    bi[i] += bi[j] / (4 * (i - j)**2 - 1)
        Bi = torch.empty((num_terms,), dtype=torch.double, device=self.device)
        Bi[0] = 0.0
        Bi[1] = bi[1] - 3.0
        Bi[2:] = bi[2:]
        return Bi

    def generate_kernel(self, num_terms=100):
        """
        Generates the WGC99 kernel based on its analytical form given in
        `Phys. Rev. B 78, 045105 <https://doi.org/10.1103/PhysRevB.78.045105>`_.

        Args:
          num_terms (int) : Number of terms at which the infinite summation is truncated
        """
        eta = self.eta
        u = 3 * (self.alpha + self.beta) - self.gamma / 2
        v = u.pow(2) - 36 * self.alpha * self.beta

        Ai = self.__get_Ai(num_terms)
        Bi = self.__get_Bi(num_terms)
        i = torch.arange(num_terms, dtype=torch.double, device=self.device)

        # --------------------------- compute homogeneous solution ---------------------------

        # Note: Ss sum starts from i=1 but i=0 term is zero anyway
        Sd = torch.sum(Ai / ((u + 2 * i).pow(2) - v) - Bi / ((u - 2 * i).pow(2) - v))
        Ss = - 2 * torch.sum(i * (Ai / ((u + 2 * i).pow(2) - v) + Bi / ((u - 2 * i).pow(2) - v)))

        if v > 0:
            c1 = torch.sign(u) * ((torch.sqrt(v) - u) * Sd + Ss)
            c2 = torch.sign(u) * ((torch.sqrt(v) + u) * Sd - Ss) / (2 * torch.sqrt(v))
        elif v == 0:
            c1 = torch.sign(u) * Sd
            c2 = torch.sign(u) * (Ss - u * Sd)
        else:
            c1 = torch.sign(u) * Sd
            c2 = torch.sign(u) * (Ss - u * Sd) / torch.sqrt(-v)

        C1 = torch.empty(eta.shape, dtype=torch.double, device=self.device)
        C2 = torch.empty(eta.shape, dtype=torch.double, device=self.device)

        if u >= 0:
            C1[eta <= 1], C1[eta > 1] = c1, 0
            C2[eta <= 1], C2[eta > 1] = c2, 0
        else:
            C1[eta <= 1], C1[eta > 1] = 0, c1
            C2[eta <= 1], C2[eta > 1] = 0, c2

        H0 = torch.zeros(eta.shape, dtype=torch.double, device=self.device)
        H1 = torch.zeros(eta.shape, dtype=torch.double, device=self.device)
        H2 = torch.zeros(eta.shape, dtype=torch.double, device=self.device)
        eta_nz, C1_nz, C2_nz = eta[eta != 0], C1[eta != 0], C2[eta != 0]

        if v > 0:
            x = u + torch.sqrt(v)
            y = u - torch.sqrt(v)
            H0[eta != 0] = C1_nz * eta_nz.pow(x) + C2_nz * eta_nz.pow(y)
            H1[eta != 0] = C1_nz * x * eta_nz.pow(x - 1) + C2_nz * y * eta_nz.pow(y - 1)
            H2[eta != 0] = C1_nz * x * (x - 1) * eta_nz.pow(x - 2) + C2_nz * y * (y - 1) * eta_nz.pow(y - 2)
        elif v == 0:
            H0[eta != 0] = eta_nz.pow(u) * (C2_nz * torch.log(eta_nz) + C1_nz)
            H1[eta != 0] = C2_nz * eta_nz.pow(u - 1) * (1 + u * torch.log(eta_nz)) + C1_nz * u * eta_nz.pow(u - 1)
            H2[eta != 0] = C2_nz * ((u - 1) * eta_nz.pow(u - 2) * (1 + u * torch.log(eta_nz)) + eta_nz.pow(u - 2)) \
                         + C1_nz * u * (u - 1) * eta_nz.pow(u - 2)
        else:
            sqrtv = torch.sqrt(-v)
            tc = torch.cos(sqrtv * torch.log(eta_nz))
            ts = torch.sin(sqrtv * torch.log(eta_nz))
            H0[eta != 0] = eta_nz.pow(u) * (C1_nz * tc + C2_nz * ts)
            H1[eta != 0] = eta_nz.pow(u - 1) * (C1_nz * (u * tc - sqrtv * ts) + C2_nz * (u * ts + sqrtv * tc))
            H2[eta != 0] = (u - 1) * eta_nz.pow(u - 2) * C1_nz * (u * tc - sqrtv * ts) \
                         - sqrtv * eta_nz.pow(u - 2) * C1_nz * (u * ts + sqrtv * tc) \
                         + (u - 1) * eta_nz.pow(u - 2) * C2_nz * (u * ts + sqrtv * tc) \
                         + sqrtv * eta_nz.pow(u - 2) * C2_nz * (u * tc - sqrtv * ts)

        # ---------------------------- compute particular solution -----------------------------
        P0 = torch.zeros(eta.shape, dtype=torch.double, device=eta.device)
        P1 = torch.zeros(eta.shape, dtype=torch.double, device=eta.device)
        P2 = torch.zeros(eta.shape, dtype=torch.double, device=eta.device)

        # Note: η<1 sum starts from i=1 but i=0 term is zero anyway becaue B0 = 0
        eta_leq1 = eta[(eta <= 1) & (eta != 0)]
        eta_leq1 = eta_leq1.unsqueeze(-1).expand(eta_leq1.shape + (num_terms,))
        aux = Bi / ((u - 2 * i).pow(2) - v)

        P0[(eta <= 1) & (eta != 0)] = torch.sum(aux * eta_leq1.pow(2 * i), axis=-1)
        P1[(eta <= 1) & (eta != 0)] = torch.sum(aux * (2 * i) * eta_leq1.pow(2 * i - 1), axis=-1)
        P2[(eta <= 1) & (eta != 0)] = torch.sum(aux * (2 * i) * (2 * i - 1) * eta_leq1.pow(2 * i - 2), axis=-1)

        eta_g1 = eta[eta > 1]
        eta_g1 = eta_g1.unsqueeze(-1).expand(eta_g1.shape + (num_terms,))
        aux = Ai / ((u + 2 * i).pow(2) - v)
        P0[eta > 1] = torch.sum(aux / eta_g1.pow(2 * i), axis=-1)
        P1[eta > 1] = torch.sum(aux * (-2 * i) / eta_g1.pow(2 * i + 1), axis=-1)
        P2[eta > 1] = torch.sum(aux * (2 * i) * (2 * i + 1) / eta_g1.pow(2 * i + 2), axis=-1)

        w0 = H0 + P0
        w1 = H1 + P1
        w2 = H2 + P2

        self.kernel = torch.cat([w0.unsqueeze(0), w1.unsqueeze(0), w2.unsqueeze(0)])

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: WGC99 kinetic energy
        """
        vol = torch.abs(torch.linalg.det(box_vecs))
        kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
        N_elec = round((torch.mean(den) * vol).detach().item())
        n0 = N_elec / vol
        n_ref = self.kappa * n0

        k_F = (3 * np.pi * np.pi * n_ref).pow(1 / 3)
        eta = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
        eta[k2 != 0] = torch.sqrt(k2[k2 != 0]) / (2 * k_F)

        # retrieve/construct the kernel
        if self.kernel is None:
            self.eta = eta
            self.generate_kernel()
        elif not torch.equal(self.eta, eta):
            self.eta = eta
            self.generate_kernel()

        T = 20 * n_ref.pow(5 / 3 - self.alpha - self.beta)
        w0, w1, w2 = T * self.kernel
        K1 = - eta * w1 / (6 * n_ref)
        K2 = (eta.pow(2) * w2 + (7 - self.gamma) * eta * w1) / (36 * n_ref.pow(2))
        K3 = (eta.pow(2) * w2 + (1 + self.gamma) * eta * w1) / (36 * n_ref.pow(2))

        theta = den - n_ref

        conv = torch.fft.irfftn(w0 * torch.fft.rfftn(den.pow(self.beta)), den.shape) \
             + theta * torch.fft.irfftn(K1 * torch.fft.rfftn(den.pow(self.beta)), den.shape) \
             + torch.fft.irfftn(K1 * torch.fft.rfftn(den.pow(self.beta) * theta), den.shape) \
             + theta.pow(2) / 2 * torch.fft.irfftn(K2 * torch.fft.rfftn(den.pow(self.beta)), den.shape) \
             + torch.fft.irfftn(K2 * torch.fft.rfftn(den.pow(self.beta) * theta.pow(2) / 2), den.shape) \
             + theta * torch.fft.irfftn(K3 * torch.fft.rfftn(den.pow(self.beta) * theta), den.shape)

        T_NL = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * torch.mean(den.pow(self.alpha) * conv) * vol
        vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
        return vW + TF + T_NL


# ------------------------------- Foley-Madden functional ----------------------------------

class FoleyMadden(KineticFunctional):
    """ Foley-Madden (FM) functional

    The Foley-Madden (FM) functional [`Phys. Rev. B 53, 10589 <https://doi.org/10.1103/PhysRevB.53.10589>`_]
    is a non-local kinetic functional that includes a kernel meant to enforce the correct quadratic response
    in the homogeneous electron gas limit, in addition to the Wang-Teter style non-local kernel integral term
    meant to enforce the correct linear response.
    """
    def __init__(self, init_args=None):
        r"""
        Args:
          init_args (tuple) : :math:`(\alpha,~\beta,~f)` where :math:`(\alpha,~\beta)` are floats and
                              :math:`f` is a function. :math:`(\alpha,~\beta)` are key parameters of the
                              Foley-Madden functional while :math:`f` is the Pauli-positivity stabilization
                              function, which must obey :math:`f(0) = f'(0) = 1`. The default parameters are
                              :math:`(\alpha,~\beta,~f) = (5/6,~1,~f(x) = 1 +x)`.
        """
        super().__init__()
        if init_args is None:
            alpha, beta, f = 5 / 6, 1, lambda x: 1 + x  # defaults to FM(5/6,1)
        else:
            alpha, beta, f = init_args
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.f = f
        zero = torch.zeros((1,), dtype=torch.double, device=self.device, requires_grad=True)
        assert self.f(zero).item() == 1.0, 'Requires f(0) = 1'
        assert torch.autograd.grad(self.f(zero), zero)[0].item() == 1.0, 'Requires f\'(0) = 1'
        self.initialize()

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: FM kinetic energy
        """
        vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)

        vol = torch.abs(torch.linalg.det(box_vecs))
        N_elec = round((torch.mean(den) * vol).detach().item())
        n0 = N_elec / vol
        k_F = (3 * np.pi * np.pi * n0).pow(1 / 3)

        eta, G_inv = G_inv_lindhard(box_vecs, den)
        q = 2 * eta

        C_TF = 0.3 * (3 * np.pi * np.pi)**(2 / 3)
        kernel = C_TF * 5 / (9 * self.alpha**2 * n0.pow(2 * self.alpha - 5 / 3)) * (1 / G_inv - 3 * eta.pow(2) - 1)
        conv = torch.fft.irfftn(kernel * torch.fft.rfftn(den.pow(self.alpha) - n0.pow(self.alpha)), den.shape)
        NL1 = torch.mean((den.pow(self.alpha) - n0.pow(self.alpha)) * conv) * vol

        K_delta = self.alpha.pow(2) * n0.pow(2 * self.alpha - 1) / 18 * k_F.pow(2) * (6 * self.alpha - 5) * kernel

        f1 = torch.zeros(q.shape, dtype=torch.double, device=den.device)
        q1, q2 = q[q <= 1.95], q[q > 1.95]
        f1[q <= 1.95] = 0.4 * q1.pow(2) / (1 + (q1 / 2.33).pow(10))
        f1[q > 1.95] = 0.06 / (q2 - 1.835).pow(0.75) + 0.05 * (q2 - 1.8) * torch.exp(-2.5 * (q2 - 2)) + 1

        f2 = torch.zeros(q.shape, dtype=torch.double, device=den.device)
        q1 = q[q != 0]
        f2[q == 0] = 1.0
        f2[q != 0] = 0.5 + (q1.pow(2) - 4) / (8 * q1) * torch.log(torch.abs((2 - q1) / (2 + q1)))

        f3 = torch.zeros(q.shape, dtype=torch.double, device=den.device)
        q1, q2 = q[q <= 1.84], q[q > 1.84]
        f3[q <= 1.84] = (-1 / 81 * q1.pow(2) - 0.002 * q1.pow(4)) / (1 + (q1 / 1.955).pow(28))
        f3[q > 1.84] = -0.055 * torch.exp(-4.2 * (q2 - 1.84))

        f4 = torch.zeros(q.shape, dtype=torch.double, device=den.device)
        q1, q2 = q[q <= 2], q[q > 2]
        f4[q <= 2] = 1
        f4[q > 2] = torch.exp(-3 * (q2 - 2))

        f5 = torch.zeros(q.shape, dtype=torch.double, device=den.device)
        q1, q2 = q[q <= 2.15], q[q > 2.15]
        f5[q <= 2.15] = 0.02 * torch.exp(-30 * (q1 - 2.15).pow(2))
        f5[q > 2.15] = 0.02 * torch.exp(-1.8 * (q2 - 2.15).pow(2))

        f6 = - 0.017 * torch.exp(-(q - 3).pow(2))

        f7 = torch.zeros(q.shape, dtype=torch.double, device=den.device)
        q1, q2, q3 = q[q <= 0.7], q[(q > 0.7) & (q <= 1.95)], q[q > 1.95]
        f7[q <= 0.7] = 0.0
        f7[(q > 0.7) & (q <= 1.95)] = (q2 - 1.95) / 1.25 + 1
        f7[q > 1.95] = torch.exp(-2 * (q3 - 1.95))

        delta_n_b_ft = torch.fft.rfftn(den.pow(self.beta) - n0.pow(self.beta))

        F1 = torch.fft.irfftn(delta_n_b_ft * f1, den.shape)
        F2 = torch.fft.irfftn(delta_n_b_ft * f1 * q.pow(4), den.shape)

        f1_over_q2 = 0.4 * torch.ones(q.shape, dtype=torch.double, device=den.device)
        f1_over_q2[q != 0] = f1[q != 0] / q[q != 0].pow(2)
        F3 = torch.fft.irfftn(delta_n_b_ft * f1_over_q2, den.shape)

        F4 = torch.fft.irfftn(delta_n_b_ft * f1 * q.pow(2), den.shape)
        F5 = torch.fft.irfftn(delta_n_b_ft * f2 * f3, den.shape)
        F6 = torch.fft.irfftn(delta_n_b_ft * f2, den.shape)
        F7 = torch.fft.irfftn(delta_n_b_ft * f5, den.shape)
        F8 = torch.fft.irfftn(delta_n_b_ft * f4, den.shape)
        F9 = torch.fft.irfftn(delta_n_b_ft * f6, den.shape)
        F10 = torch.fft.irfftn(delta_n_b_ft * f7, den.shape)
        F11 = torch.fft.irfftn(delta_n_b_ft * K_delta, den.shape)

        aux_ked_NL2 = - 13 / 540 * F1.pow(3) - 1 / 40 * F2 * F3.pow(2) + 1 / 20 * F4 * F3 * F1 \
                      + 3 * F5 * F6.pow(2) + 3 * F7 * F8.pow(2) + 3 * F9 * F10.pow(2) \
                      + 3 * F11 * (den.pow(self.beta) - n0.pow(self.beta)).pow(2)
        NL2 = - k_F.pow(2) / self.beta.pow(3) / n0.pow(3 * self.beta - 1) * torch.mean(aux_ked_NL2) * vol
        return vW + TF * self.f((NL1 + NL2) / TF)

# ------------------------------------ KGAP functional -------------------------------------


def G_inv_gap(box_vecs, den, E_gap):
    """ Linear response function of a gapped jellium. """
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)

    N_elec = round((torch.mean(den) * torch.abs(torch.linalg.det(box_vecs))).detach().item())
    n0 = N_elec / (torch.abs(torch.linalg.det(box_vecs)))

    k_F = (3 * np.pi * np.pi * n0).pow(1 / 3)
    eta = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
    eta[k2 != 0] = torch.sqrt(k2[k2 != 0]) / (2 * k_F)

    delta = 2 * (E_gap / eV_per_Ha) / k_F.pow(2)

    aux_p = 4 * (eta + eta.pow(2))
    aux_m = 4 * (eta - eta.pow(2))
    G_inv = torch.ones(eta.shape, dtype=torch.double, device=den.device)
    G_inv[eta != 0] = 0.5 - delta * (torch.arctan(aux_p[eta != 0] / delta) + torch.arctan(aux_m[eta != 0] / delta)) \
                                    / (8 * eta[eta != 0]) \
                      + (delta * delta / 128 / eta[eta != 0].pow(3) + 1 / 8 / eta[eta != 0] - eta[eta != 0] / 8) \
                      * torch.log((delta * delta + aux_p[eta != 0].pow(2)) / (delta * delta + aux_m[eta != 0].pow(2)))
    if delta != 0:
        G_inv[eta == 0] = 0
    return eta, G_inv


def KGAP(box_vecs, den, E_gap, f=lambda x: 1 + x):
    """ KGAP functional

    The KGAP functional [`Phys. Rev. B 97, 205137 <https://doi.org/10.1103/PhysRevB.97.205137>`_]
    is a Wang-Teter style non-local kinetic functional with a density-independent kernel that allows
    the functional to satisfy the homogeneous limit linear response of a gapped-jellium, instead of
    the usual Lindhard response of the free electron gas.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density
      E_gap    (float)        : Band gap of the system of interest (in eV)

    Returns:
      torch.Tensor: KGAP kinetic energy
    """
    zero = torch.zeros((1,), dtype=torch.double, device=den.device, requires_grad=True)
    assert f(zero).item() == 1.0, 'Requires f(0) = 1'
    fprime0 = torch.autograd.grad(f(zero), zero)[0].item()

    b = 5
    fraction = E_gap * E_gap / (b + E_gap * E_gap)
    alpha = 0.5 + ((5 + np.sqrt(5)) / 6 - 0.5) * fraction
    beta = 0.5 + ((5 - np.sqrt(5)) / 6 - 0.5) * fraction

    vol = torch.abs(torch.linalg.det(box_vecs))
    N_elec = round((torch.mean(den) * vol).detach().item())
    n0 = N_elec / vol

    eta, G_inv = G_inv_gap(box_vecs, den, E_gap)
    g_tilde = torch.fft.rfftn(den.pow(beta))

    # set origin of kernel to zero
    Kg_tilde = torch.zeros(eta.shape, dtype=g_tilde.dtype, device=den.device)
    Kg_tilde[eta != 0] = (1 / G_inv[eta != 0] - 3 * eta[eta != 0].pow(2) - 1) * g_tilde[eta != 0]

    conv = 5 / (9 * alpha * beta * n0.pow(alpha + beta - 5 / 3)) * torch.fft.irfftn(Kg_tilde, den.shape)
    T_NL = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * torch.mean((den.pow(alpha)) * conv) * vol
    vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)

    return vW + TF * f(T_NL / fprime0 / TF)

# ------------------------------ Huang-Carter functionals-----------------------------------


class HuangCarter(KineticFunctional):
    """ Huang-Carter (HC) functional

    The Huang-Carter (HC) functional [`Phys. Rev. B 81, 045206 <https://link.aps.org/doi/10.1103/PhysRevB.81.045206>`_]
    is a non-local kinetic functional that has a single-point density dependent kernel.

    """

    def __init__(self, init_args):
        r"""
        Args:
          init_args (tuple) : :math:`(\lambda,~\beta,~\kappa)` where each parameter is a float.
                              :math:`\lambda,~\beta` are key parameters of the HC functional,
                              while :math:`\kappa` is a parameter for the spline-based field dependent
                              convolution. Recommended values for the parameters are
                              :math:`(\lambda,~\beta) = (0.01177,~0.7143)`. Note that a geometric progression
                              based spline is used so :math:`\kappa > 1`.  It is recommended to start with
                              :math:`\kappa = 1.2` and reduce :math:`\kappa` until the energy is converged.
        """
        super().__init__()
        lamb, beta, kappa = init_args
        self.lamb = torch.nn.Parameter(torch.tensor([lamb], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.kappa = kappa
        self.mode = 'geometric'
        self.initialize()
        self.generate_kernel()

    def generate_kernel(self, eta_max=50, N_eta=10000):
        r"""
        Generates the Huang-Carter kernel, :math:`\omega(\eta)`, by solving an initial value
        problem with Xitorch. The associated ordinary differential eqaution (ODE) is obtained by
        imposing the Lindhard response of a homogeneous electron gas on the Huang-Carter functional.

        Args:
          eta_max (float) : Upper bound of :math:`\eta` to solve the ODE from
          N_eta (int)     : Number of data points in :math:`[0, \eta_\text{max}]`
        """
        def lindhard(eta):
            if eta == 0:
                return torch.tensor([1.0], dtype=torch.double, device=self.device)
            elif eta == 1:
                return torch.tensor([2.0], dtype=torch.double, device=self.device)
            else:
                return 1 / G_inv_lind_analytical(eta)

        def w_prime(eta, w, beta):
            aux = (5 / 3) * (lindhard(eta) - 3 * eta * eta - 1) - (5 - 3 * beta) * beta * w
            return - aux / beta / eta

        wInf = -(8 / 3) / ((5 - 3 * self.beta) * self.beta)
        etas = torch.linspace(0, eta_max, N_eta, dtype=torch.double, device=self.device)
        w = solve_ivp(w_prime, ts=torch.flip(etas[1:], (0,)), y0=wInf, params=[self.beta])
        w = torch.cat((torch.zeros(1, dtype=torch.double, device=self.device), torch.flip(w[:, 0], (0,))))
        self.kernel = torch.cat([etas.unsqueeze(0), w.unsqueeze(0)])

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: HC kinetic energy
        """
        kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
        # ξ(r) = 2 k_F(r) [1 + λs²(r)];  *s is not the reduced gradient
        s2 = grad_dot_grad(kx, ky, kz, den) / (den.pow(8 / 3) + 1e-30)
        k_F = (3 * np.pi * np.pi * den).pow(1 / 3)
        xis = 2 * k_F * (1 + self.lamb * s2)

        if self.debug:
            print('ξ_min: {:.6g}, ξ_max: {:.6g}'.format(xis.min().item(), xis.max().item()))

        # Huang-Carter kernel ω(η) = ω(q/ξ)
        eta_1D, w_1D = self.kernel

        def w_tilde(q, xi_sparse):
            eta = q.unsqueeze(3).expand((-1, -1, -1, len(xi_sparse))) / xi_sparse
            if self.debug:
                print('η_min: {:.6g}, η_max: {:.6g}'.format(eta.min().item(), eta.max().item()))
            return interpolate(eta_1D, w_1D, torch.minimum(eta, eta_1D[-1]))

        # g(r') = [n(r')]^β
        g = den.pow(self.beta)
        # Computes K(r) = ∫d³r' ω(|r-r'|,ξ(r)) g(r')
        q = torch.zeros(k2.shape, dtype=torch.double, device=k2.device)
        q[k2 != 0] = torch.sqrt(k2[k2 != 0])
        K = field_dependent_convolution(q, w_tilde, g, xis, kappa=self.kappa, mode=self.mode)

        C_HC = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * 8 * (3 * np.pi * np.pi)
        T_NL = C_HC * torch.mean(den.pow(8 / 3 - self.beta) * K / xis.pow(3)) * torch.abs(torch.linalg.det(box_vecs))
        vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
        return vW + TF + T_NL


class RevisedHuangCarter(KineticFunctional):
    """ revised Huang-Carter (revHC) functional

    The revised Huang-Carter (revHC) functional [`Phys. Rev. B 104, 045118
    <https://link.aps.org/doi/10.1103/PhysRevB.104.045118>`_] is a non-local kinetic functional
    that has a single-point density dependent kernel. The revision of the Huang-Carter functional
    is to handle cases where the reduced gradient term is large due to small densities,
    such as for surface calculations.
    """

    def __init__(self, init_args):
        r"""
        Args:
          init_args (tuple) : :math:`(a,~b,~\beta,~\kappa)` where each parameter is a float.
                              :math:`a,~b,~\beta` are key parameters of the revHC functional,
                              while :math:`\kappa` is a parameter for the spline-based field dependent
                              convolution. Recommended values for the parameters are
                              :math:`(a,~b,~\beta) = (0.45,~0.10,~2/3)`. Note that a geometric progression
                              based spline is used so :math:`\kappa > 1`. It is recommended to start with
                              :math:`\kappa = 1.15` and reduce :math:`\kappa` until the energy is converged.
        """
        super().__init__()
        a, b, beta, kappa = init_args
        self.a = torch.nn.Parameter(torch.tensor([a], dtype=torch.double, device=self.device))
        self.b = torch.nn.Parameter(torch.tensor([b], dtype=torch.double, device=self.device))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double, device=self.device))
        self.kappa = kappa
        self.mode = 'geometric'
        self.initialize()
        self.generate_kernel()

    def generate_kernel(self, eta_max=50, N_eta=10000):
        r"""
        Generates the Huang-Carter kernel, :math:`\omega(\eta)`, by solving an initial value
        problem with Xitorch. The associated ordinary differential eqaution (ODE) is obtained by
        imposing the Lindhard response of a homogeneous electron gas on the Huang-Carter functional.

        Args:
          eta_max (float) : Upper bound of :math:`\eta` to solve the ODE from
          N_eta (int)     : Number of data points in :math:`[0, \eta_\text{max}]`
        """
        def lindhard(eta):
            if eta == 0:
                return torch.tensor([1.0], dtype=torch.double, device=self.device)
            elif eta == 1:
                return torch.tensor([2.0], dtype=torch.double, device=self.device)
            else:
                return 1 / G_inv_lind_analytical(eta)

        def w_prime(eta, w, beta):
            aux = (5 / 3) * (lindhard(eta) - 3 * eta * eta - 1) - (5 - 3 * beta) * beta * w
            return - aux / beta / eta

        wInf = -(8 / 3) / ((5 - 3 * self.beta) * self.beta)
        etas = torch.linspace(0, eta_max, N_eta, dtype=torch.double, device=self.device)
        w = solve_ivp(w_prime, ts=torch.flip(etas[1:], (0,)), y0=wInf, params=[self.beta])
        w = torch.cat((torch.zeros(1, dtype=torch.double, device=self.device), torch.flip(w[:, 0], (0,))))
        self.kernel = torch.cat([etas.unsqueeze(0), w.unsqueeze(0)])

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: revHC kinetic energy
        """
        kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
        # ξ(r) = k_F(r) F[s(r)], where F[s] = 1 + as²/(1+bs²)
        s2 = reduced_gradient_squared(kx, ky, kz, den)
        F = 1 + self.a * s2 / (1 + self.b * s2)
        k_F = (3 * np.pi * np.pi * den).pow(1 / 3)
        xis = 2 * k_F * F

        eta_1D, w_1D = self.kernel

        # Huang-Carter kernel ω(η) = ω(q/ξ)
        def w_tilde(q, xi_sparse):
            eta = q.unsqueeze(3).expand((-1, -1, -1, len(xi_sparse))) / xi_sparse
            return interpolate(eta_1D, w_1D, torch.minimum(eta, eta_1D[-1]))

        # g(r') = [n(r')]^β
        g = den.pow(self.beta)

        # Computes K(r) = ∫d³r' ω(|r-r'|,ξ(r)) g(r')
        q = torch.zeros(k2.shape, dtype=torch.double, device=k2.device)
        q[k2 != 0] = torch.sqrt(k2[k2 != 0])
        K = field_dependent_convolution(q, w_tilde, g, xis, kappa=self.kappa, mode=self.mode)

        C_HC = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * 8 * (3 * np.pi * np.pi)
        T_NL = C_HC * torch.mean(den.pow(8 / 3 - self.beta) * K / xis.pow(3)) * torch.abs(torch.linalg.det(box_vecs))
        vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
        return vW + TF + T_NL

# ----------------------------- Mi-Genova-Pavanello functional------------------------------


class MiGenovaPavanello(KineticFunctional):
    """ Mi-Genova-Pavanello (MGP) functional

    The Mi-Genova-Pavanello (MGP) functional [`J. Chem. Phys. 148, 184107 <https://doi.org/10.1063/1.5023926>`_]
    is a non-local kinetic functional based on line integrals.
    """

    def __init__(self, init_args):
        """
        Args:
          init_args (tuple) : :math:`(a,~b)` where each parameter is a float. They are
                              key parameters of the Mi-Genova-Pavanello functional.
        """
        super().__init__()
        a, b = init_args
        self.a = torch.nn.Parameter(torch.tensor([a], dtype=torch.double, device=self.device))
        self.b = torch.nn.Parameter(torch.tensor([b], dtype=torch.double, device=self.device))
        self.initialize()
        self.kernel = None

    def generate_kernel(self, eta_max=60, N_eta=2000, N_int=10000):
        r"""
        Generates the integral part of the MGP kernel in one-dimension for later interpolation.
        This process involves perform numerical integration.

        Args:
          eta_max (float) : :math:`\eta_\text{max}` is the upper bound for which the kernel
                            :math:`K(\eta)` is generated up to
          N_eta (int)     : Number of data points in :math:`[0, \eta_\text{max}]`
          N_int (int)     : Number of integration points for numerical integration of the kernel
        """
        ts = torch.linspace(1e-4, 1, N_int, dtype=torch.double, device=self.device)
        dt = ts[1] - ts[0]
        etas = torch.linspace(0, eta_max, N_eta, dtype=torch.double, device=self.device).unsqueeze(1).expand(-1, N_int)
        etas = etas / ts.pow(1 / 3)
        G_NL = 1 / G_inv_lind(etas) - 3 * etas.pow(2) - 1
        # numerical integration of part of the kernel
        w = 0.2 * (3 * np.pi**2)**(2 / 3) * torch.sum(G_NL / ts.pow(1 / 6), axis=1) * dt
        eta = etas[:, -1]
        self.kernel = torch.cat([eta.unsqueeze(0), w.unsqueeze(0)])

    def forward(self, box_vecs, den):
        """
        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          den      (torch.Tensor) : Electron density

        Returns:
          torch.Tensor: MGP kinetic energy
        """
        vol = torch.abs(torch.linalg.det(box_vecs))
        kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
        N_elec = round((torch.mean(den) * vol).detach().item())
        n0 = N_elec / vol

        k_F = (3 * np.pi * np.pi * n0).pow(1 / 3)
        eta = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
        eta[k2 != 0] = torch.sqrt(k2[k2 != 0]) / (2 * k_F)
        eta_max = torch.max(eta).item()

        # Mi-Genova-Pavanello kernel
        w_corr = torch.empty(k2.shape, dtype=torch.double, device=den.device)
        w_corr[k2 != 0] = torch.erf(eta[k2 != 0] * 2 * k_F).pow(2) * (4 * np.pi * self.a / k2[k2 != 0]) \
                          * torch.exp(- self.b * k2[k2 != 0])
        w_corr[k2 == 0] = 16 * self.a  # avoid dividing by zero [doesn't matter since ω(q=0) = 0 is enforced later]

        if self.kernel is None:
            self.generate_kernel(1.2 * eta_max)
        elif self.kernel[0, -1] < eta_max:
            self.generate_kernel(1.2 * eta_max)
        eta_1D, w_1D = self.kernel

        MGP_kernel = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
        # MGP_kernel[k2==0] = 0 "to avoid spurious dependencies from the parameter a"
        unique_etas, ids = torch.unique(eta, return_inverse=True)
        MGP_kernel[k2 != 0] = interpolate(eta_1D, w_1D, torch.minimum(unique_etas, eta_1D[-1]))[ids][k2 != 0] \
                             + (3 / 5) * w_corr[k2 != 0]

        conv = torch.fft.irfftn(MGP_kernel * torch.fft.rfftn(den.pow(5 / 6)), den.shape)
        T_NL = torch.mean(den.pow(5 / 6) * conv) * vol
        vW, TF = Weizsaecker(box_vecs, den), ThomasFermi(box_vecs, den)
        return vW + TF + T_NL


# --------------------------------- Xu-Wang-Ma functional-----------------------------------

def XuWangMa(box_vecs, den, kappa=0):
    r""" Xu-Wang-Ma (XWM) functional

    The Xu-Wang-Ma (XWM)  functional [`Phys. Rev. B 100, 205132 <https://doi.org/10.1103/PhysRevB.100.205132>`_]
    is a non-local kinetic functional based on line integrals, with a density-dependent kernel.
    In practice however, a Taylor-expansion is used to avoid the unattractive
    computational scaling resulting from the density-dependent kernel.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density
      kappa    (float)        : Adjustable parameter (default :math:`\kappa=0`)

    Returns:
      torch.Tensor: XWM kinetic energy
    """
    vol = torch.abs(torch.linalg.det(box_vecs))
    N_elec = round((torch.mean(den) * vol).detach().item())
    n0 = N_elec / vol

    eta, G_inv_lind = G_inv_lindhard(box_vecs, den)

    kernel0 = 18 / (6 * kappa + 5)**2 * np.pi**2 / (3 * np.pi**2)**(1 / 3) \
               * (1 / G_inv_lind - 3 * eta.pow(2) - 1) / n0.pow(2 * kappa)
    conv = torch.fft.irfftn(kernel0 * torch.fft.rfftn(den.pow(kappa + 5 / 6)), den.shape)
    T_NL0 = torch.mean(den.pow(kappa + 5 / 6) * conv) * vol

    G_inv_der = torch.zeros(eta.shape, dtype=torch.double, device=eta.device)
    G_inv_der[eta != 0] = 0.5 - (0.25 * (eta[eta != 0] + 1 / eta[eta != 0])
                                 * torch.log(torch.abs((1 + eta[eta != 0]) / (1 - eta[eta != 0]))))
    kernel1 = np.pi**2 / (3 * np.pi**2)**(1 / 3) * 1 / (6 * n0) * \
             (G_inv_der * G_inv_lind.pow(-2) + 6 * eta.pow(2)) / n0.pow(2 * kappa)

    kernel1a = 1 / (kappa + 5 / 6) / (kappa + 11 / 6) * kernel1
    kernel1b = n0 / (kappa + 5 / 6)**2 * kernel1

    conva = torch.fft.irfftn(kernel1a * torch.fft.rfftn(den.pow(kappa + 11 / 6)), den.shape)
    T_NL1a = torch.mean(den.pow(kappa + 5 / 6) * conva) * vol

    convb = torch.fft.irfftn(kernel1b * torch.fft.rfftn(den.pow(kappa + 5 / 6)), den.shape)
    T_NL1b = torch.mean(den.pow(kappa + 5 / 6) * convb) * vol

    return Weizsaecker(box_vecs, den) + ThomasFermi(box_vecs, den) + T_NL0 + T_NL1a - T_NL1b


##############################################################################################
#                         Exchange-Correlation (XC) Functionals                              #
##############################################################################################

# --------------------------------------------------------------------------------------------
#                                Local Density Approximation (LDA)
# --------------------------------------------------------------------------------------------


def lda_exchange(box_vecs, den):
    E_x = -(3 / 4) * (3 / np.pi)**(1 / 3) * torch.mean(den.pow(4 / 3)) * torch.abs(torch.linalg.det(box_vecs))
    return E_x


def perdew_zunger_correlation(box_vecs, den):
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    A, B, C, D = 0.0311, -0.048, 0.002, -0.0116
    rs = (3 / 4 / np.pi / den).pow(1 / 3)
    eps_c = torch.where(rs < 1, A * torch.log(rs) + B + C * rs * torch.log(rs) + D * rs,
                        gamma / (1 + beta1 * torch.sqrt(rs) + beta2 * rs))
    return torch.mean(eps_c * den) * torch.abs(torch.linalg.det(box_vecs))


def perdew_wang_correlation(box_vecs, den):
    A, alpha = 0.0310907, 0.2137
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs = (3 / 4 / np.pi / den).pow(1 / 3)
    eps_c = -2 * A * (1 + alpha * rs) * torch.log(1 + 1 / (2 * A * (b1 * rs.pow(0.5)
                                                  + b2 * rs + b3 * rs.pow(1.5) + b4 * rs.pow(2))))
    return torch.mean(eps_c * den) * torch.abs(torch.linalg.det(box_vecs))


def chachiyo_correlation(box_vecs, den):
    a, b = (np.log(2) - 1) / 2 / np.pi / np.pi, 20.4562557
    rs = (3 / 4 / np.pi / den).pow(1 / 3)
    eps_c = a * torch.log(1 + b / rs + b / rs.pow(2))
    return torch.mean(eps_c * den) * torch.abs(torch.linalg.det(box_vecs))


def PerdewZunger(box_vecs, den):
    """ Perdew-Zunger (PZ) functional

    The Perdew-Zunger (PZ) functional [`Phys. Rev. B 23, 5048 <https://doi.org/10.1103/PhysRevB.23.5048>`_]
    is an LDA exchange-correlation functional based on the Perdew-Zunger parameterization of Ceperley and Alder's
    free electron gas quantum Monte Carlo simulations.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: PZ XC energy
    """
    return lda_exchange(box_vecs, den) + perdew_zunger_correlation(box_vecs, den)


def PerdewWang(box_vecs, den):
    """ Perdew-Wang (PW) functional

    The Perdew-Wang (PW) functional [`Phys. Rev. B 45, 13244 <https://doi.org/10.1103/PhysRevB.45.13244>`_]
    is an LDA exchange-correlation functional based on the Perdew-Wang parameterization of Ceperley and Alder's
    free electron gas quantum Monte Carlo simulations.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: PW XC energy
    """
    return lda_exchange(box_vecs, den) + perdew_wang_correlation(box_vecs, den)


def Chachiyo(box_vecs, den):
    """ Chachiyo functional

    The Chachiyo functional [`J. Chem. Phys. 145, 021101 <https://aip.scitation.org/doi/10.1063/1.4958669>`_]
    is an LDA exchange-correlation functional derived non-empirically based on second-order Moller-Plesset
    perturbation theory.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: Chachiyo XC energy
    """
    return lda_exchange(box_vecs, den) + chachiyo_correlation(box_vecs, den)

# --------------------------------------------------------------------------------------------
#                              Generalized Gradient Approximation (GGA)
# --------------------------------------------------------------------------------------------

# ------------------------ Perdew-Burke-Ernzerhof (PBE) functional----------------------------


def pbe_exchange(box_vecs, den):
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    local_exchange_density = -(3 / 4) * (3 / np.pi)**(1 / 3) * den.pow(4 / 3)
    s2 = reduced_gradient_squared(kx, ky, kz, den)
    kappa, mu = 0.804, 0.066725 * np.pi * np.pi / 3  # or 0.21951
    Fx = 1 + kappa - kappa / (1 + mu / kappa * s2)
    return torch.mean(Fx * local_exchange_density) * torch.abs(torch.linalg.det(box_vecs))


def pbe_correlation(box_vecs, den):
    kx, ky, kz, k2 = wavevecs(box_vecs, den.shape)
    A1, alpha = 0.0310907, 0.2137
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs = (3 / 4 / np.pi / den).pow(1 / 3)
    eps_c = - 2 * A1 * (1 + alpha * rs) * torch.log(1 + 1 / (2 * A1 * (b1 * rs.pow(0.5)
                                                    + b2 * rs + b3 * rs.pow(1.5) + b4 * rs.pow(2))))
    beta, gamma = 0.066725, (1 - np.log(2)) / np.pi / np.pi
    A = beta / gamma / (torch.exp(-eps_c / gamma) - 1 + 1e-30)  # 1e-30 prevents dividing by zero
    t2 = (1 / 16) * (np.pi / 3)**(1 / 3) * grad_dot_grad(kx, ky, kz, den) / (den.pow(7 / 3) + 1e-30)
    At2 = A * t2
    H = gamma * torch.log(1 + beta / gamma * t2 * ((1 + At2) / (1 + At2 + At2.pow(2))))
    return torch.mean((eps_c + H) * den) * torch.abs(torch.linalg.det(box_vecs))


def PerdewBurkeErnzerhof(box_vecs, den):
    """ Perdew-Burke-Ernzerhof (PBE) functional

    The Perdew-Burke-Ernzerhof (PBE) functional [`Phys. Rev. Lett. 77, 3865
    <https://link.aps.org/doi/10.1103/PhysRevLett.77.3865>`_]
    is a popular non-empirical GGA exchange-correlation functional.

    Args:
      box_vecs (torch.Tensor) : Lattice vectors
      den      (torch.Tensor) : Electron density

    Returns:
      torch.Tensor: PBE XC energy
    """
    return pbe_exchange(box_vecs, den) + pbe_correlation(box_vecs, den)
