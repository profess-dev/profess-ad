import numpy as np
import torch
from math import pi

from professad.functional_tools import wavevectors, grad_dot_grad, reduced_gradient, \
    reduced_gradient_squared, laplacian, reduced_laplacian
from professad.functionals import G_inv_lindhard, Hartree, ThomasFermi, non_local_KEF, \
    LocalExchange, _perdew_zunger_correlation, _perdew_wang_correlation, _chachiyo_correlation


# ----------------------  Utility Function  --------------------

def grad_i(ki: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
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


# -------  Analytical Functional Derivatives for Testing  -------

def hartree_potential(box_vecs, den):
    k2 = wavevectors(box_vecs, den.shape).square().sum(-1)
    coloumb_ft = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
    coloumb_ft[k2 != 0] = 4 * pi / k2[k2 != 0]
    return torch.fft.irfftn(torch.fft.rfftn(den) * coloumb_ft, den.shape)


def TF_kp(box_vecs, den):
    return 0.5 * (3 * pi * pi)**(2 / 3) * den**(2 / 3)


def vW_kp(box_vecs, den):
    k2 = wavevectors(box_vecs, den.shape).square().sum(-1)
    sqrt_den = torch.sqrt(den)
    return -0.5 * laplacian(k2, sqrt_den).div(sqrt_den)


def non_local_KEFD(box_vecs, den, alpha=5 / 6, beta=5 / 6):
    # likely assumes n0 is constant instead of depending on den
    N_elec = (torch.mean(den) * torch.abs(torch.linalg.det(box_vecs))).item()
    n0 = N_elec / torch.abs(torch.linalg.det(box_vecs))
    eta, G_inv = G_inv_lindhard(den, wavevectors(box_vecs, den.shape))
    kernel = 5 / (9 * alpha * beta * n0.pow(alpha + beta - 5 / 3)) * (1 / G_inv - 3 * eta * eta - 1)
    conv_a = torch.fft.irfftn(kernel * torch.fft.rfftn(den.pow(alpha)), den.shape)
    conv_b = torch.fft.irfftn(kernel * torch.fft.rfftn(den.pow(beta)), den.shape)
    LR_kp = 0.3 * (3 * pi * pi)**(2 / 3) * (alpha * den.pow(alpha - 1) * conv_b
                                                  + beta * den.pow(beta - 1) * conv_a)
    return TF_kp(box_vecs, den) + vW_kp(box_vecs, den) + LR_kp


def LKT_kp(box_vecs, den):
    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    s = reduced_gradient(kxyz, den)
    abs_grad_n = torch.sqrt(grad_dot_grad(kxyz, den))
    dsdn = 0.5 * (3 * pi * pi)**(-1 / 3) * (-4 / 3) * abs_grad_n * den.pow(-7 / 3)
    dsdgradn = 0.5 * (3 * pi * pi)**(-1 / 3) * den.pow(-4 / 3)
    dndx = grad_i(kx, den)
    dndy = grad_i(ky, den)
    dndz = grad_i(kz, den)

    F_theta = torch.cosh(1.3 * s).reciprocal()
    dFds = - 1.3 * torch.tanh(1.3 * s) * F_theta

    term1 = vW_kp(box_vecs, den) + F_theta * TF_kp(box_vecs, den)
    term2 = dFds * dsdn * ThomasFermi(den, 0)
    aux_x = dFds * dsdgradn * ThomasFermi(den, 0) * dndx / abs_grad_n
    aux_y = dFds * dsdgradn * ThomasFermi(den, 0) * dndy / abs_grad_n
    aux_z = dFds * dsdgradn * ThomasFermi(den, 0) * dndz / abs_grad_n
    term3 = - grad_i(kx, aux_x) - grad_i(ky, aux_y) - grad_i(kz, aux_z)
    return term1 + term2 + term3


def PG1_kp(box_vecs, den):
    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]

    s = reduced_gradient(kxyz, den)
    abs_grad_n = torch.sqrt(grad_dot_grad(kxyz, den))
    dsdn = 0.5 * (3 * pi * pi)**(-1 / 3) * (-4 / 3) * abs_grad_n * den.pow(-7 / 3)
    dsdgradn = 0.5 * (3 * pi * pi)**(-1 / 3) * den.pow(-4 / 3)
    dndx = grad_i(kx, den)
    dndy = grad_i(ky, den)
    dndz = grad_i(kz, den)

    F_theta = torch.exp(-s * s)
    dFds = - 2 * s * F_theta

    term1 = vW_kp(box_vecs, den) + F_theta * TF_kp(box_vecs, den)
    term2 = dFds * dsdn * ThomasFermi(den, 0)
    aux = dFds * dsdgradn * ThomasFermi(den, 0) / abs_grad_n
    aux_x = aux * dndx
    aux_y = aux * dndy
    aux_z = aux * dndz
    term3 = - grad_i(kx, aux_x) - grad_i(ky, aux_y) - grad_i(kz, aux_z)
    return term1 + term2 + term3


def PGSL_kp(box_vecs, den):
    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    k2 = kxyz.square().sum(-1)
    s = reduced_gradient(kxyz, den)
    q = reduced_laplacian(k2, den)
    abs_grad_n = torch.sqrt(grad_dot_grad(kxyz, den))
    dsdn = 0.5 * (3 * pi * pi)**(-1 / 3) * (-4 / 3) * abs_grad_n * den.pow(-7 / 3)
    dsdgradn = 0.5 * (3 * pi * pi)**(-1 / 3) * den.pow(-4 / 3)
    dqdn = 0.25 * (3 * pi * pi)**(-2 / 3) * laplacian(k2, den) * (-5 / 3) * den.pow(-8 / 3)
    dqdlapn = 0.25 * (3 * pi * pi)**(-2 / 3) * den.pow(-5 / 3)
    dndx = grad_i(kx, den)
    dndy = grad_i(ky, den)
    dndz = grad_i(kz, den)

    F_theta = torch.exp(-40 / 27 * s * s) + 0.25 * q * q
    dFds = - 2 * 40 / 27 * s * torch.exp(-40 / 27 * s * s)
    dFdq = 0.5 * q

    term1 = vW_kp(box_vecs, den) + F_theta * TF_kp(box_vecs, den)
    term2 = dFds * dsdn * ThomasFermi(den, 0)
    aux = dFds * dsdgradn * ThomasFermi(den, 0) / abs_grad_n
    aux_x = aux * dndx
    aux_y = aux * dndy
    aux_z = aux * dndz
    term3 = - grad_i(kx, aux_x) - grad_i(ky, aux_y) - grad_i(kz, aux_z)
    term_s = term1 + term2 + term3

    term4 = dFdq * dqdn * ThomasFermi(den, 0)
    term5 = laplacian(k2, dFdq * dqdlapn * ThomasFermi(den, 0))
    term_q = term4 + term5
    return term_s + term_q


def lda_exchange_potential(box_vecs, den):
    return -(3 / 4) * (3 / pi)**(1 / 3) * (4 / 3) * den.pow(1 / 3)


def perdew_zunger_correlation_potential(box_vecs, den):
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    A, B, C, D = 0.0311, -0.048, 0.002, -0.0116
    rs = (3 / 4 / pi / den).pow(1 / 3)
    return torch.where(rs < 1, torch.log(rs) * (A + 2 / 3 * C * rs) + (B - A / 3) + rs / 3 * (2 * D - C),
                       gamma * (1 + 7 / 6 * beta1 * torch.sqrt(rs) + 4 / 3 * beta2 * rs)
                       / (1 + beta1 * torch.sqrt(rs) + beta2 * rs).square())


def perdew_wang_correlation_potential(box_vecs, den):
    A, alpha = 0.0310907, 0.2137
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs = (3 / 4 / pi / den).pow(1 / 3)
    zeta = 2 * A * (b1 * rs.pow(0.5) + b2 * rs + b3 * rs.pow(1.5) + b4 * rs.square())
    eps = -2 * A * (1 + alpha * rs) * torch.log(1 + 1 / zeta)
    deps_dn = -rs / 3 / den * (-2 * A * alpha * torch.log(1 + 1 / zeta) + (2 * A * A * (1 + alpha * rs)
                               * (b1 * rs.pow(-0.5) + 2 * b2 + 3 * b3 * rs.pow(0.5) + 4 * b4 * rs)
                               / (zeta * (zeta + 1))))
    return deps_dn * den + eps


def chachiyo_correlation_potential(box_vecs, den):
    a, b = (np.log(2) - 1) / 2 / pi / pi, 20.4562557
    rs = (3 / 4 / pi / den).pow(1 / 3)
    eps = a * torch.log(1 + b / rs + b / rs.square())
    deps_drs = a / (1 + b / rs + b / rs.square()) * (-b / rs.square() - 2 * b / rs.pow(3))
    drs_dn = (3 / 4 / pi)**(1 / 3) * (-1 / 3) * den.pow(-4 / 3)
    return deps_drs * drs_dn * den + eps


def pbe_exchange_potential(box_vecs, den):
    eps = -(3 / 4) * (3 / pi)**(1 / 3) * den.pow(1 / 3)
    deps_dn = - (1 / 4) * (3 / pi)**(1 / 3) * den.pow(-2 / 3)

    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    s2 = reduced_gradient_squared(kxyz, den)
    kappa, mu = 0.804, 0.066725 * pi * pi / 3  # or 0.21951
    Fx = 1 + kappa - kappa / (1 + mu / kappa * s2)
    ds2_dgn2 = 0.25 * (3 * pi * pi)**(-2 / 3) * den.pow(-8 / 3)
    ds2_dn = -(8 / 3) * s2 / den
    dFx_ds2 = mu / (1 + mu / kappa * s2).square()

    df_dn = Fx * (deps_dn * den + eps) + dFx_ds2 * ds2_dn * eps * den
    df_dgn2 = dFx_ds2 * ds2_dgn2 * eps * den
    dndx, dndy, dndz = grad_i(kx, den), grad_i(ky, den), grad_i(kz, den)
    aux = - 2 * (grad_i(kx, df_dgn2 * dndx) + grad_i(ky, df_dgn2 * dndy) + grad_i(kz, df_dgn2 * dndz))
    return df_dn + aux


def pbe_correlation_potential(box_vecs, den):
    A1, alpha = 0.0310907, 0.2137
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs = (3 / 4 / pi / den).pow(1 / 3)
    zeta = 2 * A1 * (b1 * rs.pow(0.5) + b2 * rs + b3 * rs.pow(1.5) + b4 * rs.square())
    eps_c = -2 * A1 * (1 + alpha * rs) * torch.log(1 + 1 / zeta)
    deps_dn = -rs / 3 / den * (-2 * A1 * alpha * torch.log(1 + 1 / zeta) + (2 * A1 * A1 * (1 + alpha * rs)
                               * (b1 * rs.pow(-0.5) + 2 * b2 + 3 * b3 * rs.pow(0.5) + 4 * b4 * rs)
                               / (zeta * (zeta + 1))))

    beta, gamma = 0.066725, (1 - np.log(2)) / pi / pi
    A = beta / gamma / (torch.exp(-eps_c / gamma) - 1)
    dAdn = 1 / beta * A.square() * torch.exp(-eps_c / gamma) * deps_dn

    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    t2 = (1 / 16) * (pi / 3)**(1 / 3) * grad_dot_grad(kxyz, den) * den.pow(-7 / 3)
    dt2dn = -7 / 3 * t2 / den
    dt2dgn2 = (1 / 16) * (pi / 3)**(1 / 3) * den.pow(-7 / 3)

    At2 = A * t2
    numer = 1 + At2
    denom = 1 + At2 + At2.square()
    H = gamma * torch.log(1 + beta / gamma * t2 * (numer / denom))

    numer2 = 1 + 2 * At2
    dHdn = beta * torch.exp(-H / gamma) * ((dt2dn * numer2 + dAdn * t2.square()) / denom
                                            - t2 * numer / denom.square() * (dt2dn * A * numer2 + dAdn * t2 * numer2))
    dH_dgn2 = beta * torch.exp(-H / gamma) * (dt2dgn2 * numer2 / denom
                                              - At2 * numer / denom.square() * dt2dgn2 * numer2)
    df_dn = eps_c + H + den * (deps_dn + dHdn)
    df_dgn2 = den * dH_dgn2
    dndx, dndy, dndz = grad_i(kx, den), grad_i(ky, den), grad_i(kz, den)
    aux = - 2 * (grad_i(kx, df_dgn2 * dndx) + grad_i(ky, df_dgn2 * dndy) + grad_i(kz, df_dgn2 * dndz))
    return df_dn + aux


# -------  Analytical Stresses for Testing  -------

def hartree_stress(box_vecs, den):
    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    k2 = kxyz.square().sum(-1)
    factor = torch.full(k2.shape, 8 * pi, dtype=torch.double, device=den.device)
    if den.shape[2] % 2 == 0:
        factor[:, :, k2.shape[2] - 1] = 4 * pi
    factor[:, :, 0] = 4 * pi
    factor[0, 0, 0] = 0.0

    den_ft = torch.fft.rfftn(den, norm='forward')
    aux = (den_ft.real.square() + den_ft.imag.square()) / (k2.square() + 1e-30)

    term1 = torch.empty((3, 3), dtype=torch.double, device=den.device)
    term1[0, 0] = torch.sum(factor * aux * kx * kx)
    term1[1, 1] = torch.sum(factor * aux * ky * ky)
    term1[2, 2] = torch.sum(factor * aux * kz * kz)
    term1[0, 1] = torch.sum(factor * aux * kx * ky)
    term1[0, 2] = torch.sum(factor * aux * kx * kz)
    term1[1, 2] = torch.sum(factor * aux * ky * kz)
    term1[1, 0] = term1[0, 1]
    term1[2, 0] = term1[0, 2]
    term1[2, 1] = term1[1, 2]

    term2 = torch.mean(Hartree(den, kxyz)).neg() * torch.eye(3, dtype=den.dtype, device=den.device)

    return term1 + term2


def TF_stress(box_vecs, den):
    I = torch.eye(3, dtype=den.dtype, device=den.device)
    return -2 / 3 * torch.mean(ThomasFermi(den, I)) * I


def vW_stress(box_vecs, den):
    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    dndx, dndy, dndz = grad_i(kx, den), grad_i(ky, den), grad_i(kz, den)
    aux = torch.empty((3, 3), dtype=torch.double, device=den.device)
    aux[0, 0] = torch.mean(dndx * dndx / den)
    aux[1, 1] = torch.mean(dndy * dndy / den)
    aux[2, 2] = torch.mean(dndz * dndz / den)
    aux[0, 1] = torch.mean(dndx * dndy / den)
    aux[0, 2] = torch.mean(dndx * dndz / den)
    aux[1, 2] = torch.mean(dndy * dndz / den)
    aux[1, 0] = aux[0, 1]
    aux[2, 0] = aux[0, 2]
    aux[2, 1] = aux[1, 2]
    return -aux / 4


def non_local_KEF_stress(box_vecs, den, alpha=5 / 6, beta=5 / 6):
    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    k2 = kxyz.square().sum(-1)

    T_lr = torch.mean(non_local_KEF(den, kxyz, alpha, beta))
    term1 = - 2 / 3 * T_lr * torch.eye(3, dtype=torch.double, device=den.device)

    n0 = torch.mean(den)
    k_F = (3 * pi * pi * n0).pow(1 / 3)

    prefactor = 0.5 * pi * pi / alpha / beta / n0.pow(alpha + beta - 2) / k_F
    filter = torch.ones(k2.shape, dtype=torch.double, device=den.device)
    if den.shape[2] % 2 == 0:
        filter[:, :, k2.shape[2] - 1] = 0.5
    filter[:, :, 0] = 0.5
    filter[0, 0, 0] = 0.0

    delta_n_a = torch.fft.rfftn(den.pow(alpha) - n0.pow(alpha), norm='forward')
    delta_n_b = torch.conj(torch.fft.rfftn(den.pow(beta) - n0.pow(beta), norm='forward'))
    aux1 = (delta_n_a * delta_n_b + torch.conj(delta_n_a) * torch.conj(delta_n_b)).real

    aux2_00 = kx * kx / (k2 + 1e-30) - 1 / 3
    aux2_11 = ky * ky / (k2 + 1e-30) - 1 / 3
    aux2_22 = kz * kz / (k2 + 1e-30) - 1 / 3
    aux2_01 = kx * ky / (k2 + 1e-30)
    aux2_02 = kx * kz / (k2 + 1e-30)
    aux2_12 = ky * kz / (k2 + 1e-30)

    eta = torch.sqrt(k2) / (2 * k_F) + 1e-30
    lind = 0.5 + ((1 - eta * eta) / (4 * eta)) * torch.log(torch.abs((1 + eta) / (1 - eta)))
    aux3 = eta / lind.square() * (0.5 / eta - 0.25 * (1 + 1 / (eta * eta))
                                  * torch.log(torch.abs((1 + eta) / (1 - eta)))) + 6 * eta * eta

    term2 = torch.empty((3, 3), dtype=torch.double, device=den.device)
    term2[0, 0] = torch.sum(filter * aux1 * aux2_00 * aux3)
    term2[1, 1] = torch.sum(filter * aux1 * aux2_11 * aux3)
    term2[2, 2] = torch.sum(filter * aux1 * aux2_22 * aux3)
    term2[0, 1] = torch.sum(filter * aux1 * aux2_01 * aux3)
    term2[0, 2] = torch.sum(filter * aux1 * aux2_02 * aux3)
    term2[1, 2] = torch.sum(filter * aux1 * aux2_12 * aux3)
    term2[1, 0] = term2[0, 1]
    term2[2, 0] = term2[0, 2]
    term2[2, 1] = term2[1, 2]

    LR_stress = (term1 + prefactor * term2)
    return TF_stress(box_vecs, den) + vW_stress(box_vecs, den) + LR_stress


def pauli_stabilized_stress(box_vecs, den, alpha=5 / 6, beta=5 / 6, f=lambda x: 1 + x, fprime=lambda x: 1):

    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    k2 = kxyz.square().sum(-1)

    T_TF = torch.mean(ThomasFermi(den, kxyz))
    T_NL = (torch.mean(non_local_KEF(den, kxyz, alpha, beta))
            / fprime(torch.zeros((1,), dtype=torch.double, device=den.device)))
    X = T_NL / T_TF

    term1 = - 2 / 3 * T_NL * torch.eye(3, dtype=torch.double, device=den.device) * fprime(X)

    n0 = torch.mean(den)
    k_F = (3 * pi * pi * n0).pow(1 / 3)

    prefactor = 0.5 * pi * pi / alpha / beta / n0.pow(alpha + beta - 2) / k_F

    filter = torch.ones(k2.shape, dtype=torch.double, device=den.device)
    if den.shape[2] % 2 == 0:
        filter[:, :, k2.shape[2] - 1] = 0.5
    filter[:, :, 0] = 0.5
    filter[0, 0, 0] = 0.0

    delta_n_a = torch.fft.rfftn(den.pow(alpha) - n0.pow(alpha), norm='forward')
    delta_n_b = torch.conj(torch.fft.rfftn(den.pow(beta) - n0.pow(beta), norm='forward'))
    aux1 = (delta_n_a * delta_n_b + torch.conj(delta_n_a) * torch.conj(delta_n_b)).real

    k2[0, 0, 0] = 10  # dummy value to avoid dividing by zero
    aux2_00 = kx * kx / k2 - 1 / 3
    aux2_11 = ky * ky / k2 - 1 / 3
    aux2_22 = kz * kz / k2 - 1 / 3
    aux2_01 = kx * ky / k2
    aux2_02 = kx * kz / k2
    aux2_12 = ky * kz / k2

    eta, lind = G_inv_lindhard(den, kxyz)
    eta[0, 0, 0] = 10  # dummy value to avoid dividing by zero

    aux3 = eta / lind.square() * (0.5 / eta - 0.25 * (1 + 1 / eta.square())
                                  * torch.log(torch.abs((1 + eta) / (1 - eta)))) + 6 * eta * eta

    term2 = torch.empty((3, 3), dtype=torch.double, device=den.device)
    term2[0, 0] = torch.sum(filter * aux1 * aux2_00 * aux3)
    term2[1, 1] = torch.sum(filter * aux1 * aux2_11 * aux3)
    term2[2, 2] = torch.sum(filter * aux1 * aux2_22 * aux3)
    term2[0, 1] = torch.sum(filter * aux1 * aux2_01 * aux3)
    term2[0, 2] = torch.sum(filter * aux1 * aux2_02 * aux3)
    term2[1, 2] = torch.sum(filter * aux1 * aux2_12 * aux3)
    term2[1, 0] = term2[0, 1]
    term2[2, 0] = term2[0, 2]
    term2[2, 1] = term2[1, 2]

    term2 = term2 * prefactor * fprime(X) / fprime(torch.zeros((1,), dtype=torch.double, device=den.device))
    return vW_stress(box_vecs, den) + TF_stress(box_vecs, den) * (f(X) - fprime(X) * X) \
           + term1 + term2


def lda_exchange_stress(box_vecs, den):
    I = torch.eye(3, dtype=den.dtype, device=den.device)
    aux = torch.mean(LocalExchange(den, I)) - torch.mean(lda_exchange_potential(box_vecs, den) * den)
    return aux * I


def perdew_zunger_correlation_stress(box_vecs, den):
    I = torch.eye(3, dtype=den.dtype, device=den.device)
    aux = (torch.mean(_perdew_zunger_correlation(den, I))
           - torch.mean(perdew_zunger_correlation_potential(box_vecs, den) * den))
    return aux * I


def perdew_wang_correlation_stress(box_vecs, den):
    I = torch.eye(3, dtype=den.dtype, device=den.device)
    aux = (torch.mean(_perdew_wang_correlation(den, I))
           - torch.mean(perdew_wang_correlation_potential(box_vecs, den) * den))
    return aux * I


def chachiyo_correlation_stress(box_vecs, den):
    I = torch.eye(3, dtype=den.dtype, device=den.device)
    aux = (torch.mean(_chachiyo_correlation(den, I))
           - torch.mean(chachiyo_correlation_potential(box_vecs, den) * den))
    return aux * I


def pbe_exchange_stress(box_vecs, den):
    eps = -(3 / 4) * (3 / pi)**(1 / 3) * den.pow(1 / 3)
    deps_dn = - (1 / 4) * (3 / pi)**(1 / 3) * den.pow(-2 / 3)

    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    s2 = reduced_gradient_squared(kxyz, den)
    kappa, mu = 0.804, 0.066725 * pi * pi / 3  # or 0.21951
    Fx = 1 + kappa - kappa / (1 + mu / kappa * s2)
    ds2_dgn2 = 0.25 * (3 * pi * pi)**(-2 / 3) * den.pow(-8 / 3)
    ds2_dn = -(8 / 3) * s2 / den
    dFx_ds2 = mu / (1 + mu / kappa * s2).square()

    df_dn = Fx * (deps_dn * den + eps) + dFx_ds2 * ds2_dn * eps * den
    df_dgn2 = dFx_ds2 * ds2_dgn2 * eps * den

    term1 = torch.mean(Fx * eps * den - den * df_dn) * torch.eye(3, dtype=torch.double, device=den.device)
    gdg = grad_dot_grad(kxyz, den)
    dndx, dndy, dndz = grad_i(kx, den), grad_i(ky, den), grad_i(kz, den)
    term2 = torch.empty((3, 3), dtype=torch.double, device=den.device)
    term2[0, 0] = - 2 * torch.mean((gdg + dndx * dndx) * df_dgn2)
    term2[1, 1] = - 2 * torch.mean((gdg + dndy * dndy) * df_dgn2)
    term2[2, 2] = - 2 * torch.mean((gdg + dndz * dndz) * df_dgn2)
    term2[0, 1] = - 2 * torch.mean(dndx * dndy * df_dgn2)
    term2[0, 2] = - 2 * torch.mean(dndx * dndz * df_dgn2)
    term2[1, 2] = - 2 * torch.mean(dndy * dndz * df_dgn2)
    term2[1, 0] = term2[0, 1]
    term2[2, 0] = term2[0, 2]
    term2[2, 1] = term2[1, 2]

    return term1 + term2


def pbe_correlation_stress(box_vecs, den):
    A1, alpha = 0.0310907, 0.2137
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294
    rs = (3 / 4 / pi / den).pow(1 / 3)
    zeta = 2 * A1 * (b1 * rs.pow(0.5) + b2 * rs + b3 * rs.pow(1.5) + b4 * rs.square())
    eps_c = -2 * A1 * (1 + alpha * rs) * torch.log(1 + 1 / zeta)
    deps_dn = -rs / 3 / den * (-2 * A1 * alpha * torch.log(1 + 1 / zeta) + (2 * A1 * A1 * (1 + alpha * rs)
                               * (b1 * rs.pow(-0.5) + 2 * b2 + 3 * b3 * rs.pow(0.5) + 4 * b4 * rs)
                               / (zeta * (zeta + 1))))

    beta, gamma = 0.066725, (1 - np.log(2)) / pi / pi
    A = beta / gamma / (torch.exp(-eps_c / gamma) - 1)
    dAdn = 1 / beta * A.square() * torch.exp(-eps_c / gamma) * deps_dn

    kxyz = wavevectors(box_vecs, den.shape)
    kx, ky, kz = kxyz[..., 0], kxyz[..., 1], kxyz[..., 2]
    gdg = grad_dot_grad(kxyz, den)
    t2 = (1 / 16) * (pi / 3)**(1 / 3) * gdg * den.pow(-7 / 3)
    dt2dn = -7 / 3 * t2 / den
    dt2dgn2 = (1 / 16) * (pi / 3)**(1 / 3) * den.pow(-7 / 3)

    At2 = A * t2
    numer = 1 + At2
    denom = 1 + At2 + At2.square()
    H = gamma * torch.log(1 + beta / gamma * t2 * (numer / denom))

    numer2 = 1 + 2 * At2
    dHdn = beta * torch.exp(-H / gamma) * ((dt2dn * numer2 + dAdn * t2.square()) / denom
                                            - t2 * numer / denom.square() * (dt2dn * A * numer2 + dAdn * t2 * numer2))
    dH_dgn2 = beta * torch.exp(-H / gamma) * (dt2dgn2 * numer2 / denom
                                              - At2 * numer / denom.square() * dt2dgn2 * numer2)

    df_dn = eps_c + H + den * (deps_dn + dHdn)
    df_dgn2 = den * dH_dgn2

    term1 = torch.mean((eps_c + H) * den - den * df_dn) * torch.eye(3, dtype=torch.double, device=den.device)
    dndx, dndy, dndz = grad_i(kx, den), grad_i(ky, den), grad_i(kz, den)
    term2 = torch.empty((3, 3), dtype=torch.double, device=den.device)
    term2[0, 0] = - 2 * torch.mean((gdg + dndx * dndx) * df_dgn2)
    term2[1, 1] = - 2 * torch.mean((gdg + dndy * dndy) * df_dgn2)
    term2[2, 2] = - 2 * torch.mean((gdg + dndz * dndz) * df_dgn2)
    term2[0, 1] = - 2 * torch.mean(dndx * dndy * df_dgn2)
    term2[0, 2] = - 2 * torch.mean(dndx * dndz * df_dgn2)
    term2[1, 2] = - 2 * torch.mean(dndy * dndz * df_dgn2)
    term2[1, 0] = term2[0, 1]
    term2[2, 0] = term2[0, 2]
    term2[2, 1] = term2[1, 2]

    return term1 + term2
