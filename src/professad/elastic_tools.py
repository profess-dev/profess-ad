import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

m_per_bohr = 5.29177210903e-11
A_per_b = m_per_bohr * 1e10

J_per_Ha = 4.3597447222071e-18
eV_per_Ha = J_per_Ha / 1.602176634e-19

GPa_per_atomic = J_per_Ha / m_per_bohr**3 * 1e-9
GPa_per_Ab3 = GPa_per_atomic / (eV_per_Ha / A_per_b**3)


def fit_eos(vol, ene, eos='bm', plot=False):
    r"""
    Fits volume-energy data to the Murnaghan or Birch-Murnaghan equation of state. For an
    accurate fit, it is recommended that the energy-volume points are distributed evenly
    about the energy-volume minima.

    The Murnaghan equation of state is given by

    .. math:: E(V) = E_0 + \frac{K_0 V}{K_0'} \left(\frac{(V_0/V)^{K_0'}}{K_0'-1} + 1 \right)
              - \frac{K_0V_0}{K_0'-1}

    while the Birch-Murnaghan equation of state is given by

    .. math:: E(V) = E_0 + \frac{9 V_0 K_0}{16} \left\{ \left[\left(\frac{V_0}{V}\right)^{2/3}-1\right]^3 K_0'
              + \left[\left(\frac{V_0}{V}\right)^{2/3}-1\right]^2
              \left[6-4\left(\frac{V_0}{V}\right)^{2/3}\right] \right\}

    The parameters are returned in the following order

    1. Equilibrium bulk modulus, :math:`K_0` [GPa]
    2. Equilibrium bulk modulus derivative wrt pressure, :math:`K'_0` [dimensionless]
    3. Equilibrium energy, :math:`E_0` [eV]
    4. Equilibrium volume, :math:`V_0` [Å³]

    Args:
      vol (list)  : Volumes
      ene (list)  : Energies
      eos (string): ``bm`` (default) for Birch-Murnaghan or ``m`` for Murnaghan
      plot (bool) : Whether the volume-energy data points and the fitted curve are plotted

    Returns:
      ndarray: Fitted parameters, Fitting errors
    """
    vol, ene = np.asarray(vol), np.asarray(ene)
    # initial guess is harmonic solid, E = E0 + 0.5*K0*(V-V0)^2/V0
    apar, bpar, cpar = np.polyfit(vol, ene, 2)
    K0_g = -bpar
    V0_g = K0_g / (2 * apar)
    E0_g = cpar - 0.5 * K0_g * V0_g
    K0prime_g = 3.5

    # fit to Murnaghan or Birch-Murnaghan equation of state
    def murn(v, K0, K0prime, E0, V0):
        if eos == 'm':
            return E0 + (K0 * v / K0prime) * ((((V0 / v)**K0prime) / (K0prime - 1)) + 1) - K0 * V0 / (K0prime - 1)
        if eos == 'bm':
            return E0 + 9 * V0 * K0 / 16 * (K0prime * ((V0 / v)**(2 / 3) - 1)**3
                                            + ((V0 / v)**(2 / 3) - 1)**2 * (6 - 4 * (V0 / v)**(2 / 3)))
        else:
            raise ValueError('Only \'m\' or \'bm\' recognized for \'eos\' argument.')
    params, pcov = curve_fit(murn, vol, ene, p0=(K0_g, K0prime_g, E0_g, V0_g), maxfev=1000)
    err = np.sqrt(np.diag(pcov))
    if plot:  # optional plotting
        plt.plot(vol, ene, 'rx')
        vfit = np.linspace(0.99 * vol[0], 1.01 * vol[-1])
        efit = murn(vfit, params[0], params[1], params[2], params[3])
        plt.plot(vfit, efit, 'b-')
        plt.xlabel('Volume/Å³')
        plt.ylabel('Energy/eV')
        plt.legend(['data', 'fit'], loc='best')
        plt.show()
    return params, err


def voigt_moduli(C):
    r"""
    Processes a 6 x 6 elastic constant matrix into the Voigt bulk and shear moduli, given by

    .. math:: 9 K_V = (C_{11} + C_{22} + C_{33}) + 2(C_{12} + C_{23} + C_{31})

    and

    .. math::  15 G_V = (C_{11} + C_{22} + C_{33}) - (C_{12} + C_{23} + C_{31}) + 3(C_{44} + C_{55} + C_{66}).

    Args:
      C (torch.Tensor): 6 by 6 matrix of elastic constants

    Returns:
      torch.Tensor: Voigt bulk and shear moduli
    """
    K = (1 / 9) * ((C[0, 0] + C[1, 1] + C[2, 2]) + 2 * (C[0, 1] + C[1, 2] + C[0, 2]))
    G = (1 / 15) * ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[1, 2] + C[0, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5]))
    return K, G


def reuss_moduli(C):
    r"""
    Processes a 6 x 6 elastic constant matrix into the Reuss bulk and shear moduli, given by

    .. math:: 1/K_R = (S_{11} + S_{22} + S_{33}) + 2(S_{12} + S_{23} + S_{31})

    and

    .. math:: 15/ G_R = 4(S_{11} + S_{22} + S_{33}) - 4(S_{12} + S_{23} + S_{31}) + 3(S_{44} + S_{55} + S_{66}),

    where :math:`S = C^{-1}`.

    Args:
      C (torch.Tensor): 6 by 6 matrix of elastic constants

    Returns:
      torch.Tensor: Reuss bulk and shear moduli
    """
    S = torch.linalg.inv(C)
    K = 1 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[1, 2] + S[0, 2]))
    G = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[1, 2] + S[0, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    return K, G


def shear_average(C, mean_type='arithmetic'):
    """
    Processes a 6 x 6 elastic constant matrix into a shear modulus based on the average of
    the Reuss and Voigt shear moduli, which could be an arithmetic or geometric mean.

    Args:
      C (torch.Tensor)  : 6 by 6 matrix of elastic constants
      mean_type (string): ``arithmetic`` or ``geometric`` mean to be taken
    Returns:
      torch.Tensor: Average shear moduli
    """
    Kv, Gv = voigt_moduli(C)
    Kr, Gr = reuss_moduli(C)
    if mean_type == 'arithmetic':
        return 0.5 * (Gv + Gr)
    elif mean_type == 'geometric':
        return (Gv * Gr)**(1 / 2)
    else:
        raise ValueError('Only \'arithmetic\' or \'geometric\' recognized for \'mean_type\' argument')


def poissons_ratio(K, G):
    r"""
    Processes the bulk and shear moduli (:math:`K` and :math:`G`) into a
    Poisson's ratio (:math:`\nu`) using

    .. math:: \nu = \frac{1}{2} \left(1 - \frac{3G}{3K+G} \right)

    Args:
      K (torch.Tensor or float): Bulk modulus
      G (torch.Tensor or float): Shear modulus

    Returns:
      torch.Tensor: Poissons ratio
    """
    return 0.5 * (1 - 3 * G / (3 * K + G))


def youngs_modulus(K, G):
    r"""
    Processes the bulk and shear moduli (:math:`K` and :math:`G`) into a
    Young's modulus (:math:`E`) using

    .. math:: E = \left(\frac{1}{3G} + \frac{1}{9K} \right)^{-1}

    Args:
      K (torch.Tensor or float): Bulk modulus
      G (torch.Tensor or float): Shear modulus

    Returns:
      torch.Tensor: Young's modulus
    """
    return 1 / (1 / 3 / G + 1 / 9 / K)
