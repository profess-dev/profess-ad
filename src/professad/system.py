import numpy as np
from math import pi, sqrt
import torch
from xitorch.optimize import minimize

from torch_nl import compute_neighborlist

from professad.ion_utils import get_ion_charge, interpolate_recpot, lattice_sum, ion_interaction_sum
from professad.functional_tools import wavevectors
from professad.elastic_tools import fit_eos

from professad._optimizers.lbfgs.lbfgsnew import LBFGSNew
from professad._optimizers.tpgd.two_point_gradient_descent import TPGD

from typing import Optional, Tuple, List, Callable

# --------------------------------------------------------------------------------
# System class for a PROFESS-AD, a Pytorch-based auto-differentiable orbital-free
# density functional theory code with periodic boundary conditions.
# --------------------------------------------------------------------------------


class System():
    """
    Class that represents a system with periodic boundary conditions to facilitate orbital-free
    density functional theory calculations using Pytorch functions. This enables the use Pytorch's
    autograd to compute gradient dependent terms via auto-differentiations. The calculations are also
    GPU compatible due to the use of Pytorch functions.
    """

    # based on 2018 CODATA recommended values
    m_per_bohr = 5.29177210903e-11
    A_per_b = m_per_bohr * 1e10

    J_per_Ha = 4.3597447222071e-18
    eV_per_Ha = J_per_Ha / 1.602176634e-19

    GPa_per_atomic = J_per_Ha / m_per_bohr**3 * 1e-9

    def __init__(self,
                 box_vecs: torch.Tensor,
                 shape: List,
                 ions: List,
                 terms: List[Callable],
                 units: Optional[str] = 'b',
                 coord_type: Optional[str] = 'cartesian',
                 Rc: Optional[float] = None,
                 pme_order: Optional[int] = None,
                 device=torch.device('cpu'),
                 ):
        r"""
        Args:
          box_vecs (torch.Tensor) : Lattice vectors :math:`\mathbf{a},~\mathbf{b},~\mathbf{c}`
                                    with input format

                                    [[:math:`a_x`, :math:`a_y`, :math:`a_z`],
                                    [:math:`b_x`, :math:`b_y`, :math:`b_z`],
                                    [:math:`c_x`, :math:`c_y`, :math:`c_z`]]

          shape (torch.Size or iterable) : Real-space grid shape

          ions (list) : Ion information, with each sublist containing the following

                        ``[name (string), path_to_pseudopotential_file (string), ionic_coordinates (torch.Tensor)]``

                        For example, ``['Al', 'pseudopots/Al.recpot', torch.tensor([[0,0,0]])]``

          units (string) : Units of ``a`` for angstrom or ``b`` for bohr.
          Rc (None or float)  : Cutoff radius for ion-ion interaction summation (in bohr). Default behaviour
                                (``Rc=None``) uses the heuristic :math:`R_c = 12 h_\text{max}` where
                                :math:`h_\text{max}` is the largest interplanar spacing for the cell.
          coord_type (string) : Whether ``cartesian`` or ``fractional`` coordinates were used to
                                represent the ionic coordinates in "ions"
          pme_order (None or even int) : The spline order of the particle-mesh Ewald scheme for a
                                         quasi-linear scaling computation of the structure factor.
                                         The default value of 'None' indicates that the exact quadratic scaling
                                         method is used for computing the structure factor.
          device (torch.device) : Device to store System tensors in. Default behaviour is to use CPU.
        """
        self._device = device; self._terms = terms; self.__shape = shape
        self.set_lattice(box_vecs, units, initialization=True)  # initialize lattice vectors
        self.__pme_order = pme_order
        self.set_Rc(Rc)
        self.__process_ions(ions, coord_type, units)  # initialize ion information
        self._update_ionic_potential()  # initialize external/ionic potential
        self.initialize_density()  # initialize system with uniform density
        self._energy = self._compute_energy()  # compute energy for initialized system

    @classmethod
    def ecut2shape(self, energy_cutoff: float, box_vecs: torch.Tensor) -> Tuple:
        """
        Computes the shape of grid for a lattice given an energy cutoff.

        Args:
          ecut (float)           : Energy cutoff given in eVs
          box_vecs (torch.Tensor): Lattice vectors given in Angstroms

        Returns:
          tuple: Real space grid shape
        """
        bvs = box_vecs / self.A_per_b; ecut = energy_cutoff / self.eV_per_Ha  # atomic units
        shape = 1 + 2 * torch.ceil(sqrt(2 * ecut) / (2 * pi * torch.linalg.inv(bvs.T)).norm(p=2, dim=1))
        return tuple(shape.int().tolist())

    #####################################
    #  Initialization/Update functions  #
    #####################################

    def set_device(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Moves all System tensors to the specified device. By default, the device is set to a
        GPU if is avaliable, otherwise the device is set as a CPU.

        Args:
          device (torch.device): Device
        """
        self._device = device
        self._box_vecs = self._box_vecs.to(self._device)
        self._den = self._den.to(self._device)
        self._v_ext = self._v_ext.to(self._device)
        self._frac_ion_coords = self._frac_ion_coords.to(self._device)

    def __process_ions(self, ions, coord_type, units):
        N_elec, ion_list, name = 0, [], ''
        ion_coords = torch.empty((0, 3), dtype=torch.double, device=self._device)
        for species in ions:   # each species contains [name, path_to_recpot, ion_coordinates]
            charge = get_ion_charge(species[1])
            # ion_list contains list of tuples (ion_name, path_to_recpot, num_of_ions, charge)
            ion_list.append((species[0], species[1], species[2].shape[0], charge))  # track unique ions
            ion_coords = torch.cat([ion_coords, species[2].double().to(self._device)])
            N_elec += species[2].shape[0] * charge  # count number of electrons
            name += (species[0] + str(int(species[2].shape[0])))  # get system name
        self.name = name
        self.N_ion = ion_coords.shape[0]
        self.N_elec = N_elec
        self.ions = ion_list

        # set charges
        self.charges, counter = torch.empty(self.N_ion, dtype=torch.long), 0
        for species in self.ions:
            self.charges[counter:(counter + species[2])] = torch.full((species[2],), species[3])
            counter += species[2]
        self.charges = self.charges.to(dtype=torch.float64, device=self._device)

        # set fractional ionic coordinates
        self.place_ions(ion_coords, coord_type, units, initialization=True)

        # set neighborlist
        mapping, batch_mapping, shifts_idx = compute_neighborlist(
            self.Rc,
            torch.matmul(self._frac_ion_coords, self._box_vecs),
            self._box_vecs,
            torch.ones((3,), device=self._frac_ion_coords.device).bool(),
            torch.zeros((self._frac_ion_coords.shape[0], ), dtype=torch.long,
                        device=self._frac_ion_coords.device),
            self_interaction=False
        )
        self.neighborlist = (mapping, shifts_idx)

    def place_ions(self,
                   ion_coords: torch.Tensor,
                   coord_type: Optional[str] = 'cartesian',
                   units: Optional[str] = 'a',
                   initialization: Optional[bool] = False,
                   ):
        """
        Places ions at given positions.

        Args:
          ion_coords (torch.Tensor) : Ionic coordinates
          coord_type (string)       : Whether ``cartesian`` or ``fractional`` coordinates were used to
                                      represent the ionic coordinates
          units (string)            : Units of ``a`` (Angstrom) or ``b`` (Bohr)
          initialization (bool)     : Whether this function is called during the initialization
                                      of a new system object or not
        """
        ion_coords = ion_coords.clone().double().to(self._device)
        if coord_type == 'cartesian':
            if units == 'a':
                unit_factor = self.A_per_b  # Angstrom
            elif units == 'b':
                unit_factor = 1.0  # Bohr
            else:
                raise ValueError('Parameter \'units\' can only be \'b\' (Bohr) or \'a\' (Angstrom)')
            aux_frac_coords = torch.matmul(ion_coords / unit_factor, torch.linalg.inv(self._box_vecs))
            # make sure fractional coordinates lie in [0,1), the repeated operation is intentional as small,
            # negative values get mapped to 1 (which is outside the permitted range) after the first operation
            self._frac_ion_coords = aux_frac_coords - torch.floor(aux_frac_coords)
            self._frac_ion_coords -= torch.floor(self._frac_ion_coords)
        elif coord_type == 'fractional':
            self._frac_ion_coords = ion_coords - torch.floor(ion_coords)
            self._frac_ion_coords -= torch.floor(self._frac_ion_coords)
        else:
            raise ValueError('Parameter \'coord_type\' can only be \'cartesian\' or \'fractional\'')
        if not initialization:
            self._update_ionic_potential()
            self._energy = self._compute_energy()

    def set_lattice(self,
                    box_vecs: torch.Tensor,
                    units: Optional[str] = 'a',
                    initialization: Optional[bool] = False,
                    ):
        """
        Sets lattice vectors.

        Args:
          box_vecs (torch.Tensor) : Lattice vectors
          units (string)          : Units of ``a`` (Angstrom) or ``b`` (Bohr)
          initialization (bool)   : Whether this function is called during the initialization
                                    of a new system object or not
        """
        if not initialization:
            old_vol = self._vol()
        if units == 'a':
            unit_factor = self.A_per_b  # Angstrom
        elif units == 'b':
            unit_factor = 1.0  # Bohr
        else:
            raise ValueError('Parameter \'units\' can only be \'b\' (Bohr) or \'a\' (Angstrom)')
        self._box_vecs = box_vecs.clone().double().to(self._device) / unit_factor
        if not initialization:
            self._update_ionic_potential()
            self._den *= old_vol / self._vol()
            self._energy = self._compute_energy()

    def _potential_from_ions(self, cart_ion_coords):
        k2 = wavevectors(self._box_vecs, self.__shape).square().sum(-1)
        k = torch.zeros(k2.shape, dtype=torch.double, device=k2.device)
        k[k2 != 0] = torch.sqrt(k2[k2 != 0])
        v_ext = torch.zeros(self.__shape, dtype=torch.double, device=self._device)
        counter = 0
        for species in self.ions:
            v_s_ft = interpolate_recpot(species[1], k)
            positions = cart_ion_coords[counter:(counter + species[2]), :]
            v_ext += lattice_sum(self._box_vecs, self.__shape, positions, v_s_ft, self.__pme_order)
            counter += species[2]
        return v_ext

    def _update_ionic_potential(self):
        cart_ion_coords = torch.matmul(self._frac_ion_coords, self._box_vecs)
        need_vext = False
        for functional in self._terms:
            if not isinstance(functional, torch.nn.Module):
                if functional.__qualname__ == 'IonElectron':
                    need_vext = True
        if need_vext:
            self._v_ext = self._potential_from_ions(cart_ion_coords)
        else:
            self._v_ext = torch.zeros(self.__shape, dtype=torch.double, device=self._device)

    def set_potential(self, pot):
        """
        Sets the external potential to a given one.

        Args:
          pot (torch.Tensor): Given external potential (must match the system's shape attribute)
        """
        assert pot.shape == self.__shape, 'Shape of new potential must match the system\'s.'
        self._v_ext = pot.clone().double().to(self._device)
        self._energy = self._compute_energy()

    def initialize_density(self):
        """
        Initializes the density to that of a uniform density profile.
        """
        self._den = (self.N_elec / self._vol().detach()).repeat(self.__shape)

    def set_density(self, den):
        """
        Sets the electron density to a given one.

        Args:
          den (torch.Tensor): Given electron density (must match the system's shape attribute)
        """
        assert den.shape == self.__shape, 'Shape of new density must match the system\'s.'
        self._den = den.double().to(self._device)
        self._energy = self._compute_energy()

    def _vol(self):
        return torch.linalg.det(self._box_vecs).abs()

    def detach(self):
        """
        Detaches all core variables from any computational graphs they may be in.
        This is used primarily as a clean-up tool to avoid autograd-related bugs.
        """
        self._box_vecs = self._box_vecs.detach()
        self._den = self._den.detach()
        self._v_ext = self._v_ext.detach()
        self._frac_ion_coords = self._frac_ion_coords.detach()

    ########################
    #  "Getter" functions  #
    ########################
    def device(self):
        """
        Returns the device that the system's parameters are stored in.

        Returns:
          torch.device: Device
        """
        return self._device

    def lattice_vectors(self, units='a'):
        """
        Returns the lattice vectors of the system.

        Args:
          units (string): Units of ``a`` (Angstrom) or ``b`` (Bohr)

        Returns:
          torch.Tensor: Lattice vectors
        """
        if units == 'a':
            unit_factor = self.A_per_b  # Angstrom
        elif units == 'b':
            unit_factor = 1.0  # Bohr
        else:
            raise ValueError('Parameter \'units\' can only be \'b\' (Bohr) or \'a\' (Angstrom)')
        return unit_factor * self._box_vecs

    def cartesian_ionic_coordinates(self, units='a'):
        """
        Returns the Cartesian ionic coordinates of the system.

        Args:
          units (string): Units of ``a`` (Angstrom) or ``b`` (Bohr)

        Returns:
          torch.Tensor: Cartesian ionic coordinates
        """
        if units == 'a':
            unit_factor = self.A_per_b  # Angstrom
        elif units == 'b':
            unit_factor = 1.0  # Bohr
        else:
            raise ValueError('Parameter \'units\' can only be \'b\' (Bohr) or \'a\' (Angstrom)')
        return unit_factor * torch.matmul(self._frac_ion_coords, self._box_vecs)

    def fractional_ionic_coordinates(self):
        """
        Returns the fractional ionic coordinates of the system.

        Returns:
          torch.Tensor: Fractional ionic coordinates
        """
        return self._frac_ion_coords

    def ionic_potential(self, units='Ha'):
        """
        Returns the ionic potential / external potential of the system.

        Args:
          units (string)      : Units of energy ('Ha' or 'eV')

        Returns:
          torch.Tensor: Ionic potential
        """
        if units == 'Ha':
            return self._v_ext
        elif units == 'eV':
            return self._v_ext * self.eV_per_Ha
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha\' or \'eV\'')

    def density(self, requires_grad=False):
        r"""
        Returns the electron density of the system.

        Args:
          requires_grad (bool): Whether the returned density can be differentiated
                                (used for training kinetic functionals for example)

        Returns:
          torch.Tensor: Electron density in :math:`\text{bohr}^{-3}`
        """
        if requires_grad:
            return self._differentiable_gs_properties('density')
        else:
            return self._den.detach()

    def check_density_convergence(self, method='dEdchi'):
        r"""
        Utility function that computes the measures of density convergence in atomic units.
        I.e. how well the Euler equation,

        .. math:: \frac{\delta E}{\delta n(\mathbf{r})} = \mu,

        is obeyed. :math:`\mu` is the chemical potential.

        If the ``method = 'dEdchi'``, this function computes the quantity

        .. math:: \text{Max}\left(\left|\frac{\delta E}{\delta \chi(\mathbf{r})}\right|\right)
                  ~~\text{where} ~~
                  n(\mathbf{r}) = \frac{N_e \chi^2(\mathbf{r})}{\int~d^3\mathbf{r}'\chi^2(\mathbf{r}')}

        If the ``method = 'euler'``, this function computes the quantity

        .. math:: \text{Max}\left(\left|\mu - \frac{\delta E}{\delta n(\mathbf{r})}\right|\right)
                  ~~\text{where} ~~
                  \mu = \frac{1}{N_e} \int~d^3\mathbf{r} \frac{\delta E}{\delta n(\mathbf{r})} n(\mathbf{r})

        :math:`N_e` is the number of electrons.

        Args:
          method (string): ``dEdchi`` or ``euler``

        Returns:
          float: A measure of density convergence
        """
        if method == 'dEdchi':
            dEdchi = self.functional_derivative('chi')
            return torch.max(torch.abs(dEdchi)).item()
        elif method == 'euler':
            dEdn = self.functional_derivative('density')
            mu = torch.mean(dEdn * self._den) * self._vol() / self.N_elec
            return torch.max(torch.abs(mu - dEdn)).item()

    def functional_derivative(self, type='density', requires_grad=False):
        r"""
        Returns, in atomic units, the functional derivative of the system,

        .. math:: \frac{\delta E}{\delta n(\mathbf{r})} ~~\text{or} ~~ \frac{\delta E}{\delta \chi(\mathbf{r})}
                  ~~\text{where} ~~
                  n(\mathbf{r}) = \frac{N_e \chi^2(\mathbf{r})}{\int~d^3\mathbf{r}'\chi^2(\mathbf{r}')}.

        Args:
          type (str)          : Functional derivative computed with respect to ``density`` or ``chi``
          requires_grad (bool): Whether the returned energy has ``requires_grad = True`` or not
                                (used for training kinetic functionals for example)

        Returns:
          torch.Tensor: Functional derivative
        """
        self.detach()  # to avoid any core variables from having 'requires_grad = True'
        if type == 'density':
            self._den.requires_grad = True
            E = self._compute_energy(for_den_opt=True)
            dEdn = torch.autograd.grad(E, self._den, create_graph=requires_grad)[0] \
                   / (self._vol() / self._den.numel())
            self._den = self._den.detach()
            return dEdn
        elif type == 'chi':
            chi = torch.sqrt(self._den)
            chi.requires_grad = True
            N_tilde = torch.mean(chi.square()) * self._vol()
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy(for_den_opt=True)
            dEdchi = torch.autograd.grad(E, chi, create_graph=requires_grad)[0] \
                     / (self._vol() / self._den.numel())
            self._den = self._den.detach()
            return dEdchi

    def chemical_potential(self):
        """
        Returns the chemical potential of the system.

        Returns:
          torch.Tensor: Chemical potential
        """
        dEdn = self.functional_derivative('density')
        return (torch.mean(dEdn * self._den) * self._vol() / self.N_elec).item()

    def energy(self, units='Ha', requires_grad=False):
        """
        Returns the energy of the system.

        Args:
          units (string)      : Units of energy (``Ha`` or ``eV``)
          requires_grad (bool): Whether the returned energy has ``requires_grad = True`` or not
                                (used for training kinetic functionals for example)

        Returns:
          float or torch.Tensor (depending on requires_grad): Energy
        """
        if requires_grad:
            E = self._differentiable_gs_properties('energy')
        else:
            E = self._energy.item()
        if units == 'Ha':
            return E
        elif units == 'eV':
            return E * self.eV_per_Ha
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha\' or \'eV\'')

    def volume(self, units='b3'):
        r"""
        Computes the cell volume :math:`\Omega`.

        Args:
          units (string): Units of the pressure returned (``b3`` or ``a3``)

        Returns:
          float: Volume
        """
        if units == 'b3':
            return self._vol().item()
        elif units == 'a3':
            return self._vol().item() * self.A_per_b**3
        else:
            raise ValueError('Parameter \'units\' can only be \'b3\' or \'a3\'')

    def pressure(self, units='Ha/b3'):
        r"""
        Computes, via autograd and/or Xitorch, the pressure

        .. math:: P = \frac{dE[n]}{d\Omega}

        where the cell volume is :math:`\Omega`.

        Args:
          units (string)      : Units of the pressure returned (``Ha/b3``, ``eV/a3`` or ``GPa``)
          requires_grad (bool): Whether the returned pressure has ``requires_grad = True`` or not
                                (used for training kinetic functionals for example)
        Returns:
          float or torch.Tensor (depending on requires_grad): Pressure
        """
        if units == 'Ha/b3':
            unit_factor = 1.0
        elif units == 'eV/a3':
            unit_factor = self.eV_per_Ha / self.A_per_b**3
        elif units == 'GPa':
            unit_factor = self.GPa_per_atomic
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha/b3\', \'eV/a3\' or \'GPa\'')
        return self.__compute_volume_derivatives(bulk_modulus=False) * unit_factor

    def enthalpy(self, units='Ha'):
        """
        Returns the enthalpy, :math:`H=U+PV`, of the system.

        Args:
          units (string): Units of energy (``Ha`` or ``eV``)

        Returns:
          float: Enthalpy
        """
        H = self._energy.item() + self.pressure() * self.volume()
        if units == 'Ha':
            return H
        elif units == 'eV':
            return H * self.eV_per_Ha
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha\' or \'eV\'')

    def bulk_modulus(self, units='Ha/b3'):
        r"""
        Computes, via autograd and Xitorch, the bulk modulus

        .. math:: K = \Omega \frac{d^2 E[n]}{d \Omega^2}

        where the cell volume is :math:`\Omega`.

        Args:
          units (string)      : Units of the bulk modulus returned (``Ha/b3``, ``eV/a3`` or ``GPa``)
          requires_grad (bool): Whether the returned bulk modulus has ``requires_grad = True`` or not
                                (used for training kinetic functionals for example)
        Returns:
          float or torch.Tensor (depending on requires_grad): Bulk modulus
        """
        if units == 'Ha/b3':
            unit_factor = 1.0
        elif units == 'eV/a3':
            unit_factor = self.eV_per_Ha / self.A_per_b**3
        elif units == 'GPa':
            unit_factor = self.GPa_per_atomic
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha/b3\', \'eV/a3\' or \'GPa\'')
        P, K = self.__compute_volume_derivatives(bulk_modulus=True)
        return K * unit_factor

    def eos_fit(self, f=0.05, N=9, eos='bm', verbose=False, plot=False, **den_opt_kwargs):
        """
        Perform a Birch-Murnaghan equation of state fit for a given system.
        The volume of the system is taken to be an approximation of the true
        equilibrium volume.

        The parameters are returned in the following order

        1. Equilibrium bulk modulus, :math:`K_0` [GPa]
        2. Equilibrium bulk modulus derivative wrt pressure, :math:`K'_0` [dimensionless]
        3. Equilibrium energy, :math:`E_0` [eV]
        4. Equilibrium volume, :math:`V_0` [Å³]

        Args:
          f (float)             : Fraction by which the volume is compressed and stretched by
                                  (default: 0.05)
          N (int)               : Number of energy-volume data points used for the fit
                                  (default: 9)
          eos (string)          : ``bm`` (default) for Birch-Murnaghan or ``m`` for Murnaghan
          verbose (boolean)     : Whether the energy-volume values are printed
          plot (boolean)        : Whether an energy-volume plot is made
          den_opt_kwargs        : Arguments for density optimization. The default values are:
                                  ``{'ntol': 1e-10, 'n_conv_cond_count': 3, 'n_method': 'LBFGS',
                                  'n_step_size': 0.1, 'n_maxiter': 1000, 'conv_target': 'dE',
                                  'n_verbose': False, 'from_uniform': False}``

        Returns:
          ndarray: Fitted parameters, Fitting errors
        """
        den_opt_inputs = {'ntol': 1e-10, 'n_conv_cond_count': 3, 'n_method': 'LBFGS',
                          'n_step_size': 0.1, 'n_maxiter': 1000, 'conv_target': 'dE',
                          'n_verbose': False, 'from_uniform': False}
        den_opt_inputs.update(den_opt_kwargs)

        pred_v0 = self.volume('a3')
        norm_box_vecs = self.lattice_vectors('a') / pred_v0**(1 / 3)
        vs = pred_v0 * np.linspace(1 - f, 1 + f, N)
        energies, volumes = [], []
        if verbose:
            print('\n{:^22} {:^22}'.format('Volume [Å³ per atom]', 'Energy [eV per atom]'))
        for v in vs:
            self.set_lattice(v**(1 / 3) * norm_box_vecs, units='a')
            self.optimize_density(**den_opt_inputs)
            vol_per_atom = self.volume('a3') / self.N_ion
            ene_per_atom = self.energy('eV') / self.N_ion
            volumes.append(vol_per_atom)
            energies.append(ene_per_atom)
            if verbose:
                print('{:^22.10f} {:^22.10f}'.format(vol_per_atom, ene_per_atom))
        params, err = fit_eos(volumes, energies, eos, plot)
        # convert units
        params[0] *= self.GPa_per_atomic / (self.eV_per_Ha / self.A_per_b**3)  # convert bulk modulus to GPa
        err[0] *= self.GPa_per_atomic / (self.eV_per_Ha / self.A_per_b**3)  # convert bulk modulus to GPa
        return params, err

    def forces(self, units='Ha/b'):
        r"""
        Computes, via autograd, the forces

        .. math:: F_{\alpha, i} = - \frac{dE[n]}{dR_{\alpha, i}}

        where :math:`\alpha` is an ion index, :math:`i \in \{x,y,z\}` and
        :math:`R_{\alpha, i}` are the Cartesian ionic coordinates.

        Args:
          units (string): Units of the forces returned (``Ha/b`` or ``eV/a``)

        Returns:
          torch.Tensor: Forces
        """
        if units == 'Ha/b':
            return self._compute_forces()
        elif units == 'eV/a':
            return self._compute_forces() * self.eV_per_Ha / self.A_per_b
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha/b\' or \'eV/a\'')

    def stress(self, units='Ha/b3'):
        r"""
        Computes, via autograd, the stress tensor

        .. math:: \sigma_{ij} = \frac{1}{\Omega} \sum_k \frac{\partial E[n]}{\partial h_{ik}} h_{jk}

        where :math:`h_{ij}` are matrix elements of a matrix whose columns are lattice vectors
        and the cell volume is :math:`\Omega`.

        Args:
          units (string): Units of the stress tensor returned (``Ha/b3``, ``eV/a3`` or ``GPa``)

        Returns:
          torch.Tensor: Stress tensor
        """
        if units == 'Ha/b3':
            unit_factor = 1.0
        elif units == 'eV/a3':
            unit_factor = self.eV_per_Ha / self.A_per_b**3
        elif units == 'GPa':
            unit_factor = self.GPa_per_atomic
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha/b3\', \'eV/a3\' or \'GPa\'')
        return self._compute_stress() * unit_factor

    def elastic_constants(self, units='Ha/b3'):
        r"""
        Computes, via autograd and Xitorch, the elastic constants (Birch coefficients)

        .. math:: C_{ijk\ell} = \frac{\partial \sigma_{ij}}{\partial \epsilon_{k\ell}}
                  = \sum_m \frac{\partial \sigma_{ij}}{\partial h_{km}} h_{\ell m}

        where :math:`h_{ij}` are matrix elements of a matrix whose columns are lattice vectors.

        Args:
          units (string): Units of the elastic constants returned (``Ha/b3``, ``eV/a3`` or ``GPa``)

        Returns:
          torch.Tensor: Elastic constants
        """
        if units == 'Ha/b3':
            unit_factor = 1.0
        elif units == 'eV/a3':
            unit_factor = self.eV_per_Ha / self.A_per_b**3
        elif units == 'GPa':
            unit_factor = self.GPa_per_atomic
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha/b3\', \'eV/a3\' or \'GPa\'')
        return self._compute_elastic_constants() * unit_factor

    def force_constants(self, primitive_ion_indices, units='eV/a2'):
        r"""
        Computes, via autograd, the force constants

        .. math:: \Phi_{\alpha, i, \beta, j} = - \frac{dF_{\alpha, i} }{dR_{\beta, j}}

        where :math:`\alpha, \beta` are ion indices, :math:`i,j \in \{x,y,z\}` and
        :math:`R_{\beta, j}` are the Cartesian ionic coordinates.

        Args:
          units (string)              : Units of the force constants returned (``Ha/b2`` or ``eV/a2``)
          primitive_ion_indices (list): List of indices corresponding to ions in the primitive cell.
                                        ``primitive_ion_indices`` begins counting from 0

        Returns:
          torch.Tensor: Force constants with shape ``[len(primitive_ion_indices), N_ions, 3, 3]``
        """
        if units == 'Ha/b2':
            return self._compute_force_constants(primitive_ion_indices)
        elif units == 'eV/a2':
            return self._compute_force_constants(primitive_ion_indices) * self.eV_per_Ha / self.A_per_b**2
        else:
            raise ValueError('Parameter \'units\' can only be \'Ha/b2\' or \'eV/a2\'')

    ####################################
    #  Ion/Electron Interaction Terms  #
    ####################################
    def set_Rc(self, Rc=None):
        r"""
        Sets the cutoff radius for the ion-ion interaction energy. Default behaviour (``Rc=None``)
        uses the heuristic :math:`R_c = 12 h_\text{max}` where :math:`h_\text{max}` is the largest
        interplanar spacing for the cell.

        Args:
          Rc (None or float): Cutoff radius for ion-ion interaction summation (in bohr).
        """
        h_max = torch.linalg.inv(self._box_vecs.T).norm(p=2, dim=1).reciprocal().max().item()
        if Rc is None:  # original paper's heuristic
            self.Rd = 2 * h_max
            self.Rc = 3 * self.Rd * self.Rd / h_max
        else:  # PROFESS 4.0 uses Rc = 250 bohr
            self.Rc = Rc
            self.Rd = torch.sqrt(h_max * Rc / 3)

    def _ion_ion_interaction(self, cart_ion_coords: torch.Tensor):
        """
        Computes the ion-ion interaction energy of the system.

        Args:
          cart_ion_coords (torch.Tensor): Cartesian ionic coordinates
        """
        E_ion = ion_interaction_sum(self._box_vecs, cart_ion_coords, self.charges, self.Rc, self.Rd,
                                    self.neighborlist)
        self._Eion_cache = E_ion.item()
        return E_ion

    ##################################
    # Density Optimization Functions #
    ##################################
    def _compute_energy(self, for_den_opt=False, use_ion_cache=False):
        energy_density = torch.zeros(self._den.shape,
                                     dtype=self._den.dtype,
                                     device=self._den.device)
        kxyz = wavevectors(self._box_vecs, self._den.shape)
        include_ion_ion = False
        for functional in self._terms:
            if not isinstance(functional, torch.nn.Module):
                if functional.__qualname__ == 'IonElectron':
                    energy_density = energy_density + (self._den * self._v_ext)
                elif functional.__qualname__ == 'IonIon':
                    include_ion_ion = True
                else:
                    energy_density = energy_density + functional(self._den, kxyz)
            else:
                energy_density = energy_density + functional(self._den, kxyz)

        E = torch.mean(energy_density) * self._vol()
        if include_ion_ion and (not for_den_opt):
            if use_ion_cache:
                E = E + self._Eion_cache
            else:
                E = E + self._ion_ion_interaction(torch.matmul(self._frac_ion_coords, self._box_vecs))

        return E

    def optimize_density(self, ntol=1e-7, n_conv_cond_count=3, n_method='LBFGS', n_step_size=0.1,
                               n_maxiter=1000, conv_target='dE', n_verbose=False, from_uniform=False,
                               potentials=None):
        r"""
        Performs an orbital-free density optimization procedure to minimize the energy via a direct optimization using
        a Pytorch-based modified LBFGS [`10.1109/eScience.2018.00112 <https://ieeexplore.ieee.org/document/8588731>`_]
        optimizer, or a Two-Point Gradient Descent (TPGD)
        [`IMA Journal of Numerical Analysis, Volume 8, Issue 1, January 1988, Pages 141–148
        <https://doi.org/10.1093/imanum/8.1.141>`_] algorithm.

        Args:
          ntol (float)           : Convergence tolerance for density optimization. The optimization
                                   procedure stops when the convergence target (``conv_target``) variable
                                   is lower than ``ntol`` for a number (``n_conv_cond_count``) of consecutive iterations
                                   (default value is 1e-7)
          n_conv_cond_count (int): 'Convergence condition count', which is the number of times the convergence
                                   condition has to be met in consecutive iterations before the optimization
                                   terminates
          n_method (string)      : ``LBFGS`` or ``TPGD``
          n_step_size (float)    : Step size for the optimizers
          n_maxiter (int)        : Maximum number of density optimization iterations
          conv_target (string)   : ``dE`` (energy difference), ``dEdchi`` (Max :math:`|\delta E/\delta \chi|`),
                                    or ``euler`` (Max :math:`|\mu-\delta E/\delta n|`)
          n_verbose (bool)       : Whether the density optimization progress is printed out or not
          from_uniform (bool)    : Whether the density optimization begins from a uniform density or not. When
                                   'False', the behaviour is to compare the energy computed with the current
                                   density and the energy computed with a uniform density and initialize the
                                   density to whichever that yields a lower energy.
          potentials (function)  : Explicitly implemented functional derivatives (can be used to check that
                                   analytically derived and explicitly implemented functional derivatives are correct)
        """
        self.detach()  # to avoid any core variables from having 'requires_grad = True'

        if from_uniform:
            self.initialize_density()  # set uniform density
        else:
            # compare energy computed from current density and uniform density
            # and choose the lower one for the initial state to speed up convergence
            current_den = self._den
            current_E = self._compute_energy(for_den_opt=True)
            self.initialize_density()
            uniform_E = self._compute_energy(for_den_opt=True)
            if current_E < uniform_E:
                self.set_density(current_den)

        # initialize unconstrained optimization variable
        chi = torch.sqrt(self._den).requires_grad_()
        if n_method == 'LBFGS':
            optimizer = LBFGSNew([chi], lr=n_step_size, history_size=8, max_iter=6)
        elif n_method == 'TPGD':
            optimizer = TPGD([chi], lr=n_step_size)
        else:
            raise ValueError('Only \'LBFGS\' or \'TPGD\' recognized for \'n_method\' argument')

        if potentials is None:

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                N_tilde = torch.mean(chi.square()) * self._vol()
                self._den = (self.N_elec / N_tilde) * chi.square()
                E = self._compute_energy(for_den_opt=True)
                if E.requires_grad:
                    E.backward()
                return E

        else:

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                chi.requires_grad = False
                N_tilde = torch.mean(chi.square()) * self._vol()
                self._den = ((self.N_elec / N_tilde) * chi.square())
                E = self._compute_energy(for_den_opt=True)
                dEdn = potentials(self._box_vecs, self._den)
                dEdchi = (self.N_elec / N_tilde) * 2 * chi * \
                         (dEdn - torch.mean(dEdn * self._den) * self._vol() / self.N_elec)
                chi.requires_grad = True
                chi.grad = dEdchi * (self._vol() / self._den.numel())
                return E

        E_prev = self._compute_energy(for_den_opt=True).item() * self.eV_per_Ha

        if n_verbose:
            print('Starting density optimization')
            print('{:^8} {:^12} {:^12} {:^18} {:^18}'
                  .format('Iter', 'E [eV]', 'dE [eV]', 'Max |𝛿E/𝛿χ|', 'Max |µ-𝛿E/𝛿n|'))
            dEdchi = self.check_density_convergence('dEdchi')
            euler = self.check_density_convergence('euler')
            print('{:^8} {:^12.6f} {:^12.6g} {:^18.6g} {:^18.6g}'.format(0, E_prev, 0, dEdchi, euler))

        conv_counter = 0
        for iter in range(1, round(n_maxiter) + 1):
            optimizer.step(closure)
            dEdchi = torch.abs(chi.grad / (self._vol() / self._den.numel())).max().item()

            E = self._compute_energy(for_den_opt=True).item() * self.eV_per_Ha
            dE = E - E_prev
            E_prev = E

            if n_verbose or conv_target == 'euler':
                euler = self.check_density_convergence('euler')
            if n_verbose:
                print('{:^8} {:^12.6f} {:^12.6g} {:^18.6g} {:^18.6g}'
                      .format(iter, E_prev, dE, dEdchi, euler))

            # getting convergence target
            if conv_target == 'dE':
                stop_var = np.abs(dE)
            elif conv_target == 'dEdchi':
                stop_var = dEdchi
            elif conv_target == 'euler':
                stop_var = euler
            else:
                raise ValueError('Only \'dE\', \'dEdchi\' or \'euler\' recognized as \'conv_target\' argument')

            if iter > 5:  # check for convergence only after the 5th iteration
                if stop_var < ntol:
                    conv_counter += 1
                else:
                    conv_counter = 0

            # exit due to success
            if conv_counter == n_conv_cond_count:
                if n_verbose:
                    print('Density optimization successfully converged in {} step(s) \n'.format(iter))
                break

            # exit due to failure
            if iter == round(n_maxiter):
                if n_verbose:
                    print('Density optimization failed to converge in {} steps \n'.format(int(iter)))
        self.detach()
        self._energy = self._compute_energy(use_ion_cache=True)

    ##############################################
    # First-order Derivative Terms and Functions #
    ##############################################
    def _compute_forces(self):
        cart_ion_coords = torch.matmul(self._frac_ion_coords, self._box_vecs)
        cart_ion_coords.requires_grad_()
        U = 0
        for functional in self._terms:
            if not isinstance(functional, torch.nn.Module):
                if functional.__name__ == 'IonElectron':
                    U = U + torch.mean(self._den * self._potential_from_ions(cart_ion_coords)) * self._vol()
                elif functional.__name__ == 'IonIon':
                    U = U + self._ion_ion_interaction(cart_ion_coords)
        return torch.autograd.grad(U, [cart_ion_coords])[0].neg_()

    def _compute_stress(self):
        # use voigt strain trick to get stress
        voigt_strain = torch.zeros((6,),
                                   dtype=self._box_vecs.dtype,
                                   device=self._box_vecs.device,
                                   requires_grad=True)
        deformation = (torch.eye(3, dtype=voigt_strain.dtype, device=voigt_strain.device)
                       .add_(_voigt_to_3by3_strain(voigt_strain)))
        self._box_vecs = torch.matmul(self._box_vecs, deformation)
        # incorporate lattice vector dependence in density
        self._den = self._den * self._vol().detach() / self._vol()
        # incorporate lattice vector dependence in positions
        self._v_ext = self._potential_from_ions(torch.matmul(self._frac_ion_coords, self._box_vecs))
        E = self._compute_energy()
        dEdstrain = torch.autograd.grad(E, voigt_strain)[0]
        self.detach()  # reset autograd features
        return _voigt_to_3by3_stress(dEdstrain.div(self._vol()))

    def optimize_geometry(self, ftol=0.02, stol=0.002, g_conv_cond_count=3, g_method='LBFGSlinesearch',
                          g_step_size=0.1, g_maxiter=1000, g_verbose=False, **den_opt_kwargs):
        """
        Performs a geometry optimization to minimize the energy by varying the ionic positions and/or lattice vectors,
        minimizing the ionic forces and stress in the process.

        Set ``ftol = None`` to only vary lattice vectors.
        Set ``stol = None`` to only vary ionic positions.

        Args:
          ftol (float)           : Force tolerance in eV/Å - optimization terminates when the largest
                                   force component goes below this value for a number (``g_conv_cond_count``)
                                   of consecutive iterations
          stol (float)           : Stress tolerance in eV/Å³ - optimization terminates when the largest
                                   stress component goes below this value  for a number (``g_conv_cond_count``)
                                   of consecutive iterations
          g_conv_cond_count (int): 'Convergence condition count', which is the number of times the convergence
                                   condition has to be met in consecutive iterations before the optimization
                                   terminates
          g_method (string)      : ``LBFGSlinesearch``, ``LBFGS``, ``RPROP`` or ``TPGD``
          g_step_size (float)    : Step size for the optimizer
          g_maxiter (int)        : Maximum number of geometry optimization iterations
          g_verbose (bool)       : Whether the geometry optimization progress is printed out or not
          den_opt_kwargs         : Arguments for density optimization. The default values are:
                                   ``{'ntol': 1e-10, 'n_conv_cond_count': 3, 'n_method': 'LBFGS',
                                   'n_step_size': 0.1, 'n_maxiter': 1000, 'conv_target': 'dE',
                                   'n_verbose': False, 'from_uniform': False}``

        Returns:
          bool: Whether the optimization was successful or not
        """
        # handle den_opt_kwargs
        den_opt_inputs = {'ntol': 1e-10, 'n_conv_cond_count': 3, 'n_method': 'LBFGS',
                          'n_step_size': 0.1, 'n_maxiter': 1000, 'conv_target': 'dE',
                          'n_verbose': False, 'from_uniform': False}
        den_opt_inputs.update(den_opt_kwargs)

        frac_ion_coords = self._frac_ion_coords.clone()
        box_vecs = self._box_vecs.clone()
        param_list = []
        if ftol is not None:
            frac_ion_coords.requires_grad = True
            param_list.append(frac_ion_coords)
        if stol is not None:
            box_vecs.requires_grad = True
            param_list.append(box_vecs)
        if (ftol is None) and (stol is None):
            raise ValueError('At least one of \'stol\' or \'ftol\' cannot be \'None\'')

        if g_method == 'RPROP':
            optimizer = torch.optim.Rprop(param_list, lr=g_step_size)
        elif g_method == 'TPGD':
            optimizer = TPGD(param_list, lr=g_step_size)
        elif g_method == 'LBFGSlinesearch':
            optimizer = LBFGSNew(param_list, lr=g_step_size, history_size=8, max_iter=6, line_search_fn=True)
        elif g_method == 'LBFGS':
            optimizer = LBFGSNew(param_list, lr=g_step_size, history_size=8, max_iter=6)
        else:
            raise ValueError('Only \'LBFGSlinesearch\', \'LBFGS\', \'RPROP\' or \'TPGD\' recognized for \'g_method\'')

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            self._frac_ion_coords = frac_ion_coords
            self._box_vecs = box_vecs
            # incorporate lattice vector dependence in ionic potential
            self._update_ionic_potential()
            N_tilde = torch.mean(chi.square()) * self._vol()
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy()
            if E.requires_grad:
                E.backward(inputs=param_list)
            return E

        self.optimize_density(**den_opt_inputs)
        E_prev = self.energy('eV') / self.N_ion
        if g_verbose:
            max_force = torch.max(torch.abs(self.forces('eV/a'))).item()
            max_stress = torch.max(torch.abs(self.stress('eV/a3'))).item()
            print('{:^7} {:^20} {:^20} {:^20} {:^20}'
                  .format('Iter', 'E [eV per atom]', 'dE [eV per atom]', 'Max Force [eV/Å]', 'Max Stress [eV/Å³]'),
                  flush=True)
            print('{:^7} {:^20.6f} {:^20.6g} {:^20.6g} {:^20.6g}'
                  .format(0, E_prev, 0, max_force, max_stress), flush=True)

        conv_counter = 0; success_iter = None
        for iter in range(1, round(g_maxiter) + 1):
            chi = torch.sqrt(self._den)
            optimizer.step(closure)
            self.detach()

            self.optimize_density(**den_opt_inputs)
            E_new = self.energy('eV') / self.N_ion
            max_force = torch.max(torch.abs(self.forces('eV/a'))).item()
            max_stress = torch.max(torch.abs(self.stress('eV/a3'))).item()
            if g_verbose:
                print('{:^7} {:^20.6f} {:^20.6g} {:^20.6g} {:^20.6g}'
                      .format(iter, E_new, E_new - E_prev, max_force, max_stress), flush=True)
            E_prev = E_new

            if iter > 3:  # check for convergence only after the 3rd iteration
                if ftol is None:
                    if max_stress < stol:
                        conv_counter += 1
                    else:
                        conv_counter = 0
                elif stol is None:
                    if max_force < ftol:
                        conv_counter += 1
                    else:
                        conv_counter = 0
                else:
                    if (max_force < ftol) and (max_stress < stol):
                        conv_counter += 1
                    else:
                        conv_counter = 0

            if conv_counter == g_conv_cond_count:
                success_iter = iter
                break

        if g_verbose:
            if success_iter is not None:
                print('Geometry optimization successfully converged in {} step(s) \n'.format(success_iter), flush=True)
            else:
                print('Geometry optimization failed to converge in {} step(s) \n'.format(g_maxiter), flush=True)

        return success_iter is not None

    def optimize_parameterized_geometry(self, params, parameterized_geometry, ftol=0.02, stol=0.002,
                                        g_conv_cond_count=3, g_method='LBFGSlinesearch', g_step_size=0.1,
                                        g_maxiter=1000, g_verbose=False, param_string=None, **den_opt_kwargs):
        """
        Performs a parameterized geometry optimization to minimize the energy by varying
        the parameters that describe the ionic positions and/or lattice vectors.

        Set ``ftol = None`` to only vary lattice vectors.
        Set ``stol = None`` to only vary ionic positions.

        Args:
          params (torch.Tensor)            : The parameters that describe the geometry
          parameterized_geometry (function): Function that accepts ``params`` as input and returns
                                             ``box_vecs, frac_ion_coords``, that is the lattice vectors
                                             and fractional ionic coordinates based on the parameters
                                             (``box_vecs`` must be in units of Bohr)
          ftol (float)           : Force tolerance in eV/Å - optimization terminates when the largest
                                   force component goes below this value for a number (``g_conv_cond_count``)
                                   of consecutive iterations
          stol (float)           : Stress tolerance in eV/Å³ - optimization terminates when the largest
                                   stress component goes below this value  for a number (``g_conv_cond_count``)
                                   of consecutive iterations
          g_conv_cond_count (int): 'Convergence condition count', which is the number of times the convergence
                                   condition has to be met in consecutive iterations before the optimization
                                   terminates
          g_method (string)      : ``LBFGSlinesearch``, ``LBFGS``, ``RPROP`` or ``TPGD``
          g_step_size (float)    : Step size for the optimizer
          g_maxiter (int)        : Maximum number of geometry optimization iterations
          g_verbose (bool)       : Whether the geometry optimization progress is printed out or not
          param_string (function): For printing out the parameter values during the optimization
          den_opt_kwargs         : Arguments for density optimization. The default values are:
                                   ``{'ntol': 1e-10, 'n_conv_cond_count': 3, 'n_method': 'LBFGS',
                                   'n_step_size': 0.1, 'n_maxiter': 1000, 'conv_target': 'dE',
                                   'n_verbose': False, 'from_uniform': False}``

        Returns:
          bool: Whether the optimization was successful or not
        """
        # handle den_opt_kwargs
        den_opt_inputs = {'ntol': 1e-10, 'n_conv_cond_count': 3, 'n_method': 'LBFGS',
                          'n_step_size': 0.1, 'n_maxiter': 1000, 'conv_target': 'dE',
                          'n_verbose': False, 'from_uniform': False}
        den_opt_inputs.update(den_opt_kwargs)

        if g_method == 'RPROP':
            optimizer = torch.optim.Rprop([params], lr=g_step_size)
        elif g_method == 'TPGD':
            optimizer = TPGD([params], lr=g_step_size)
        elif g_method == 'LBFGSlinesearch':
            optimizer = LBFGSNew([params], lr=g_step_size, history_size=8, max_iter=6, line_search_fn=True)
        elif g_method == 'LBFGS':
            optimizer = LBFGSNew([params], lr=g_step_size, history_size=8, max_iter=6)
        else:
            raise ValueError('Only \'LBFGSlinesearch\', \'LBFGS\', \'RPROP\' or \'TPGD\' recognized for \'g_method\'')

        if (ftol is None) and (stol is None):
            raise ValueError('At least one of \'stol\' or \'ftol\' cannot be \'None\'')

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            box_vecs, frac_ion_coords = parameterized_geometry(params)
            self._frac_ion_coords = frac_ion_coords
            self._box_vecs = box_vecs
            # incorporate lattice vector dependence in ionic potential
            self._update_ionic_potential()
            N_tilde = torch.mean(chi.square()) * self._vol()
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy()
            if E.requires_grad:
                E.backward(inputs=[params])
            return E

        self.optimize_density(**den_opt_inputs)
        E_prev = self.energy('eV') / self.N_ion

        if g_verbose:
            max_force = torch.max(torch.abs(self.forces('eV/a'))).item()
            max_stress = torch.max(torch.abs(self.stress('eV/a3'))).item()
            param_print = ''
            if param_string is not None:
                param_str = param_string(params); param_print = 'Params'
            print('{:^7} {:^20} {:^20} {:^20} {:^20}'
                   .format('Iter', 'E [eV per atom]', 'dE [eV per atom]', 'Max Force [eV/Å]', 'Max Stress [eV/Å³]')
                   + param_print)
            print('{:^7} {:^20.6f} {:^20.6g} {:^20.6g} {:^20.6g}'
                   .format(0, E_prev, 0, max_force, max_stress) + param_str)

        conv_counter = 0; success_iter = None
        for iter in range(1, round(g_maxiter) + 1):
            chi = torch.sqrt(self._den)
            optimizer.step(closure)
            self.detach()

            self.optimize_density(**den_opt_inputs)
            E_new = self.energy('eV') / self.N_ion
            max_force = torch.max(torch.abs(self.forces('eV/a'))).item()
            max_stress = torch.max(torch.abs(self.stress('eV/a3'))).item()
            if g_verbose:
                if param_string is not None:
                    param_str = param_string(params)
                print('{:^7} {:^20.6f} {:^20.6g} {:^20.6g} {:^20.6g}'
                       .format(iter, E_new, E_new - E_prev, max_force, max_stress) + param_str)
            E_prev = E_new

            if iter > 3:  # check for convergence only after the 3rd iteration
                if ftol is None:
                    if max_stress < stol:
                        conv_counter += 1
                    else:
                        conv_counter = 0
                elif stol is None:
                    if max_force < ftol:
                        conv_counter += 1
                    else:
                        conv_counter = 0
                else:
                    if (max_force < ftol) and (max_stress < stol):
                        conv_counter += 1
                    else:
                        conv_counter = 0

            if conv_counter == g_conv_cond_count:
                success_iter = iter
                break

        if g_verbose:
            if success_iter is not None:
                print('Geometry optimization successfully converged in {} step(s) \n'.format(success_iter))
            else:
                print('Geometry optimization failed to converge in {} step(s) \n'.format(g_maxiter))

        return success_iter is not None

    ##################################
    # Second-order Derivative Terms  #
    ##################################

    def _differentiable_gs_properties(self, output='energy'):
        # output is 'energy' or 'density'

        def energy(chi):
            N_tilde = torch.mean(chi.square()) * self._vol()
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy()
            return E

        # use Xitorch's minimize to get "minimization functional" derivative
        chi = torch.sqrt(self._den)
        chi = minimize(energy, chi, method='gd', maxiter=0)

        if output == 'energy':
            E = energy(chi)
            self.detach()  # reset autograd features
            return E
        elif output == 'density':
            N_tilde = torch.mean(chi.square()) * self._vol()
            return (self.N_elec / N_tilde) * chi.square()

    def __compute_volume_derivatives(self, bulk_modulus=True):
        box_vecs = self._box_vecs.clone()
        vol = self._vol()
        vol.requires_grad = True

        def energy(chi, vol):
            # incorporate volume dependence in lattice vectors and hence ionic potential
            self._box_vecs = box_vecs * (vol / vol.detach()).pow(1 / 3)
            self._v_ext = self._potential_from_ions(torch.matmul(self._frac_ion_coords, self._box_vecs))
            N_tilde = torch.mean(chi.square()) * vol
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy()
            return E

        # use Xitorch's minimize to get "minimization functional" derivative
        chi = torch.sqrt(self._den)
        if bulk_modulus:
            chi = minimize(energy, chi, params=(vol,), method='gd', maxiter=0)
        E = energy(chi, vol)

        # calculate bulk modulus by auto-differentiating energy wrt volume
        dEdV = torch.autograd.grad(E, vol, create_graph=True)[0]
        if bulk_modulus:
            d2EdV2 = torch.autograd.grad(dEdV, vol)[0]
            K = self._vol().detach() * d2EdV2

        self.detach()  # reset autograd features

        if bulk_modulus:
            return dEdV.neg().item(), K.item()
        else:
            return dEdV.neg().item()

    def _compute_elastic_constants(self):

        box_vecs = self._box_vecs.clone()

        def energy(chi, voigt_strain):
            deformation = (torch.eye(3, dtype=voigt_strain.dtype, device=voigt_strain.device)
                           .add_(_voigt_to_3by3_strain(voigt_strain)))
            self._box_vecs = torch.matmul(box_vecs, deformation)
            self._v_ext = self._potential_from_ions(torch.matmul(self._frac_ion_coords, self._box_vecs))
            N_tilde = torch.mean(chi.square()) * self._vol()
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy()
            return E

        voigt_strain = torch.zeros((6,),
                                   dtype=self._box_vecs.dtype,
                                   device=self._box_vecs.device,
                                   requires_grad=True)

        # use Xitorch's minimize to get "minimization functional" derivative
        chi_init = torch.sqrt(self._den)
        chi_opt = minimize(energy, chi_init, params=(voigt_strain,), method='gd', maxiter=0)
        E = energy(chi_opt, voigt_strain)

        # calculate elastic constants by auto-differentiating energy wrt voigt strain
        voigt_stress = torch.autograd.grad(E, voigt_strain, create_graph=True)[0].div(self._vol())
        Cs = torch.stack([torch.autograd.grad(voigt_stress[i],
                                              voigt_strain,
                                              retain_graph=True)[0] for i in range(6)], -1)
        self.detach()  # reset autograd features
        return Cs

    def _compute_force_constants(self, primitive_ion_indices):
        cart_ion_coords = torch.matmul(self._frac_ion_coords, self._box_vecs)
        cart_ion_coords.requires_grad = True

        def energy(chi, cart_ion_coords):
            # incorporate ionic coordinate dependence in ionic potential
            self._v_ext = self._potential_from_ions(cart_ion_coords)
            N_tilde = torch.mean(chi.square()) * self._vol()
            self._den = (self.N_elec / N_tilde) * chi.square()
            E = self._compute_energy(for_den_opt=True) + self._ion_ion_interaction(cart_ion_coords)
            return E

        # use Xitorch's minimize to get "minimization functional" derivative
        chi_init = torch.sqrt(self._den)
        chi_opt = minimize(energy, chi_init, params=(cart_ion_coords,), method='gd', maxiter=0)
        E = energy(chi_opt, cart_ion_coords)

        # calculate force constants by auto-differentiating energy wrt ionic coordinates
        forces = torch.autograd.grad(E, cart_ion_coords, create_graph=True)[0]  # no neg
        force_constants = torch.empty((len(primitive_ion_indices), self.N_ion, 3, 3),
                                      dtype=torch.double, device=self._device)
        for pion in primitive_ion_indices:
            for i in range(3):
                force_constant = torch.autograd.grad(forces[pion, i], cart_ion_coords, retain_graph=True)[0]
                force_constants[pion, :, i, :] = force_constant

        self.detach()  # reset autograd features
        return force_constants


# -------------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------------

def _voigt_to_3by3_strain(voigt_strain: torch.Tensor):
    """
    Autograd compatible function that has the same effect as the following:

    return torch.tensor([[voigt_strain[0], 0.5 * voigt_strain[5], 0.5 * voigt_strain[4]],
                         [0.5 * voigt_strain[5], voigt_strain[1], 0.5 * voigt_strain[3]],
                         [0.5 * voigt_strain[4], 0.5 * voigt_strain[3], voigt_strain[2]]],
                         dtype=voigt_strain.dtype, device=voigt_strain.device)
    """
    ids = torch.tensor([0, 5, 4, 1, 3, 2], dtype=torch.long, device=voigt_strain.device)
    sorted_vstrain = voigt_strain[ids]
    id1, id2 = torch.triu_indices(3, 3)
    strain = torch.zeros((3, 3), dtype=voigt_strain.dtype, device=voigt_strain.device)
    strain[id1, id2] = sorted_vstrain
    return 0.5 * (strain + strain.T)


def _voigt_to_3by3_stress(voigt_stress: torch.Tensor):
    return torch.tensor([[voigt_stress[0], voigt_stress[5], voigt_stress[4]],
                         [voigt_stress[5], voigt_stress[1], voigt_stress[3]],
                         [voigt_stress[4], voigt_stress[3], voigt_stress[2]]],
                         dtype=voigt_stress.dtype, device=voigt_stress.device)
