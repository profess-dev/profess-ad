Elastic Properties
==================

Equation of State Fit (Basic)
-----------------------------

Equation of state fits have become the most common way to assess the quality of new kinetic
functionals. PROFESS-AD provides a simplified workflow to perform such fits, using the 
``system.eos_fit()`` class method. First, let us consider a "bare-bones" example. ::

  # define the system at a close estimate to the equilibrium volume
  terms = [IonIon, IonElectron, Hartree, vWGTF1, PerdewBurkeErnzerhof]
  box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.9, coord_type='fractional')
  ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
  shape = System.ecut2shape(2000, box_vecs)
  system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')

  # perform the equation of state (EOS) fit

  # "f" is the fraction of the equilibrium volume by which the system is stretched or squeezed by
  # "N" is the number of energy-volume points used for the EOS fit
  # "ntol" is a density optimization tolerace argument
  # "eos" specifies the equation of state used for the fit, it can either be 'bm' for Birch-Murnaghan
  # or 'm' for Murnaghan
  params, err = system.eos_fit(f=0.05, N=11, ntol=1e-7, eos='bm')
  K0, K0prime, E0, V0 = params  # unpack params

  print('Bulk modulus, K₀ = {:.5g} GPa'.format(K0))
  print('Bulk modulus derivative (wrt pressure), K₀\' = {:.5g}'.format(K0prime))
  print('Equilibrium energy, E₀ = {:.5g} eV per atom'.format(E0))
  print('Equilibrium volume, V₀ = {:.5g} A³ per atom'.format(V0))

This results in ::

  Bulk modulus, K₀ = 87.821 GPa
  Bulk modulus derivative (wrt pressure), K₀' = 4.2268
  Equilibrium energy, E₀ = -57.231 eV per atom
  Equilibrium volume, V₀ = 16.86 A³ per atom

If one does not have a good initial estimate for the volume, one solution is to increase the value of
the ``f`` parameter of ``system.eos_fit()`` to a larger value (e.g. ``f=0.3``) to try and capture the
energy-volume minima. One can then use the equilibrium volume computed from that fit with a smaller ``f`` 
to fine-tune the results.

Equation of State Fit (Advanced)
--------------------------------
The ``system.eos_fit()`` convenience method can easily be used for rapid testing of functionals. Take the
following for example. ::

  f = 0.05
  N = 11
  energy_cutoff = 2000  # eV
  tol = 1e-7

  print('Performing aluminium calculations with Wang-Teter KE and PBE XC functionals.')
  print('Energy cut-off = {} eV with density optimization tolerance {}'.format(energy_cutoff, tol))
  print('Stretched up to ±{}% over {} points and fitted with the {} EOS\n'
        .format(f * 100, N, 'Birch-Murnaghan'))

  terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
  crystal_predv0s = [('fcc', 16.8), ('hcp', 16.9), ('bcc', 17.2), ('sc', 19.9), ('dc', 28.8)]

  print('{:^8} {:^17} {:^17} {:^14} {:^14}'.format('Crystal', 'V₀/A³ per atom', 'E₀/eV per atom', 'ΔE₀/meV', 'K₀/GPa'))
  for crystal, pred_v0 in crystal_predv0s:
      box_vecs, frac_ion_coords = get_cell(crystal, vol_per_atom=pred_v0, c_over_a=1.66, coord_type='fractional')
      ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
      shape = System.ecut2shape(energy_cutoff, box_vecs)
      system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')
      params, err = system.eos_fit(f=f, N=N, ntol=tol, eos='bm')
      K0, K0prime, E0, V0 = params
      if crystal == 'fcc':
          E_fcc = E0
      print('{:^8} {:^17.5f} {:^17.5f} {:^14.2f} {:^14.5f}'.format(crystal, V0, E0, (E0 - E_fcc) * 1e3, K0))

This results in ::

  Performing aluminium calculations with Wang-Teter KE and PBE XC functionals.
  Energy cut-off = 2000 eV with density optimization tolerance 1e-07
  Stretched up to ±5.0% over 11 points and fitted with the Birch-Murnaghan EOS

  Crystal   V₀/A³ per atom    E₀/eV per atom      ΔE₀/meV         K₀/GPa
    fcc        16.76389          -57.18370          0.00         78.80961
    hcp        16.87622          -57.16592         17.78         77.00603
    bcc        17.16419          -57.11107         72.63         71.66677
     sc        19.88597          -56.87121         312.48        57.53359
     dc        28.78790          -56.39261         791.09        23.52562


Elastic Constants
-----------------

.. _elastic_constants_example:

With the use of ξ-torch, higher derivative quantities, such as the bulk modulus and second-order elastic 
constants, can be computed directly with auto-differentiation. The following example computes the zero-pressure
elastic properties of fcc-aluminium. Due to its cubic symmetry, this system only possesses three 
independent elastic constant elements, :math:`C_{11}`, :math:`C_{12}` and :math:`C_{44}`.

We first performs a Birch-Murnaghan equation of state fit to find the equilibrium volume before 
computing the zero-pressure elastic constants. Comparions of the bulk moduli obtained via different 
approaches and an example of how the elastic constants can be post-processed are also included. 
More post-processing options are listed in the :ref:`Elastic Properties <elastic_properties>` section 
of the :doc:`elastic_tools` page.  ::

  # define system
  terms = [IonIon, IonElectron, Hartree, XuWangMa, PerdewBurkeErnzerhof]
  box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=16.52, coord_type='fractional')
  ions = [['Al', 'al.gga.recpot', frac_ion_coords]]
  shape = System.ecut2shape(2000, box_vecs)
  system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')

  # perform Birch-Murnaghan fit to determine equilibrium volume and bulk modulus
  params, err = system.eos_fit(f=0.05, N=11, ntol=1e-10, eos='bm')
  K0, K0prime, E0, V0 = params

  print('Birch-Murnaghan fit results:')
  print('Equilibrium volume = {:.5g} Å³'.format(V0))
  print('Equilibrium bulk modulus = {:.5g} GPa'.format(K0))

  # set system to equilibrium volume
  box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=V0, coord_type='fractional')
  system.set_lattice(box_vecs, units='a')

  # use higher density optimization tolerance for second-derivative calculations
  system.optimize_density(ntol=1e-10)

  # check if pressure is zero
  pressure = system.pressure('GPa')
  print('Pressure = {:.5g} GPa (expect zero pressure at equilibrium volume)'.format(pressure))

  # get auto-differentiated elastic constants
  Cs = system.elastic_constants('GPa')

  print('\nElastic constants from auto-differentiation:')
  print('C11 = {:.5g} GPa'.format(Cs[0, 0].item()))
  print('C12 = {:.5g} GPa'.format(Cs[0, 1].item()))
  print('C44 = {:.5g} GPa'.format(Cs[3, 3].item()))

  # compute bulk modulus from elastic constants
  # for cubic systems, K = (C11 + 2 * C12) / 3
  K_ec = (Cs[0, 0].item() + 2 * Cs[0, 1].item()) / 3

  # get auto-differentiated bulk modulus
  K_ad = system.bulk_modulus('GPa')

  print('\nCompare bulk moduli:')
  print('EOS bulk modulus = {:.5g} GPa'.format(K0))
  print('Auto-differentiation bulk modulus = {:.5g} GPa'.format(K_ad))
  print('Bulk modulus from elastic constants = {:.5g} GPa'.format(K_ec))

  # post-process elastic constants matrix to shear modulus and poisson's ratio
  G = shear_average(Cs, mean_type='arithmetic')
  v = poissons_ratio(K_ec, G)

  print('\nPost-processed quantities:')
  print('Shear modulus = {:.5g}'.format(G))
  print('Poisson\'s ratio = {:.5g}'.format(v))

This results in ::

  Birch-Murnaghan fit results:
  Equilibrium volume = 16.524 Å³
  Equilibrium bulk modulus = 76.494 GPa
  Pressure = 0.0021408 GPa (expect zero pressure at equilibrium volume)

  Elastic constants from auto-differentiation:
  C11 = 107.08 GPa
  C12 = 61.215 GPa
  C44 = 37.861 GPa

  Compare bulk moduli:
  EOS bulk modulus = 76.494 GPa
  Auto-differentiation bulk modulus = 76.502 GPa
  Bulk modulus from elastic constants = 76.502 GPa
  
  Post-processed quantities:
  Shear modulus = 30.963
  Poisson's ratio = 0.32169
