Density Optimization
====================

Introduction
------------

The core functionality of any orbital-free density functional theory code is to be able to 
perform density optimizations. In theory, we seek to minimize the energy functional subject 
to the constraints that the electron density is positive and normalized. We can hence form a
Lagrangian,

.. math::  \mathcal{L}[n] = E[n] - \mu \left[\int n(\mathbf{r}) d^3\mathbf{r} - N_e \right],

which yields an Euler-Lagrange equation 

.. math:: \mu - \frac{\delta E}{\delta n(\mathbf{r})} \Bigg\vert_{n_\text{gs}} = 0,

to be solved for the ground state density. While introduced as a Lagrange multiplier constant, 
:math:`\mu` can be interpreted as the chemical potential based on the Euler-Lagrange equation. 

PROFESS-AD adopts a direct minimization scheme to solve the Euler-Lagrange equation.
To account for the density positivity and normalization constraints during the optimization, 
PROFESS-AD minimizes the energy with respect to an unconstrained variable :math:`\chi(\mathbf{r})`
instead, which is related to the constrained density :math:`n(\mathbf{r})` via

.. math:: n(\mathbf{r}) = \frac{N_e~\chi^2(\mathbf{r})}{\int~d^3\mathbf{r}'\chi^2(\mathbf{r}')},

where :math:`N_e` is the number of electrons. The convergence condition is hence

.. math:: \frac{\delta E}{\delta \chi} \Bigg\vert_{\chi_\text{gs}} = 0

By default, PROFESS-AD utilizes a PyTorch-based LBFGS optimizer to perform density optimizations. 
The default convergence criteria is that the energy difference after each iteration must be below
1e-7 eV (controlled by ``ntol``), 3 times in a row (controlled by ``n_conv_cond_count``). Other 
parameter options can be found in the description of the ``system.optimize_density()`` method in the 
:doc:`system` page.

Basic Example
-------------
Let us consider a basic example with the following code ::

  # set energy terms and functionals to be used
  terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]

  # use "get cell" to get the lattice vectors and fractional ionic coordinates of
  # a face-centred cubic (fcc) lattice
  box_vecs, frac_ion_coords = get_cell('fcc', vol_per_atom=24.8, coord_type='fractional')

  # defining the ions in the system
  ions = [['Al', 'al.gga.recpot', frac_ion_coords]]

  # set plane-wave cutoff at 2000 eV
  shape = System.ecut2shape(2000, box_vecs)

  # create an fcc-aluminium system object
  system = System(box_vecs, shape, ions, terms, units='a', coord_type='fractional')

  # perform density optimization (by default n_verbose is False, but we want to
  # display the progress of the density optimization)
  system.optimize_density(ntol=1e-7, conv_target='dE', n_method='LBFGS', n_verbose=True)

  # check the measures of convergence
  dEdchi_max = system.check_density_convergence('dEdchi')
  mu_minus_dEdn_max = system.check_density_convergence('euler')

  print('Convergence check:')
  print('Max |𝛿E/𝛿χ| = {:.4g}'.format(dEdchi_max))
  print('Max |µ-𝛿E/𝛿n| = {:.4g}'.format(mu_minus_dEdn_max))


This sets up a face-centred cubic (fcc) aluminium system and performs a density optimization. 
With the default settings, termination of the density optimization is based on the energy 
differences ``dE``. We can also use the other measures of convergence (which correspond to the
extent that the Euler-Lagrange equations are obeyed) to make sure that the density has been
optimized to the desired extent.  ::

  Starting density optimization
    Iter      E [eV]      dE [eV]       Max |𝛿E/𝛿χ|       Max |µ-𝛿E/𝛿n|
     0       9.677157        0            0.865837           3.23348
     1       8.403218     -1.27394        0.473481           2.20042
     2       7.702040    -0.701178        0.550324           3.37185
     3       7.548173    -0.153867        0.211747           1.38436
     4       7.507185    -0.0409875       0.107438           0.771551
     5       7.485764    -0.0214207       0.089752           0.769107
     6       7.458289    -0.0274754      0.0758372           0.628908
     7       7.429842    -0.0284467      0.0384012           0.223996
     8       7.421262   -0.00857984      0.0292648           0.124719
     9       7.418701   -0.00256096      0.0197278           0.106185
     10      7.417557   -0.00114492      0.0106644          0.0758516
     11      7.417031   -0.000525209     0.00736959         0.0539031
     12      7.416772   -0.000259712     0.00520798         0.0398683
     13      7.416638   -0.000133214     0.00248938         0.0136218
     14      7.416570   -6.88236e-05     0.00358424          0.018048
     15      7.416518   -5.1554e-05      0.00181108         0.00946193
     16      7.416493   -2.47783e-05     0.00164527         0.00834793
     17      7.416479   -1.46257e-05     0.00139808         0.00779606
     18      7.416465   -1.38169e-05     0.00105684         0.0047043
     19      7.416458   -6.52723e-06     0.00103864         0.00740266
     20      7.416455   -3.72415e-06    0.000876987         0.00351695
     21      7.416452   -2.1472e-06     0.000503422         0.00219884
     22      7.416451   -9.56176e-07    0.000351898         0.00153222
     23      7.416451   -2.24164e-07    0.000156075        0.000990379
     24      7.416451   -2.25588e-08    0.000131899        0.000806672
     25      7.416451   -1.69511e-08    0.000139933        0.000889683
     26      7.416451   -1.92808e-08    9.49472e-05        0.000544465
  Density optimization successfully converged in 26 step(s)

  Convergence check:
  Max |𝛿E/𝛿χ| = 9.424e-05
  Max |µ-𝛿E/𝛿n| = 0.0005445

Example with Custom Potentials (Quantum Harmonic Oscillator)
------------------------------------------------------------

Let us consider a use case where we want to use a custom external potential for the
density optimization. The procedure is similar to the above, just that we have to 
supply a dummy ``ions`` parameter to initialize the ``system`` object before setting the
potential to our desired one, remembering to change the electron number of the system too. :: 

  # the single electron quantum harmonic oscillator (QHO) is a non-interacting
  # single-orbital system - hence, it can be modelled well with just the 
  # ion-electron interaction and Weizsaecker terms
  terms = [IonElectron, Weizsaecker]

  # use a large box to simulate such localied systems with periodic
  # boundary conditions so that the electron density will approach zero
  # at the box boundaries
  L = 20.0
  box_vecs = L * torch.eye(3, dtype=torch.double)

  # set low energy cutoff of 300 eV
  shape = System.ecut2shape(300, box_vecs)

  # as we will set the external potential ourselves later, we just need to
  # submit a dummy "ions" parameter (the recpot file and ionic coordinates 
  # are arbitrary for this example)
  ions = [['-', 'al.gga.recpot', torch.tensor([[0.5, 0.5, 0.5]]).double()]]

  # create system object
  system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional')

  # as we have used an arbitrary recpot file, we need to set the electron number explicitly
  system.set_electron_number(1)

  # QHO quadratic potential
  k = 10
  xf, yf, zf = np.meshgrid(np.arange(shape[0]) / shape[0], np.arange(shape[1]) / shape[1],
                           np.arange(shape[2]) / shape[2], indexing='ij')
  x = box_vecs[0, 0] * xf + box_vecs[1, 0] * yf + box_vecs[2, 0] * zf
  y = box_vecs[0, 1] * xf + box_vecs[1, 1] * yf + box_vecs[2, 1] * zf
  z = box_vecs[0, 2] * xf + box_vecs[1, 2] * yf + box_vecs[2, 2] * zf
  r = np.sqrt(x * x + y * y + z * z)
  qho_pot = 0.5 * k * ((x - L / 2).pow(2) + (y - L / 2).pow(2) + (z - L / 2).pow(2))

  # set external potential to QHO potential
  system.set_potential(torch.as_tensor(qho_pot).double())

  # perform density optimization
  system.optimize_density(ntol=1e-7, n_verbose=True)

  # compare optimized energy and the ones expected from elementary quantum mechanics
  print('Optimized energy = {:.8f} Ha'.format(system.energy('Ha')))
  print('Expected energy = {:.8f} Ha'.format(3 / 2 * np.sqrt(k)))

  # check measures of convergence
  dEdchi_max = system.check_density_convergence('dEdchi')
  mu_minus_dEdn_max = system.check_density_convergence('euler')
  print('\nConvergence check:')
  print('Max |𝛿E/𝛿χ| = {:.4g}'.format(dEdchi_max))
  print('Max |µ-𝛿E/𝛿n| = {:.4g}'.format(mu_minus_dEdn_max))


This results in ::

  Starting density optimization
    Iter      E [eV]      dE [eV]       Max |𝛿E/𝛿χ|       Max |µ-𝛿E/𝛿n|
     0     13613.510241      0            22.3543            999.713
     1     6558.513125     -7055          14.9388            60922.2
     2     3187.559139    -3370.95        10.6157            9693.09
     3     1626.061603    -1561.5         20.9591            10707.9
     4      889.606308    -736.455        6.05023            36347.1
     5      548.774621    -340.832        4.41879            16886.9
     6      376.605141    -172.169        3.18798            30169.6
     7      286.048990    -90.5562        13.3591          8.54989e+06
     8      230.137857    -55.9111        5.08921            14889.6
     9      193.892397    -36.2455        4.23115            21310.5
     10     169.331505    -24.5609        3.55797            10083.9
     11     152.787931    -16.5436        1.00027             269847
     12     143.654847    -9.13308        2.72997            56765.4
     13     135.496019    -8.15883        1.54196            91377.3
     14     132.008561    -3.48746         1.3482          2.23426e+06
     15     130.441073    -1.56749        0.514763            407990
     16     129.689363    -0.75171        1.57811             129319
     17     129.298829   -0.390534        0.993079           55204.9
     18     129.166347   -0.132481        0.241797            36499
     19     129.113693   -0.0526547       0.114015            932106
     20     129.094428   -0.0192641       0.104126            462777
     21     129.083642   -0.0107861       0.192647            134088
     22     129.078364  -0.00527784      0.0367585            204466
     23     129.076561  -0.00180324      0.0659604            221888
     24     129.075609  -0.000952299     0.0215513           67498.4
     25     129.075250  -0.000358949     0.0398467            559964
     26     129.075041  -0.000208968     0.0111308            471584
     27     129.074965  -7.56737e-05     0.00604273           307790
     28     129.074947  -1.8423e-05      0.00204187           104320
     29     129.074942  -5.12278e-06     0.00102419           459483
     30     129.074940  -1.90488e-06    0.000324086        1.26339e+06
     31     129.074939  -7.6968e-07     0.000166146        1.66163e+06
     32     129.074939  -3.65745e-07    0.000154676        1.66135e+07
     33     129.074939  -1.66777e-07    9.27869e-05           223466
     34     129.074939  -2.56686e-08    0.000148262           340946
     35     129.074939  -2.22304e-08     0.00010925        1.36216e+06
     36     129.074939  -2.2209e-08      0.00016803           592491
  Density optimization successfully converged in 36 step(s)

  Optimized energy = 4.74341650 Ha
  Expected energy = 4.74341649 Ha

  Convergence check:
  Max |𝛿E/𝛿χ| = 0.0007487
  Max |µ-𝛿E/𝛿n| = 5.925e+05
  
Note how the Max :math:`|\mu - \delta E / \delta n|` measure of convergence becomes very
large at convergence eventhough the optimized energy agrees with the theoretical one and 
Max :math:`|\delta E / \delta \chi|` measure of convergence is small. This is a result of 
the vacuum regions with low densities leading to divergences in the von Weizsaecker term.

