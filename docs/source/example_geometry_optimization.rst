Geometry Optimization
=====================

Conventional Geometry Optimization
----------------------------------

The following code is an example of how a geometry optimization can be set up for body-centred cubic
lithium (bcc-Li). This example is meant to illustrate how one can perform different types of geometry 
optimizations, including ones where 

* the lattice is fixed but the ions can move,
* the lattice can deform but the fractional ionic coordinates are fixed, and
* both the lattice and ions can be changed  

::

  import torch

  from system import System
  from functionals import IonIon, IonElectron, Hartree, WangTeterStyleFunctional, PerdewBurkeErnzerhof


  # create system and compute ground state energy
  box_len = 3.48
  box_vecs = box_len * torch.eye(3, dtype=torch.double)
  shape = System.ecut2shape(800, box_vecs)
  ions = [['Li', 'li.gga.recpot', box_len * torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]).double()]]
  WTexp = WangTeterStyleFunctional((5 / 6, 5 / 6, lambda x: torch.exp(x)))
  terms = [IonIon, IonElectron, Hartree, WTexp.forward, PerdewBurkeErnzerhof]
  system = System(box_vecs, shape, ions, terms, units='a')

  system.optimize_density(1e-10)
  energy = system.energy('eV')
  print('Initial Energy = {:.4f} eV per atom'.format(energy / system.ion_count()))

  # OPTIMIZE IONIC POSITIONS / MINIMIZE FORCES

  # peturb ions
  print('Perturbing ions ...')
  system.place_ions(box_len * torch.tensor([[0.0, 0.1, 0.0], [0.6, 0.4, 0.6]], dtype=torch.double), units='a')
  system.optimize_density(1e-10)
  print('Perturbed energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

  # restore optimal ionic positions by minimizing forces, keeping lattice fixed
  print('Performing force minimization ...')
  system.optimize_geometry(stol=None, ftol=1e-4, g_method='LBFGSlinesearch', g_verbose=True)
  print('Optimized Energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

  # OPTIMIZE LATTICE / MINIMIZE STRESS

  # predict relaxed energy by fitting to murnaghan equation
  print('\nPerforming EOS fit for equilibrium volume ...')
  params, err = system.eos_fit(N=5)
  relaxed_energy = system.ion_count() * params[2]
  print('Equilibrium energy = {:.4f} eV per atom'.format(relaxed_energy / system.ion_count()))

  # distort lattice
  print('Deforming lattice ...')
  tm = torch.tensor([[0.94, -0.03, 0.05],
                     [-0.03, 0.98, 0.04],
                     [0.05, 0.04, 1.05]], dtype=torch.double)
  system.set_lattice(torch.matmul(tm, system.lattice_vectors('a').T).T, units='a')
  system.optimize_density(1e-10)
  print('Perturbed energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

  # relax by minimizing stress, keeping box coordinates of ions fixed
  print('Performing stress minimization ...')
  system.optimize_geometry(ftol=None, stol=1e-4, g_method='LBFGSlinesearch', g_verbose=True)
  print('Optimized Energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

  # OPTIMIZE GEOMETRY / MINIMIZE FORCES AND STRESS

  print('\nPerturbing overall geometry ...')
  # peturb ions and distort lattice
  tm = torch.tensor([[0.94, -0.03, 0.05],
                     [-0.03, 0.98, 0.04],
                     [0.05, 0.04, 1.05]], dtype=torch.double)
  system.place_ions(torch.matmul(tm, system.cartesian_ionic_coordinates('a').T).T, units='a')
  system.set_lattice(torch.matmul(tm, system.lattice_vectors('a').T).T, units='a')
  system.optimize_density(1e-10)
  print('Perturbed energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))

  # restore optimal geometry by minimizing forces and stress
  print('Performing geometry optimization ...')
  system.optimize_geometry(stol=1e-4, ftol=1e-4, g_method='LBFGSlinesearch', g_verbose=True)
  print('Optimized Energy = {:.4f} eV per atom'.format(system.energy('eV') / system.ion_count()))


This code results in the following output. ::

  Initial Energy = -7.3578 eV per atom
  Perturbing ions ...
  Perturbed energy = -7.1217 eV per atom
  Performing force minimization ...
   Iter     E [eV per atom]      dE [eV per atom]     Max Force [eV/Å]    Max Stress [eV/Å³]
     0         -7.121696                0                 0.842513            0.0185083
     1         -7.310906            -0.189211             0.386264            0.00338438
     2         -7.345332            -0.0344256            0.200588            0.00365126
     3         -7.354303           -0.00897128            0.106939            0.00374702
     4         -7.356813           -0.00251014           0.0572883            0.00379373
     5         -7.357531           -0.000717481          0.0307186            0.00380226
     6         -7.357738           -0.000206621          0.0164658            0.00380478
     7         -7.357797           -5.91924e-05          0.00882966           0.00380545
     8         -7.357814           -1.7049e-05           0.00473394           0.00380565
     9         -7.357819           -4.90418e-06          0.00256241           0.00380571
    10         -7.357820           -1.42075e-06          0.0013795            0.00380573
    11         -7.357821           -4.08516e-07         0.000735595           0.00380574
    12         -7.357821           -1.16475e-07         0.000396698           0.00380574
    13         -7.357821           -3.37541e-08         0.000208924           0.00380575
    14         -7.357821           -9.54252e-09         0.000110661           0.00380575
    15         -7.357821           -4.41454e-09         1.77384e-05           0.00380575
    16         -7.357821           -4.93685e-11         1.77835e-05           0.00380575
    17         -7.357821           -3.02247e-11          1.7895e-05           0.00380575
  Geometry optimization successfully converged in 17 step(s)

  Optimized Energy = -7.3578 eV per atom

  Performing EOS fit for equilibrium volume ...
  Equilibrium energy = -7.3595 eV per atom
  Deforming lattice ...
  Perturbed energy = -7.3351 eV per atom
  Performing stress minimization ...
   Iter     E [eV per atom]      dE [eV per atom]     Max Force [eV/Å]    Max Stress [eV/Å³]
     0         -7.335086                0                0.00012742           0.00944689
     1         -7.358675            -0.0235888          8.01787e-05           0.0015873
     2         -7.358963           -0.000288282          5.3544e-05           0.00115537
     3         -7.359343           -0.000380017         5.01887e-05          0.000458943
     4         -7.359427           -8.35168e-05         4.60166e-05          0.000255862
     5         -7.359440           -1.35338e-05         4.27197e-05          0.000363761
     6         -7.359472            -3.224e-05          4.11179e-05          0.000176818
     7         -7.359483           -1.10674e-05         3.99874e-05          8.06105e-05
     8         -7.359487           -3.86039e-06         3.92089e-05          4.99504e-05
     9         -7.359489           -2.03904e-06         3.85918e-05          3.05821e-05
  Geometry optimization successfully converged in 9 step(s)

  Optimized Energy = -7.3595 eV per atom

  Perturbing overall geometry ...
  Perturbed energy = -7.3068 eV per atom
  Performing geometry optimization ...
   Iter     E [eV per atom]      dE [eV per atom]     Max Force [eV/Å]    Max Stress [eV/Å³]
     0         -7.306823                0                 0.396871            0.0116692
     1         -7.335214            -0.0283912            0.26266             0.00706001
     2         -7.353728            -0.0185138            0.126866            0.00332661
     3         -7.357243           -0.00351547           0.0623917            0.00269824
     4         -7.358423           -0.00118024           0.0434388           0.000865097
     5         -7.358744           -0.000321098          0.0213561           0.000882875
     6         -7.359002           -0.000257791          0.0111404           0.000413472
     7         -7.359038           -3.54953e-05          0.00549463          0.000416077
     8         -7.359083           -4.53939e-05          0.00307135          0.000412545
     9         -7.359114           -3.08373e-05          0.00150737          0.000655856
    10         -7.359122           -7.70214e-06         0.000644913          0.000701684
    11         -7.359159           -3.71719e-05          0.0012657           0.000468603
    12         -7.359179           -2.04144e-05         0.000537675          0.000531482
    13         -7.359198           -1.87877e-05         0.000983285          0.000464925
    14         -7.359246           -4.82697e-05         0.000741491          0.000537872
    15         -7.359276            -2.935e-05          0.000590132          0.000664748
    16         -7.359330           -5.46165e-05         0.000638868          0.000601527
    17         -7.359360           -2.92485e-05          0.00941012          0.000724566
    18         -7.359400           -4.03731e-05          0.00473564          0.000584351
    19         -7.359438           -3.81269e-05          0.00231378          0.000216816
    20         -7.359450           -1.2374e-05          0.000602998           0.00028038
    21         -7.359474           -2.3748e-05          0.000323331           0.00011232
    22         -7.359479           -4.69937e-06         0.000132017          0.000117203
    23         -7.359484           -4.73172e-06         0.000899799          0.000158323
    24         -7.359486           -2.43998e-06         0.000292098           0.00015559
    25         -7.359486           -1.38117e-07         0.000260749          8.92468e-05
    26         -7.359489           -2.7876e-06          0.000147261          6.82452e-05
    27         -7.359491           -1.57216e-06         0.000107321          4.29506e-05
    28         -7.359488           2.63341e-06          8.10592e-05          4.74667e-05
    29         -7.359490           -2.3244e-06          5.92544e-05          3.53178e-05
    30         -7.359492           -2.08458e-06         4.42381e-05          1.96232e-05
  Geometry optimization successfully converged in 30 step(s)

  Optimized Energy = -7.3595 eV per atom



Parameterized Geometry Optimization
-----------------------------------

The following code is an example of how a parameterized geometry optimization can be set up for hexagonal close-packed
magnesium (hcp-Mg). ::

  import numpy as np
  import torch

  from system import System
  from functionals import IonIon, IonElectron, Hartree, WangTeterStyleFunctional, PerdewBurkeErnzerhof

  # use GPU if available else use CPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  params = torch.tensor([24 / System.A_per_b**3, 1.5], dtype=torch.double, device=device).requires_grad_()
  print('Initial Guess: Volume per atom = {:.5f} Å³, c/a = {:.5f}'
        .format(params[0].item() * System.A_per_b**3, params[1].item()))


  # define the lattice vectors and fractional ionic coordinates as a function of the parameters
  def parameterized_geometry(params):
      vol_per_atom, c_over_a = params
      a = ((2 * torch.abs(vol_per_atom)) / (np.sqrt(3) / 2 * c_over_a)).pow(1 / 3)
      box_vecs = torch.tensor([[1.0, 0.0, 0.0],
                               [-0.5, np.sqrt(3) / 2, 0.0],
                               [0.0, 0.0, 0.0]], dtype=torch.double, device=device)
      box_vecs[2, 2] = torch.abs(c_over_a)
      box_vecs = a * box_vecs
      frac_ion_coords = torch.tensor([[1 / 3, 2 / 3, 3 / 4],
                                      [2 / 3, 1 / 3, 1 / 4]], dtype=torch.double, device=device)
      return box_vecs, frac_ion_coords


  box_vecs, frac_ion_coords = parameterized_geometry(params)

  WTexp = WangTeterStyleFunctional((5 / 6, 5 / 6, lambda x: torch.exp(x)))
  # required for GPU usage with functionals that inherit from the KineticFunctional class
  WTexp.set_device(device)
  terms = [IonIon, IonElectron, Hartree, WTexp.forward, PerdewBurkeErnzerhof]

  # construct the system object
  ions = [['Mg', 'mg.gga.recpot', frac_ion_coords.detach()]]
  # lattice vectors must be in angstroms for ecut2shape
  shape = System.ecut2shape(2000, box_vecs.detach() * System.A_per_b)
  system = System(box_vecs, shape, ions, terms, units='b', coord_type='fractional', device=device)


  # define a print statement to track how the parameters evolve over the optimization
  def param_string(params):
      return '{:.5f} {:.5f}'.format(params[0].item() * System.A_per_b**3, params[1].item())


  system.optimize_parameterized_geometry(params, parameterized_geometry, g_method='LBFGSlinesearch',
                                         g_verbose=True, param_string=param_string, ftol=1e-3, stol=1e-3)
  print('Optimized Results: Volume per atom = {:.5f} Å³, c/a = {:.5f}\n'
        .format(params[0].item() * System.A_per_b**3, params[1].item()))



This code results in the following output. ::

  Initial Guess: Volume per atom = 24.00000 Å³, c/a = 1.50000
   Iter     E [eV per atom]      dE [eV per atom]     Max Force [eV/Å]    Max Stress [eV/Å³] Params
     0         -24.198576               0               3.87304e-06           0.0138319      24.00000 1.50000
     1         -24.208906           -0.0103299          4.73783e-06            0.01135       23.99641 1.55999
     2         -24.214733          -0.00582784          4.90708e-06           0.00547138     23.48824 1.59311
     3         -24.216084          -0.00135011          5.02009e-06           0.00247967     23.28204 1.61115
     4         -24.216418          -0.000333999         5.09427e-06           0.00116472     23.21050 1.62080
     5         -24.216488          -7.04247e-05         5.12923e-06          0.000593536     23.16811 1.62585
     6         -24.216517          -2.87005e-05         5.15291e-06          0.000389594     23.15198 1.62861
     7         -24.216529          -1.2568e-05          5.17015e-06          0.000188783     23.15116 1.63011
  Geometry optimization successfully converged in 7 step(s)

  Optimized Results: Volume per atom = 23.15116 Å³, c/a = 1.63011

