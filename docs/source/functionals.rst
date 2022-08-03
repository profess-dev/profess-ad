Energy Terms and Functionals
============================

This page lists the energy terms and functionals that have been implemented. They can be included
in a list of energy terms in the ``terms`` argument that goes into the initialization of a :doc:`system` object.

Take the following for example :: 

  terms = [IonIon, IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof]
  ...
  system = System(box_vecs, shape, ions, terms)

Note that apart from the ``IonIon`` and ``IonElectron`` functions, the other functions must only take in inputs of 
``box_vecs`` and ``den``, which represent the lattice vectors and elecron density respectively, i.e.
``functional(box_vecs, den)``.

For example, the :ref:`Xu-Wang-Ma functional <xwm>` was implemented to take in an
additional argument that acts as a functional parameter, i.e. ``XuWangMa(box_vecs, den, kappa)``. A simple way
to handle this is to use lambda functions. ::

  terms = [IonIon, IonElectron, Hartree, lambda bv, n: XuWangMa(bv, n, 0), PerdewZunger]
  ...
  system = System(box_vecs, shape, ions, terms)

Some kinetic functionals are represented by classes that inherit from the ``torch.nn.Module`` class, in which
case it is their ``.forward()`` method that should be included in the list of terms. For example, ::

  pg = PauliGaussian()
  pg.set_PGSLr()
  terms = [IonIon, IonElectron, Hartree, pg.forward, PerdewBurkeErnzerhof]  
  ...
  system = System(box_vecs, shape, ions, terms)

Contents
--------

.. toctree::

   ion_electron 
   kinetic_functionals
   xc_functionals
   
