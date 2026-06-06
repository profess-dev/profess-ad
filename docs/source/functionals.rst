Energy Terms and Functionals
============================

This page lists the energy terms and functionals that have been implemented. They can be included
in a list of energy terms in the ``terms`` argument that goes into the initialization of a :doc:`system` object.

Take the following for example ::

  from professad.functionals import IonElectron, Hartree, WangTeter, PerdewBurkeErnzerhof
  from professad.ion_utils import PickardSum

  terms = [PickardSum(), IonElectron(), Hartree(), WangTeter(), PerdewBurkeErnzerhof()]
  ...
  system = System(box_vecs, shape, ions, terms)

Each term in the list is a callable instance. Functionals that accept parameters can be configured
at construction time. For example, the :ref:`Xu-Wang-Ma functional <xwm>` accepts a ``kappa`` parameter ::

  terms = [PickardSum(), IonElectron(), Hartree(), XuWangMa(kappa=0), PerdewZunger()]
  ...
  system = System(box_vecs, shape, ions, terms)

Some kinetic functionals are represented by classes that inherit from the ``torch.nn.Module`` class, in which
case the instance itself should be included in the list of terms. For example, ::

  pg = PauliGaussian()
  pg.set_PGSLr()
  terms = [PickardSum(), IonElectron(), Hartree(), pg, PerdewBurkeErnzerhof()]
  ...
  system = System(box_vecs, shape, ions, terms)

Contents
--------

.. toctree::

   ion_electron
   kinetic_functionals
   xc_functionals
