Ion and Electron Interaction Terms
==================================

For any practical OFDFT calculation of a real system, all three of the

* ion-ion interaction term,

* ion-electron interaction term, and

* electron-electron interaction term (Hartree term)

must be included in the calculation. Some of these terms can be omitted 
depending on the use case or for testing purposes.

Ion-Ion Interaction 
-------------------

.. autofunction:: functionals.IonIon

Ion-Electron Interaction
------------------------

.. autofunction:: functionals.IonElectron

Hartree Energy
--------------

.. _hart:

.. autofunction:: functionals.Hartree

