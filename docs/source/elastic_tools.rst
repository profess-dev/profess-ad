Elastic Tools
=============

Equation of State Fit
---------------------

.. autofunction:: elastic_tools.fit_eos

Elastic Properties
------------------

.. _elastic_properties:

The following are various post-processing tools that can be used to convert the 6 x 6 elastic 
constant matrix of Birch coefficients into various quantities that characterize the elastic
properties of the system. An example of how this might be used is given in the 
:ref:`elastic constant example <elastic_constants_example>`. 


Reuss Moduli
^^^^^^^^^^^^

.. autofunction:: elastic_tools.reuss_moduli

Voigt Moduli
^^^^^^^^^^^^

.. autofunction:: elastic_tools.voigt_moduli

Shear Modulus
^^^^^^^^^^^^^

.. autofunction:: elastic_tools.shear_average

Young's Modulus
^^^^^^^^^^^^^^^

.. autofunction:: elastic_tools.youngs_modulus

Poisson's Ratio
^^^^^^^^^^^^^^^

.. autofunction:: elastic_tools.poissons_ratio
