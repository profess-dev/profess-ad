Kinetic Energy Functionals
==========================

Kinetic Energy Functional Template Class
----------------------------------------

.. _kinetic_template:

.. autoclass:: functionals.KineticFunctional
   :members:

Exact Kinetic Energy Functionals
--------------------------------

Thomas-Fermi Functional
^^^^^^^^^^^^^^^^^^^^^^^

.. _tf:

.. autofunction:: functionals.ThomasFermi

von Weizsaecker Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _vw:

.. autofunction:: functionals.Weizsaecker

Semi-local Kinetic Energy Functionals
-------------------------------------

Semi-local functionals generally take the form

.. math:: \tau_\text{S}(\mathbf{r}) = \tau_\text{vW}(\mathbf{r}) + F_\theta[n](\mathbf{r}) \tau_\text{TF}(\mathbf{r}) 

where :math:`\tau_\text{vW}` is the von Weizsaecker kinetic energy density,
:math:`\tau_\text{TF}(\mathbf{r})` is the Thomas-Fermi kinetic energy density and 
:math:`F_\theta[n](\mathbf{r})` is the Pauli enhancement factor to be approximated.

Often, the Pauli enhancement factor is made a function of the :ref:`reduced gradient <s>`, 

.. math:: s = \frac{|\nabla f|}{2(3\pi)^{1/3} n^{4/3}}

and the :ref:`reduced Laplacian <q>`,

.. math:: q = \frac{\nabla^2 f}{4(3\pi)^{2/3} n^{5/3}}

vWGTF Functionals
^^^^^^^^^^^^^^^^^

.. autofunction:: functionals.vWGTF1

.. autofunction:: functionals.vWGTF2

Luo-Karasiev-Trickey (LKT) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _lkt:

.. autofunction:: functionals.LuoKarasievTrickey

Pauli-Gaussian Functionals
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: functionals.PauliGaussian
   :members:

Yukawa GGA Functionals
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: functionals.YukawaGGA
   :members:

Non-local Kinetic Energy Functionals
------------------------------------

Wang-Teter (WT) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: functionals.WangTeter

Perrot Functional
^^^^^^^^^^^^^^^^^

.. autofunction:: functionals.Perrot

Smargiassi-Madden (SM) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: functionals.SmargiassiMadden

Wang-Govind-Carter 98 (WGC98) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: functionals.WangGovindCarter98

Wang-Teter Style Functional 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: functionals.WangTeterStyleFunctional
   :members:

Wang-Govind-Carter 99 (WGC99) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: functionals.WangGovindCarter99
   :members:

Foley-Madden (FM) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: functionals.FoleyMadden
   :members:
   
KGAP Functional
^^^^^^^^^^^^^^^

.. autofunction:: functionals.KGAP

Huang-Carter (HC) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _hc:

.. autoclass:: functionals.HuangCarter
   :members:

Revised Huang-Carter (revHC) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _revhc:

.. autoclass:: functionals.RevisedHuangCarter
   :members:

Mi-Genova-Pavanello (MGP) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: functionals.MiGenovaPavanello   
   :members:
   
Xu-Wang-Ma (XWM) Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
.. _xwm:
   
.. autofunction:: functionals.XuWangMa