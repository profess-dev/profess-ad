About
=====

Introduction
------------

**PROFESS-AD** is PyTorch-based auto-differentiable plane-wave orbital-free density functional theory (OFDFT) code. 
Its auto-differentiable implementation allows for various operations in conventional OFDFT programs to be performed
with only the base energy functional expressions. This means that functional derivatives and stress expressions need 
not be analytically derived to perform density optimizations and geometry optimizations. PROFESS-AD also takes
advantage of the `ξ-torch <https://xitorch.readthedocs.io/en/latest/index.html>`_ library (a PyTorch-based library for
scientific computing) to compute higher-order derivative quantities such as bulk moduli, elastic constants and force 
constants directly. This method is exact in principle, in contrast to the finite-difference methods generally used for 
such quantities that have a dependence on the optimized density. 

Due to the ease of implementing and testing new functionals, without the need for explicit and possibly lengthy derivations
of derivative quantities, PROFESS-AD serves as a prototyping tool to simplify and accelerate functional development.
PROFESS-AD's various utility functions and PyTorch-based implementation is also meant to support the growing popularity of 
machine-learning techniques for kinetic functional development in the community. 
