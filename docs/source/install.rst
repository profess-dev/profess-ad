Install
=======

The exact requirements and installation instructions can be found in PROFESS-AD's
`GitHub page <https://github.com/profess-dev/profess-ad>`_ too.

Installation
------------

Users are recommended to create a virtual environment to install PROFESS-AD. For example, a conda environment. ::

  conda create -n professad python
  conda activate professad

To use PROFESS-AD, one can clone the GitHub repository and pip install it. ::

  git clone https://github.com/profess-dev/profess-ad.git
  cd profess-ad
  pip install .

Testing
-------
To check that all the dependencies have been installed correctly, and that PROFESS-AD is working correctly,
one can perform some tests as follows. ::

  cd profess-ad/tests
  python -m unittest
