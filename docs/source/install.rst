Install
=======

The exact requirements and installation instructions can be found in PROFESS-AD's
`GitHub page <https://github.com/profess-dev/profess-ad>`_ too.

Requirements
------------
It is recommended for users to create a conda environment to install all the required Python packages.
Besides Numpy, Scipy and Matplotlib, the PyTorch library (version 1.12.1 or higher) is required.
Instructions to install PyTorch can be found `here <https://pytorch.org/>`_.

The `ξ-torch <https://xitorch.readthedocs.io/en/latest/index.html>`_ library (`Github <https://github.com/xitorch/xitorch>`_)
is essential for the higher-order derivative functionalities of PROFESS-AD. It can be installed using pip: ::

  pip install xitorch


Installation
------------
To use PROFESS-AD, one can fork or clone the GitHub repository. ::

  git clone https://github.com/profess-dev/profess-ad.git

To conveniently use PROFESS-AD modules, one can then add the path to the ``profess-ad/professad`` directory to one's 
``PYTHONPATH`` in one's ``~/.bashrc``. ::

  export PYTHONPATH=$PYTHONPATH:"/USERS_PATH/profess-ad/professad"

Testing
-------
To check that all the dependencies have been installed correctly, and that PROFESS-AD is working correctly,
one can perform some tests as follows. ::

  cd profess-ad/tests
  python3 -m unittest
