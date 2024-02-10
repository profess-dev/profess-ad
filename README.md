# PROFESS-AD
Auto-differentiable orbital-free density functional theory (OFDFT) package in PyTorch.

Details, documentation and examples can be found at [PROFESS-AD's website](https://profess-ad.readthedocs.io/en/latest/index.html).

Install
-------

It is recommended for users to create a virtual environemnt to install all the required Python packages. For example, a conda environment,
```
conda create -n professad python
conda activate professad
```

To use PROFESS-AD, one can fork or clone this repository and pip install it. The necessary requirements will be installed, including `torch` and `xitorch`.
This might take a few minutes.

```
git clone https://github.com/profess-dev/profess-ad.git
cd profess-ad
pip install .
```

To check that all the dependencies have
been installed correctly, one can perform tests as follows.
```
cd profess-ad/tests
python -m unittest
```

Cite
----
C.W. Tan, C.J. Pickard, and W.C. Witt. *Automatic Differentiation for Orbital-Free Density Functional Theory*.
[J. Chem. Phys. 158, 124801 (2023)](https://pubs.aip.org/aip/jcp/article/158/12/124801/2881839/Automatic-differentiation-for-orbital-free-density)

