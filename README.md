# PROFESS-AD
PyTorch-based auto-differentiable orbital-free density functional theory (OFDFT) package.

Details, documentation and examples can be found at [PROFESS-AD's website](https://profess-ad.readthedocs.io/en/latest/index.html).

Requirements
------------
- Python 3.9 or higher 
- NumPy 1.21.0 or higher
- SciPy 1.7.0 or higher
- Matplotlib 3.4.3 or higher
- PyTorch 1.12.1 or higher (installation instructions found [here](https://pytorch.org/))
- ξ-torch or `xitorch` (installation instructions found [here](https://github.com/xitorch/xitorch))

Install
-------
To use PROFESS-AD, one can fork or clone this repository.

```
git clone https://github.com/profess-dev/profess-ad.git
```

It is recommended for users to create a conda environment to install all the required Python packages. To check that all the dependencies have
been installed correctly, one can perform tests as follows.
```
cd profess-ad/tests
python3 -m unittest
```

To conveniently use PROFESS-AD modules, one can then add the path to the `profess-ad/professad` directory to one's `PYTHONPATH` in one's `~/.bashrc`.
```
export PYTHONPATH=$PYTHONPATH:"/USERS_PATH/profess-ad/professad"
```

Cite
----
C.W. Tan, C.J. Pickard, and W.C. Witt. *Automatic Differentiation for Orbital-Free Density Functional Theory*.
[J. Chem. Phys. 158, 124801 (2023)](https://pubs.aip.org/aip/jcp/article/158/12/124801/2881839/Automatic-differentiation-for-orbital-free-density)

