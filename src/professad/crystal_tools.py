import numpy as np
import torch
from math import sqrt
from typing import Optional

# --------------------------------------------------------------------
# Helper script containing functions that provide the lattice vectors
# and ionic coordinates of simples crystal structures in the form of
# Pytorch tensors to double precision.
# --------------------------------------------------------------------


def get_cell(crystal: str,
             vol_per_atom: float,
             c_over_a: Optional[float] = sqrt(8 / 3),
             coord_type: Optional[str] = 'fractional',
             ):
    """ Crystal utility function

    This is a convenience function that returns the lattice vectors and ionic coordinates
    of simple crystal structures given the volume per atom in the cell and possibly the
    :math:`c/a` ratio.

    The crystal structures supported and their ``crystal`` arguments are as follows.

    * simple cubic -``sc``
    * body-centred cubic - ``bcc`` (primitive cell) or ``bcc-c`` (conventional cell)
    * face-centred cubic - ``fcc`` (primitive cell) or ``fcc-c`` (conventional cell)
    * diamond cubic - ``dc`` (primitive cell) or ``dc-c`` (conventional cell)
    * hexagonal close-packed - ``hcp``

    Args:
      crystal (string)    : Crystal structure
      vol_per_atom (float): Volume per atom of the cell
      c_over_a (float)    : :math:`c/a` ratio
      coord_type (string) : Whether the ionic coordinates returned are ``fractional`` (default)
                            or ``cartesian`` coordinates

    Returns:
      torch.Tensor, torch.Tensor: Lattice vectors, Ionic coordinates
    """
    match crystal:
        case 'sc':
            lattice_vectors, frac_ion_coords = simple_cubic(vol_per_atom)
        case 'bcc':
            lattice_vectors, frac_ion_coords = body_centered_cubic(vol_per_atom, 'primitive')
        case 'bcc-c':
            lattice_vectors, frac_ion_coords = body_centered_cubic(vol_per_atom, 'conventional')
        case 'fcc':
            lattice_vectors, frac_ion_coords = face_centered_cubic(vol_per_atom, 'primitive')
        case 'fcc-c':
            lattice_vectors, frac_ion_coords = face_centered_cubic(vol_per_atom, 'conventional')
        case 'dc':
            lattice_vectors, frac_ion_coords = diamond_cubic(vol_per_atom, 'primitive')
        case 'dc-c':
            lattice_vectors, frac_ion_coords = diamond_cubic(vol_per_atom, 'conventional')
        case 'hcp':
            lattice_vectors, frac_ion_coords = hexagonal_close_packed(vol_per_atom, c_over_a)
        case _:
            raise ValueError('\'crystal\' argument \'' + crystal + '\' not recognized')
    match coord_type:
        case 'fractional':
            return lattice_vectors, frac_ion_coords
        case 'cartesian':
            return lattice_vectors, frac_ion_coords @ lattice_vectors
        case _:
            raise ValueError('Only \'fractional\' or \'cartesian\' allowed for argument \'coord_type\'.')


def simple_cubic(vol_per_atom):
    a = vol_per_atom**(1 / 3)
    lattice_vectors = a * torch.eye(3, dtype=torch.double)
    frac_ion_coords = torch.zeros((1, 3), dtype=torch.double)
    return lattice_vectors, frac_ion_coords


def body_centered_cubic(vol_per_atom, cell_type='conventional'):
    a = (2 * vol_per_atom)**(1 / 3)
    if cell_type == 'primitive':
        lattice_vectors = a * torch.tensor([[-0.5, 0.5, 0.5],
                                            [0.5, -0.5, 0.5],
                                            [0.5, 0.5, -0.5]], dtype=torch.double)
        frac_ion_coords = torch.zeros((1, 3), dtype=torch.double)
    elif cell_type == 'conventional':
        lattice_vectors = a * torch.eye(3, dtype=torch.double)
        frac_ion_coords = torch.tensor([[0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.5]], dtype=torch.double)
    else:
        raise ValueError('Only \'primitive\' or \'conventional\' allowed for argument \'cell_type\'.')
    return lattice_vectors, frac_ion_coords


def face_centered_cubic(vol_per_atom, cell_type='primitive'):
    a = (4 * vol_per_atom)**(1 / 3)
    if cell_type == 'primitive':
        lattice_vectors = a * torch.tensor([[0.0, 0.5, 0.5],
                                            [0.5, 0.0, 0.5],
                                            [0.5, 0.5, 0.0]], dtype=torch.double)
        frac_ion_coords = torch.zeros((1, 3), dtype=torch.double)
    elif cell_type == 'conventional':
        lattice_vectors = a * torch.eye(3, dtype=torch.double)
        frac_ion_coords = torch.tensor([[0.0, 0.0, 0.0],
                                        [0.5, 0.5, 0.0],
                                        [0.5, 0.0, 0.5],
                                        [0.0, 0.5, 0.5]], dtype=torch.double)
    else:
        raise ValueError('Only \'primitive\' or \'conventional\' allowed for argument \'cell_type\'.')
    return lattice_vectors, frac_ion_coords


def diamond_cubic(vol_per_atom, cell_type='conventional'):
    a = (8 * vol_per_atom)**(1 / 3)
    if cell_type == 'primitive':
        lattice_vectors = a * torch.tensor([[0.0, 0.5, 0.5],
                                            [0.5, 0.0, 0.5],
                                            [0.5, 0.5, 0.0]], dtype=torch.double)
        frac_ion_coords = torch.tensor([[0.0, 0.0, 0.0],
                                        [1 / 4, 1 / 4, 1 / 4]], dtype=torch.double)
    elif cell_type == 'conventional':
        lattice_vectors = a * torch.eye(3, dtype=torch.double)
        frac_ion_coords = torch.tensor([[0.00, 0.00, 0.00],
                                        [0.50, 0.50, 0.00],
                                        [0.50, 0.00, 0.50],
                                        [0.00, 0.50, 0.50],
                                        [0.25, 0.25, 0.25],
                                        [0.25, 0.75, 0.75],
                                        [0.75, 0.75, 0.25],
                                        [0.75, 0.25, 0.75]], dtype=torch.double)
    else:
        raise ValueError('Only \'primitive\' or \'conventional\' allowed for argument \'cell_type\'.')
    return lattice_vectors, frac_ion_coords


def hexagonal_close_packed(vol_per_atom, c_over_a=1.633):
    # default c/a value (1.633) corresponds to ideal close-packing
    a = ((2 * vol_per_atom) / (np.sqrt(3) / 2 * c_over_a))**(1 / 3)
    lattice_vectors = a * torch.tensor([[1, 0, 0],
                                        [-0.5, np.sqrt(3) / 2, 0],
                                        [0, 0, c_over_a]], dtype=torch.double)
    frac_ion_coords = torch.tensor([[1 / 3, 2 / 3, 3 / 4],
                                    [2 / 3, 1 / 3, 1 / 4]], dtype=torch.double)
    return lattice_vectors, frac_ion_coords
