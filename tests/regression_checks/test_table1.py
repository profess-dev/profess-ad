import os

import pytest
import torch

from professad.crystal_tools import get_cell
from professad.functionals import Hartree, IonElectron, PerdewBurkeErnzerhof, WangTeter
from professad.ion_utils import EwaldSum
from professad.system import System

POTENTIALS_DIR = os.path.join(os.path.dirname(__file__), "..", "potentials")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ECUT = 2000  # eV

RECPOT = {
    "Li": os.path.join(POTENTIALS_DIR, "li.gga.recpot"),
    "Mg": os.path.join(POTENTIALS_DIR, "mg.gga.recpot"),
    "Al": os.path.join(POTENTIALS_DIR, "al.gga.recpot"),
}

# reference values from Table I (OF-WT row)
# V0 in Å³/atom, K0 in GPa, dE0 in meV/atom relative to ground state
REF = {
    "Li": {
        "V0": {"fcc": 20.2, "hcp": 20.3, "bcc": 20.2, "sc": 21.9, "dc": 30.3},
        "dE0": {"fcc": 0, "hcp": 0, "bcc": 1, "sc": 151, "dc": 429},
        "K0": {"fcc": 16, "hcp": 16, "bcc": 16, "sc": 12, "dc": 6},
    },
    "Mg": {
        "V0": {"fcc": 23.1, "hcp": 23.2, "bcc": 23.0, "sc": 27.2, "dc": 39.5},
        "dE0": {"fcc": 10, "hcp": 0, "bcc": 26, "sc": 391, "dc": 839},
        "K0": {"fcc": 37, "hcp": 37, "bcc": 37, "sc": 24, "dc": 11},
    },
    "Al": {
        "V0": {"fcc": 16.8, "hcp": 16.9, "bcc": 17.2, "sc": 19.9, "dc": 28.8},
        "dE0": {"fcc": 0, "hcp": 18, "bcc": 73, "sc": 312, "dc": 791},
        "K0": {"fcc": 79, "hcp": 77, "bcc": 72, "sc": 58, "dc": 24},
    },
}

STRUCTURES = ["fcc", "hcp", "bcc", "sc", "dc"]


def run_eos(element, crystal):
    vol = REF[element]["V0"][crystal]
    box_vecs, frac_ion_coords = get_cell(crystal, vol, coord_type="fractional")
    ions = [[element, RECPOT[element], frac_ion_coords]]
    shape = System.ecut2shape(ECUT, box_vecs)
    terms = [
        EwaldSum(alpha=0.5, Rc=10.0, shape=shape, pme_order=10),
        IonElectron(),
        Hartree(),
        WangTeter(),
        PerdewBurkeErnzerhof(),
    ]
    system = System(box_vecs, shape, ions, terms, units="a", coord_type="fractional", device=DEVICE)
    params, err = system.eos_fit(f=0.05, N=11, ntol=1e-7, eos="bm")
    K0, K0prime, E0, V0 = params
    return {"V0": float(V0), "E0": float(E0), "K0": float(K0)}


# collect all results per element so we can compute dE0
_element_results = {}


def get_element_results(element):
    if element not in _element_results:
        res = {}
        for crystal in STRUCTURES:
            res[crystal] = run_eos(element, crystal)
        E0_min = min(r["E0"] for r in res.values())
        for crystal in STRUCTURES:
            res[crystal]["dE0"] = (res[crystal]["E0"] - E0_min) * 1000
        _element_results[element] = res

        # print table for this element
        print(f"\n  {element:>4s}{'':>8s}{'fcc':>8s}{'hcp':>8s}{'bcc':>8s}{'sc':>8s}{'dc':>8s}")
        print("  " + "-" * 50)
        for prop in ["V0", "K0", "dE0"]:
            fmt = "6.1f" if prop == "V0" else "6.0f"
            row = "".join(f"  {res[c][prop]:{fmt}}" for c in STRUCTURES)
            row_ref = "".join(f"  {REF[element][prop][c]:{fmt}}" for c in STRUCTURES)
            print(f"  {prop:>4s}  {'OF-WT':>8s}{row}")
            print(f"        {'(ref)':>8s}{row_ref}")

    return _element_results[element]


@pytest.fixture(params=[(e, c) for e in ["Li", "Mg", "Al"] for c in STRUCTURES], ids=lambda x: f"{x[0]}-{x[1]}")
def eos_result(request):
    element, crystal = request.param
    res = get_element_results(element)
    return element, crystal, res[crystal]


def test_V0(eos_result):
    element, crystal, res = eos_result
    assert res["V0"] == pytest.approx(REF[element]["V0"][crystal], abs=0.15)


def test_K0(eos_result):
    element, crystal, res = eos_result
    assert res["K0"] == pytest.approx(REF[element]["K0"][crystal], abs=1.5)


def test_dE0(eos_result):
    element, crystal, res = eos_result
    assert res["dE0"] == pytest.approx(REF[element]["dE0"][crystal], abs=5)
