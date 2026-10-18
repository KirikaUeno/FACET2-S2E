"""
Tests for twiss_match_beam (simulators/bmad/runs.py): linear Twiss matching of a beam file.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from FACET2_S2E.simulators.bmad.runs import _beam_twiss


def write_mismatched_beam(path, n=20000, seed=1):
    """Beam with beta ~80 m, alpha ~-16 in both planes (like the 2024-12-11 IMPACT bunch at L0AFEND)."""
    rng = np.random.default_rng(seed)
    pz = 88e6*(1 + 1e-3*rng.standard_normal(n))
    data = {"t": 1e-12*rng.standard_normal(n), "z": np.zeros(n), "pz": pz, "weight": np.full(n, 1.6e-9/n),
            "status": np.ones(n, dtype=int), "species": "electron"}
    eps, beta, alpha = 2.5e-8, 80.0, -16.0
    for u, pu in (("x", "px"), ("y", "py")):
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        data[u] = np.sqrt(eps*beta)*a
        data[pu] = np.sqrt(eps/beta)*(b - alpha*a)*pz
    ParticleGroup(data=data).write(path + ".h5")


def test_matches_explicit_targets_and_keeps_emittance(tmp_path):
    path = str(tmp_path / "beam")
    write_mismatched_beam(path)
    P0 = ParticleGroup(path + ".h5")
    targets = {"beta_x": 15.3, "alpha_x": 2.18, "beta_y": 18.1, "alpha_y": 1.46}

    assert qs.twiss_match_beam(None, path, "L0AFEND", targets=targets) == targets

    P = ParticleGroup(path + ".h5")
    for plane in ("x", "y"):
        beta, alpha = _beam_twiss(P, plane)
        np.testing.assert_allclose(beta, targets[f"beta_{plane}"], rtol=1e-6)
        np.testing.assert_allclose(alpha, targets[f"alpha_{plane}"], rtol=1e-6)
        np.testing.assert_allclose(P[f"norm_emit_{plane}"], P0[f"norm_emit_{plane}"], rtol=1e-9)
    np.testing.assert_array_equal(P.pz, P0.pz)
    np.testing.assert_array_equal(P.t, P0.t)


def test_matching_twice_changes_nothing(tmp_path):
    path = str(tmp_path / "beam")
    write_mismatched_beam(path)
    targets = {"beta_x": 15.3, "alpha_x": 2.18, "beta_y": 18.1, "alpha_y": 1.46}
    qs.twiss_match_beam(None, path, "L0AFEND", targets=targets)
    P1 = ParticleGroup(path + ".h5")
    qs.twiss_match_beam(None, path, "L0AFEND", targets=targets)
    P2 = ParticleGroup(path + ".h5")
    for u in ("x", "px", "y", "py"):
        np.testing.assert_allclose(getattr(P2, u), getattr(P1, u), rtol=1e-6)   # floating-point level


class _FakeTao:
    """Only what golden_twiss_targets needs: tao.matrix(a, b)["mat6"]."""
    def __init__(self, mat6):
        self.mat6 = mat6

    def matrix(self, start, end):
        return {"mat6": self.mat6}


def test_golden_targets_propagate_onto_golden():
    # accelerating (non-symplectic) transport, det = p0/p1 = 0.55 per plane
    m = np.eye(6)
    m[0:2, 0:2] = np.array([[1.3, 4.2], [-0.05, 0.26]])
    m[2:4, 2:4] = np.array([[0.8, 3.1], [0.12, 1.15]]) * np.sqrt(0.55/np.linalg.det([[0.8, 3.1], [0.12, 1.15]]))
    targets = qs.golden_twiss_targets(_FakeTao(m), "L0AFEND")
    golden = qs.GOLDEN_TWISS_PR10571
    for plane, i in (("x", 0), ("y", 2)):
        b0, a0 = targets[f"beta_{plane}"], targets[f"alpha_{plane}"]
        B0 = np.array([[b0, -a0], [-a0, (1 + a0**2)/b0]])
        M = m[i:i+2, i:i+2]
        B1 = M @ B0 @ M.T / abs(np.linalg.det(M))
        np.testing.assert_allclose(B1[0, 0], golden[f"beta_{plane}"], rtol=1e-10)
        np.testing.assert_allclose(-B1[0, 1], golden[f"alpha_{plane}"], rtol=1e-8, atol=1e-10)
