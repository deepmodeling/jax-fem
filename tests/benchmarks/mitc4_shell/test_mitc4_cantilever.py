"""
Cantilever plate benchmark for jax_fem.shells.mitc4.

Kirchhoff plate theory gives the tip deflection of a clamped-free plate under
uniform pressure q as:

    delta = q * L^4 / (8 * D)    where D = E * t^3 / (12 * (1 - nu^2))

This is the "wide-plate" (plane-strain) limit.  For a narrow plate (b << L)
the boundary condition at the free sides removes the Poisson constraint and the
beam formula applies:

    delta = 12 * q * L^4 / (8 * E * t^3)

We test both limits and verify MITC4 is within 3% of wide-plate theory and
within 5% of beam theory, and that results are physically reasonable.

Geometry: L = 1 m (clamped direction), variable width b.
Material: E = 70 GPa (aluminium), nu = 0.3.
Load:     q = 100 N/m^2 uniform pressure in z (out-of-plane).
"""

import numpy as np
import jax
import jax.numpy as jnp
import unittest

jax.config.update('jax_enable_x64', True)

from jax_fem.shells.mitc4 import run_shell_fem


def _plate_delta(E, nu, t, L, q):
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    return q * L**4 / (8.0 * D)


def _beam_delta(E, t, L, q):
    return 12.0 * q * L**4 / (8.0 * E * t**3)


def _make_plate(n_span, n_chord, L, b):
    eta = np.linspace(0, 1, n_span)
    xi  = np.linspace(0, 1, n_chord)
    pts = np.zeros((n_span, n_chord, 3))
    for i in range(n_span):
        for j in range(n_chord):
            pts[i, j, 0] = xi[j] * b
            pts[i, j, 1] = eta[i] * L
            pts[i, j, 2] = 0.0
    return jnp.array(pts, dtype=jnp.float64)


def _consistent_loads(n_span, n_chord, L, b, q):
    """Lumped consistent nodal loads for uniform pressure q in z."""
    dA = (L / (n_span - 1)) * (b / (n_chord - 1))
    F = np.zeros((n_span, n_chord, 3))
    for i in range(n_span):
        for j in range(n_chord):
            ei = (i == 0 or i == n_span - 1)
            ej = (j == 0 or j == n_chord - 1)
            if ei and ej:
                F[i, j, 2] = q * dA / 4.0
            elif ei or ej:
                F[i, j, 2] = q * dA / 2.0
            else:
                F[i, j, 2] = q * dA
    return jnp.array(F, dtype=jnp.float64)


class TestMITC4Cantilever(unittest.TestCase):

    E  = 70e9
    nu = 0.3
    t  = 0.01
    L  = 1.0
    q  = 100.0

    def _run(self, n_span, n_chord, b):
        pts = _make_plate(n_span, n_chord, self.L, b)
        F   = _consistent_loads(n_span, n_chord, self.L, b, self.q)
        _, delta_tip, _ = run_shell_fem(
            pts, F, self.E, self.nu, self.t, n_span, n_chord)
        return float(delta_tip)

    def test_wide_plate_matches_kirchhoff(self):
        """b/L = 4: plane-strain limit, plate formula should hold within 3%."""
        ref = _plate_delta(self.E, self.nu, self.t, self.L, self.q)
        delta = self._run(n_span=16, n_chord=8, b=4.0)
        err = abs(delta - ref) / ref * 100
        self.assertLess(err, 3.0,
            f"Wide-plate error {err:.2f}% exceeds 3% (got {delta*1e3:.4f} mm,"
            f" expected {ref*1e3:.4f} mm)")

    def test_narrow_plate_matches_beam(self):
        """b/L = 0.1: free-edge limit, beam formula should hold within 5%."""
        ref = _beam_delta(self.E, self.t, self.L, self.q)
        delta = self._run(n_span=16, n_chord=3, b=0.1)
        err = abs(delta - ref) / ref * 100
        self.assertLess(err, 5.0,
            f"Narrow-plate error {err:.2f}% exceeds 5% (got {delta*1e3:.4f} mm,"
            f" expected {ref*1e3:.4f} mm)")

    def test_deflection_positive_and_bounded(self):
        """Tip deflection in +z under +z pressure; magnitude within physical bounds."""
        ref_plate = _plate_delta(self.E, self.nu, self.t, self.L, self.q)
        ref_beam  = _beam_delta(self.E, self.t, self.L, self.q)
        delta = self._run(n_span=16, n_chord=4, b=0.5)
        self.assertGreater(delta, 0.0, "Tip deflection should be positive")
        self.assertGreater(delta, ref_plate * 0.9,
            "Deflection unexpectedly small (possible over-stiffness / locking)")
        self.assertLess(delta, ref_beam * 1.1,
            "Deflection unexpectedly large (possible instability)")


if __name__ == '__main__':
    unittest.main()
