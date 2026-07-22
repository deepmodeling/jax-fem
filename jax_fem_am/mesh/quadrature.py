"""Thermal quadrature modification: vertex-rule mass lumping.

Origin: 159_local/v03/am_thermal_stress_macro_intersection_mech100.py
(apply_thermal_mass_lumping). Moved verbatim in the 2026-07-22 restructure.
"""
import jax.numpy as np
import numpy as onp


def apply_thermal_mass_lumping(problem):
    """Switch the thermal volume quadrature to the TET4 vertex rule (lumping).

    With linear tets the stiffness integrand is constant per cell, so moving
    the quadrature points to the vertices (equal weights V/4) leaves
    conduction bitwise unchanged while making the capacitance matrix exactly
    the row-sum lumped diagonal: shape_vals becomes the identity, so the
    mass-map trial/test products N_i*N_j collapse to delta_ij. Source and
    loss terms in the mass map become nodal collocation, which is why the
    physical quad points must be relocated to the vertices as well.
    """
    fe = problem.fes[0]
    if fe.num_quads != fe.num_nodes:
        raise ValueError(
            "--thermal-mass-lumping requires num_quads == num_nodes "
            f"(TET4 with --quadrature-order 2); got {fe.num_quads} quads, "
            f"{fe.num_nodes} nodes")
    weights = onp.asarray(fe.quad_weights)
    if not onp.allclose(weights, weights[0]):
        raise ValueError(
            "--thermal-mass-lumping requires an equal-weight quadrature rule; "
            f"got weights {weights}")
    grads = onp.asarray(fe.shape_grads_ref)
    if onp.allclose(grads, grads[0:1]):
        # Linear simplex (TET4): constant gradients, so the vertex rule also
        # leaves conduction bitwise unchanged.
        fe.shape_vals = np.eye(fe.num_nodes)
        label = "TET4 vertex quadrature (conduction unchanged)"
    else:
        # HEX8 2x2x2 Gauss: pair each quad point with the vertex of its
        # octant (the trilinear basis peaks at its own vertex, so argmax of
        # the original shape values is that pairing). Capacitance/source
        # terms collocate at the vertices with weight V/8 while conduction
        # keeps its Gauss-point gradients - the Abaqus first-order
        # heat-transfer element (DC3D8) split.
        sv = onp.asarray(fe.shape_vals)
        nearest = onp.argmax(sv, axis=1)
        if len(set(nearest.tolist())) != fe.num_nodes:
            raise ValueError(
                "--thermal-mass-lumping: quadrature points do not pair "
                "one-to-one with element vertices; use --quadrature-order 2")
        perm = onp.zeros_like(sv)
        perm[onp.arange(fe.num_quads), nearest] = 1.0
        fe.shape_vals = np.asarray(perm)
        label = "HEX8 vertex collocation (conduction stays at Gauss points)"
    problem.physical_quad_points = fe.get_physical_quad_points()
    print(f"thermal mass lumping: {label} installed (diagonal capacitance)")
    return problem
