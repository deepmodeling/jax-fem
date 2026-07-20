"""
jax_fem.shells.mitc4 — MITC4 flat-shell finite element in pure JAX
===================================================================

Implements the MITC4 (Mixed Interpolation of Tensorial Components) 4-node
quadrilateral flat-shell element entirely in JAX, with no external FEM
library dependencies.

Element formulation
-------------------
Each element combines a Q4 plane-stress membrane with a Mindlin-Reissner
plate-bending element.  The key feature is the MITC4 mixed interpolation
for transverse shear strains (Bathe & Dvorkin 1984): instead of evaluating
γ_x = ∂w/∂x + θx and γ_y = ∂w/∂y + θy directly at Gauss points — which
causes shear locking as thickness t → 0 — the strains are sampled at four
edge-midpoint "tying points" and interpolated linearly to the Gauss points.
This gives the correct thin-plate limit without any special-case branching.

Why this cannot use jax_fem's FiniteElement / Basix path
---------------------------------------------------------
1. Mixed interpolation for transverse shear is not a standard Lagrangian
   basis and cannot be expressed through Basix shape functions.
2. Shell elements carry rotational DOFs (θx, θy, θz) requiring element-local
   orthonormal frames; the existing framework assumes translational DOFs only.

DOF layout
----------
Every node carries 6 global DOFs:  [u, v, w, θx, θy, θz]
  u, v, w    — translations [m]
  θx, θy     — rotations [rad]
  θz         — drilling DOF (weakly stabilised; not kinematically active)

Internally, each node has 5 local DOFs:  [u_l, v_l, w_l, θx_l, θy_l]
The 5×6 per-node transformation L_node maps global → local via the element
rotation matrix T = [e1 | e2 | e3]ᵀ.

Both B matrices and the local stiffness use the same node-by-node DOF
ordering:  [u,v,w,θx,θy]₀  [u,v,w,θx,θy]₁  [u,v,w,θx,θy]₂  [u,v,w,θx,θy]₃
(20 DOFs total per element in local frame).

Differentiability
-----------------
Every operation is a JAX primitive.  The linear solve

    u = K(pts)⁻¹ f(F_nodes)

is differentiated via the implicit function theorem, which JAX implements
automatically through the backward pass of jnp.linalg.solve:

    du/d(pts, F_nodes) = K⁻¹ (df/d· − dK/d· · u)

The backward pass costs one additional solve — approximately 2× forward time.

Performance note
----------------
jnp.linalg.solve is a dense O(N³) solve.  For meshes with N_dofs > ~5000
replace with jax.scipy.sparse.linalg.cg wrapped in jax.lax.scan for a
fixed-iteration differentiable iterative solve.

Validation
----------
Cantilever plate benchmark (see tests/benchmarks/mitc4_shell/):
  - Wide plate  (b/L=4): 1.02% error vs Kirchhoff plate theory  δ = qL⁴/8D
  - Narrow plate (b/L=0.1): 1.38% error vs beam theory  δ = 12qL⁴/8Et³

References
----------
  Bathe & Dvorkin (1984) "A four-node plate bending element based on
      Mindlin/Reissner plate theory and a mixed interpolation"
      Int J Numer Methods Eng 21:367-383.
  Bathe, Finite Element Procedures (1996), §5.4.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

# ---------------------------------------------------------------------------
# Gauss quadrature: 2×2 rule, weights = 1 each
# ---------------------------------------------------------------------------

_GP    = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_GP, -_GP, 1.0),
          ( _GP, -_GP, 1.0),
          ( _GP,  _GP, 1.0),
          (-_GP,  _GP, 1.0)]


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

def _shape(xi, eta):
    """
    Bilinear Q4 shape functions and their reference-space derivatives.

    Node ordering (counter-clockwise in ξ-η reference space):
      0: (−1,−1)   1: (+1,−1)   2: (+1,+1)   3: (−1,+1)

    Parameters
    ----------
    xi, eta : float

    Returns
    -------
    N   : (4,)   shape function values
    dN  : (2,4)  derivatives  [∂N/∂ξ ; ∂N/∂η]
    """
    N = 0.25 * jnp.array([
        (1. - xi) * (1. - eta),
        (1. + xi) * (1. - eta),
        (1. + xi) * (1. + eta),
        (1. - xi) * (1. + eta),
    ])
    dNdxi  = 0.25 * jnp.array([-(1. - eta),  (1. - eta),  (1. + eta), -(1. + eta)])
    dNdeta = 0.25 * jnp.array([-(1. - xi),  -(1. + xi),   (1. + xi),  (1. - xi) ])
    return N, jnp.stack([dNdxi, dNdeta])  # (4,), (2,4)


def _phys_derivs(xi, eta, xy):
    """
    Shape function derivatives in the physical element plane and |J|.

    Parameters
    ----------
    xi, eta : float
    xy      : (4,2)  corner coordinates in the local 2D element plane

    Returns
    -------
    N       : (4,)
    dN_phys : (2,4)  [∂N/∂x ; ∂N/∂y] in physical space
    detJ    : float  |J| > 0 for non-degenerate elements
    """
    N, dN  = _shape(xi, eta)
    J      = dN @ xy                                        # (2,2)
    detJ   = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    Jinv   = jnp.array([[ J[1, 1], -J[0, 1]],
                         [-J[1, 0],  J[0, 0]]]) / (detJ + 1e-12)
    return N, Jinv @ dN, detJ                               # (4,), (2,4), scalar


# ---------------------------------------------------------------------------
# Local coordinate frame
# ---------------------------------------------------------------------------

def _local_frame(coords):
    """
    Orthonormal local frame for a flat quadrilateral element.

    The frame is constructed as:
      e1  — normalised first edge (node 0 → node 1)
      e3  — element normal = normalise(e1 × diagonal)
      e2  = e3 × e1   (right-handed system)

    Parameters
    ----------
    coords : (4,3)  corner node coordinates in global 3-D [m]

    Returns
    -------
    e1, e2, e3 : (3,) each   orthonormal basis vectors
    xy         : (4,2)       corners projected to the element plane
                             (origin at node 0, x-axis = e1)
    T          : (3,3)       rotation matrix;  local = T @ global
    """
    e1   = coords[1] - coords[0]
    e1   = e1 / (jnp.linalg.norm(e1) + 1e-12)

    diag = coords[2] - coords[0]
    e3   = jnp.cross(e1, diag)
    e3   = e3 / (jnp.linalg.norm(e3) + 1e-12)

    e2 = jnp.cross(e3, e1)
    T  = jnp.stack([e1, e2, e3])               # rows are basis vectors

    rel = coords - coords[0]                    # (4,3)
    xy  = (T[:2, :] @ rel.T).T                 # (4,2) in-plane projection
    return e1, e2, e3, xy, T


# ---------------------------------------------------------------------------
# B matrices  (all use node-by-node DOF ordering, 20 DOFs per element)
# DOF layout: [u,v,w,θx,θy]_n  at columns 5n+{0,1,2,3,4}  for n = 0…3
# ---------------------------------------------------------------------------

def _bmat_mem(dN_phys):
    """
    Membrane strain–displacement matrix B_m (3,20).

    Maps the 20-DOF local displacement vector to in-plane strains
    [ε_xx, ε_yy, γ_xy].  Only u (col 5n) and v (col 5n+1) DOFs contribute.

    Parameters
    ----------
    dN_phys : (2,4)

    Returns
    -------
    B_m : (3,20)
    """
    dNdx, dNdy = dN_phys[0], dN_phys[1]
    u_cols = jnp.array([0,  5, 10, 15])   # u DOF: 5n+0
    v_cols = jnp.array([1,  6, 11, 16])   # v DOF: 5n+1
    B = jnp.zeros((3, 20))
    B = B.at[0, u_cols].set(dNdx)   # ε_xx = ∂u/∂x
    B = B.at[1, v_cols].set(dNdy)   # ε_yy = ∂v/∂y
    B = B.at[2, u_cols].set(dNdy)   # γ_xy = ∂u/∂y + ∂v/∂x
    B = B.at[2, v_cols].set(dNdx)
    return B


def _bmat_bend(dN_phys):
    """
    Bending curvature–displacement matrix B_b (3,20).

    Maps the 20-DOF local vector to bending curvatures
    [κ_xx, κ_yy, 2κ_xy] = [∂θx/∂x, ∂θy/∂y, ∂θx/∂y + ∂θy/∂x].
    Only θx (col 5n+3) and θy (col 5n+4) DOFs contribute; w has zero
    columns because the Mindlin formulation treats it independently.

    Parameters
    ----------
    dN_phys : (2,4)

    Returns
    -------
    B_b : (3,20)
    """
    dNdx, dNdy = dN_phys[0], dN_phys[1]
    tx_cols = jnp.array([3,  8, 13, 18])  # θx DOF: 5n+3
    ty_cols = jnp.array([4,  9, 14, 19])  # θy DOF: 5n+4
    B = jnp.zeros((3, 20))
    B = B.at[0, tx_cols].set(dNdx)   # κ_xx = ∂θx/∂x
    B = B.at[1, ty_cols].set(dNdy)   # κ_yy = ∂θy/∂y
    B = B.at[2, tx_cols].set(dNdy)   # 2κ_xy: ∂θx/∂y
    B = B.at[2, ty_cols].set(dNdx)   #        ∂θy/∂x
    return B


def _shear_B_at(xi, eta, xy):
    """
    Transverse shear strain–displacement matrix (2,20) at one reference point.

    Used to evaluate γ_x = ∂w/∂x + θx and γ_y = ∂w/∂y + θy at the MITC4
    tying points.  Contributions from w (col 5n+2), θx (col 5n+3), θy (5n+4).

    Parameters
    ----------
    xi, eta : float
    xy      : (4,2)  element corners in local 2-D

    Returns
    -------
    B_s : (2,20)
    """
    N, dN_phys, _ = _phys_derivs(xi, eta, xy)
    w_cols  = jnp.array([2,  7, 12, 17])  # w  DOF: 5n+2
    tx_cols = jnp.array([3,  8, 13, 18])  # θx DOF: 5n+3
    ty_cols = jnp.array([4,  9, 14, 19])  # θy DOF: 5n+4
    B = jnp.zeros((2, 20))
    B = B.at[0, w_cols ].set(dN_phys[0])  # γ_x: ∂w/∂x
    B = B.at[0, tx_cols].set(N)            #       + θx
    B = B.at[1, w_cols ].set(dN_phys[1])  # γ_y: ∂w/∂y
    B = B.at[1, ty_cols].set(N)            #       + θy
    return B


def _bmat_shear_mitc4(xi, eta, xy):
    """
    MITC4 mixed-interpolation transverse shear B matrix (2,20).

    Evaluates covariant shear strains at four edge-midpoint tying points
    (A,B,C,D) and interpolates them to the Gauss point (ξ,η).  This mixed
    interpolation eliminates the shear locking that afflicts standard Mindlin
    elements as t → 0.

    Tying points in reference space:
      A (0, −1): bottom edge midpoint  → contributes γ_x interpolation
      B (+1,  0): right edge midpoint  → contributes γ_y interpolation
      C (0, +1): top edge midpoint     → contributes γ_x interpolation
      D (−1,  0): left edge midpoint   → contributes γ_y interpolation

    Interpolation (Cartesian MITC4, valid for flat elements):
      γ_x(ξ,η) = ½(1−η) γ_x|_A  +  ½(1+η) γ_x|_C
      γ_y(ξ,η) = ½(1−ξ) γ_y|_D  +  ½(1+ξ) γ_y|_B

    Parameters
    ----------
    xi, eta : float  Gauss point
    xy      : (4,2)  element corners in local 2-D

    Returns
    -------
    B_s : (2,20)
    """
    BA = _shear_B_at( 0., -1., xy)
    BB = _shear_B_at(+1.,  0., xy)
    BC = _shear_B_at( 0., +1., xy)
    BD = _shear_B_at(-1.,  0., xy)
    Bγx = 0.5 * (1. - eta) * BA[0] + 0.5 * (1. + eta) * BC[0]
    Bγy = 0.5 * (1. - xi ) * BD[1] + 0.5 * (1. + xi ) * BB[1]
    return jnp.stack([Bγx, Bγy])  # (2,20)


# ---------------------------------------------------------------------------
# Constitutive matrices
# ---------------------------------------------------------------------------

def _constitutive(E, nu, t):
    """
    Plane-stress constitutive matrices for membrane, bending, and shear.

    Dimensional analysis:
      D_m : [Pa]        — stress per membrane strain (dimensionless)
      D_b : [Pa·m³=N·m] — moment per unit curvature per unit width
      D_s : [Pa·m=N/m]  — shear force per unit shear strain per unit width

    In the stiffness integral:
      K_mem  = t  · ∫ B_m^T D_m B_m dA   →  [N/m] force per translation
      K_bend =     ∫ B_b^T D_b B_b dA   →  [N·m] moment per rotation
      K_shear=     ∫ B_s^T D_s B_s dA   →  mixed (see note)

    The mixed units in K (N/m for translation-translation blocks, N·m for
    rotation-rotation blocks, N for cross-blocks) are expected and correct.

    Parameters
    ----------
    E, nu, t : float   Young's modulus [Pa], Poisson's ratio, thickness [m]

    Returns
    -------
    D_m : (3,3)
    D_b : (3,3)
    D_s : (2,2)
    """
    c   = E / (1. - nu ** 2)
    D0  = c * jnp.array([
        [1.,  nu,          0.          ],
        [nu,  1.,          0.          ],
        [0.,  0.,  (1. - nu) / 2.      ],
    ])
    D_m = D0                                    # [Pa]
    D_b = D0 * (t ** 3 / 12.)                  # [Pa·m³ = N·m]
    G   = E / (2. * (1. + nu))
    D_s = (5. / 6.) * G * t * jnp.eye(2)       # [Pa·m = N/m], k=5/6
    return D_m, D_b, D_s


# ---------------------------------------------------------------------------
# Element stiffness
# ---------------------------------------------------------------------------

def _element_K_local(xy, E, nu, t):
    """
    20×20 element stiffness matrix in the local element frame.

    Integrates membrane, bending, and MITC4 shear contributions with 2×2
    Gauss quadrature.  All contributions use the same 20-DOF node-by-node
    ordering so they can be summed directly into a single K.

    Parameters
    ----------
    xy   : (4,2)  corner coordinates in local 2-D
    E, nu, t : float

    Returns
    -------
    K_local : (20,20)
    """
    D_m, D_b, D_s = _constitutive(E, nu, t)
    K = jnp.zeros((20, 20))
    for xi, eta, w in _GAUSS:
        _, dN_p, detJ = _phys_derivs(xi, eta, xy)
        fac = w * jnp.abs(detJ)
        Bm  = _bmat_mem(dN_p)
        Bb  = _bmat_bend(dN_p)
        Bs  = _bmat_shear_mitc4(xi, eta, xy)
        K   = K + fac * (
            t     * (Bm.T @ D_m @ Bm)   # membrane: [N/m]
          +          (Bb.T @ D_b @ Bb)   # bending:  [N·m] for θ-θ blocks
          +          (Bs.T @ D_s @ Bs)   # shear:    [N/m] for w-w, [N·m] for θ-θ
        )
    return K


def _Lmat(T):
    """
    20×24 matrix mapping global 6-DOF nodal vectors to local 5-DOF ones.

    Per node:
      [u_l, v_l, w_l]  = T (3×3) @ [u_g, v_g, w_g]
      [θx_l, θy_l   ]  = T[:2,:] (2×3) @ [θx_g, θy_g, θz_g]

    Assembled as a block-diagonal matrix over 4 nodes.

    Parameters
    ----------
    T : (3,3)  rotation matrix  (rows = local basis vectors)

    Returns
    -------
    L : (20,24)
    """
    Ln = jnp.zeros((5, 6))
    Ln = Ln.at[0:3, 0:3].set(T)           # translations
    Ln = Ln.at[3:5, 3:6].set(T[:2, :])    # rotations (θx, θy; θz dropped)
    L  = jnp.zeros((20, 24))
    for n in range(4):
        L = L.at[5*n:5*n+5, 6*n:6*n+6].set(Ln)
    return L


def element_stiffness_global(coords, E, nu, t, alpha_drill=1e-3):
    """
    24×24 element stiffness matrix in the global frame.

    Steps:
      1. Compute orthonormal local frame and in-plane projection of nodes.
      2. Compute 20×20 local stiffness (membrane + bending + MITC4 shear).
      3. Transform to global via Lᵀ K_local L  →  (24×24).
      4. Add small drilling stabilisation  α·E·t  to the θz diagonal entries
         (DOFs 5, 11, 17, 23) to prevent a near-singular global K.

    The drilling stiffness is a regularisation only.  Choosing α = 1e-3 keeps
    the θz DOFs physically inert while eliminating the zero pivot.

    Parameters
    ----------
    coords      : (4,3)  corner coordinates in global 3-D [m]
    E, nu, t    : float
    alpha_drill : float  fraction of E·t added to θz diagonal

    Returns
    -------
    K_g : (24,24)
    """
    _, _, _, xy, T = _local_frame(coords)
    K_loc = _element_K_local(xy, E, nu, t)
    L     = _Lmat(T)
    K_g   = L.T @ K_loc @ L
    drill = alpha_drill * E * t
    drill_dofs = jnp.array([5, 11, 17, 23])
    K_g   = K_g.at[drill_dofs, drill_dofs].add(drill)
    return K_g


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def assemble_K(pts, conn, E, nu, t):
    """
    Assemble the global stiffness matrix from element contributions.

    Uses vmap to compute all element stiffnesses in parallel (differentiable
    w.r.t. pts via JAX AD), then jax.lax.scan to scatter them into the global
    (N_dofs × N_dofs) matrix one element at a time.

    jax.lax.scan compiles to a single loop body, avoiding the JIT compilation
    cost of unrolling N_elem separate scatter operations.

    Parameters
    ----------
    pts  : (N_nodes, 3)   node coordinates [m] — JAX array
    conn : (N_elem, 4)    element connectivity — static numpy int32 array
    E, nu, t : float

    Returns
    -------
    K : (N_dofs, N_dofs)   where N_dofs = 6 * N_nodes
    """
    N_nodes = pts.shape[0]
    N_dofs  = 6 * N_nodes

    # (N_elem, 24, 24) — differentiable w.r.t. pts
    K_elem = vmap(lambda c: element_stiffness_global(c, E, nu, t))(pts[conn])

    # DOF index array: (N_elem, 24)  — static, precomputed from connectivity
    dof_idx = jnp.array(
        [[6 * n + d for n in row for d in range(6)] for row in conn],
        dtype=jnp.int32)

    def _scatter(K, args):
        K_e, dofs = args
        return K.at[dofs[:, None], dofs[None, :]].add(K_e), None

    K, _ = jax.lax.scan(_scatter, jnp.zeros((N_dofs, N_dofs)),
                          (K_elem, dof_idx))
    return K


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def apply_dirichlet(K, f, fixed_dofs):
    """
    Enforce homogeneous Dirichlet BCs by row/column elimination.

    Zeros the rows and columns corresponding to constrained DOFs, sets their
    diagonal to 1, and zeros the corresponding entries in f.  This preserves
    symmetry of K and yields u[fixed] = 0 in the solution.

    Parameters
    ----------
    K          : (N_dofs, N_dofs)
    f          : (N_dofs,)
    fixed_dofs : (N_fixed,) int — static numpy array

    Returns
    -------
    K_mod, f_mod
    """
    fd    = jnp.array(fixed_dofs, dtype=jnp.int32)
    K_mod = (K.at[fd, :].set(0.)
               .at[:, fd].set(0.)
               .at[fd, fd].set(1.))
    f_mod = f.at[fd].set(0.)
    return K_mod, f_mod


# ---------------------------------------------------------------------------
# Stress recovery
# ---------------------------------------------------------------------------

def recover_stress(u_nodes, pts, conn, E, nu, t):
    """
    Von Mises stress at each element centroid, evaluated at the top surface.

    Combines membrane (mid-plane) and bending stresses at z = +t/2:

      σ_total = σ_membrane + σ_bending
              = D_m @ ε_m  +  (t/2) · D_m @ κ

    where ε_m = B_m(0,0) @ u_local  (membrane strain at centroid)
      and κ   = B_b(0,0) @ u_local  (curvature at centroid).

    The von Mises criterion for plane stress:
      σ_vm = √(σ_xx² − σ_xx σ_yy + σ_yy² + 3 σ_xy²)

    Transverse shear stresses τ_xz, τ_yz are zero at the outer surface by
    traction boundary conditions and are omitted.

    Parameters
    ----------
    u_nodes : (N_nodes, 6)  global nodal displacements/rotations
    pts     : (N_nodes, 3)  node coordinates
    conn    : (N_elem, 4)   connectivity (numpy int32)
    E, nu, t : float

    Returns
    -------
    sigma_vm : (N_elem,)  von Mises stress [Pa]
    """
    D_m, _, _ = _constitutive(E, nu, t)
    D_bs      = D_m * (t / 2.)   # bending stress factor at z = +t/2 [Pa·m]

    def _vm_one(conn_e):
        coords_e = pts[conn_e]                        # (4,3)
        _, _, _, xy, T = _local_frame(coords_e)
        u_e_l = _Lmat(T) @ u_nodes[conn_e].reshape(24)  # (20,) local DOFs

        _, dN_p, _ = _phys_derivs(0., 0., xy)           # centroid
        sigma_m = D_m  @ (_bmat_mem (dN_p) @ u_e_l)     # (3,) membrane [Pa]
        sigma_b = D_bs @ (_bmat_bend(dN_p) @ u_e_l)     # (3,) bending  [Pa]
        s = sigma_m + sigma_b

        return jnp.sqrt(jnp.maximum(
            s[0]**2 - s[0]*s[1] + s[1]**2 + 3.*s[2]**2,
            0.))

    return vmap(_vm_one)(jnp.array(conn))   # (N_elem,)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_shell_fem(pts, F_nodes, E, nu, t, n_span, n_chord):
    """
    Solve the thin-shell structural problem for a structured wing surface.

    The wing surface is modelled as a cantilever: all DOFs at the root span
    station (index 0) are clamped.  The structured mesh of n_span × n_chord
    nodes is divided into (n_span−1) × (n_chord−1) quad elements.

    Parameters
    ----------
    pts     : (n_span, n_chord, 3) JAX array  node coordinates [m]
    F_nodes : (n_span, n_chord, 3) JAX array  external nodal forces [N]
              Translational forces only; no applied moments.
    E       : float  Young's modulus [Pa]
    nu      : float  Poisson's ratio
    t       : float  shell thickness [m]
    n_span  : int    number of spanwise nodes
    n_chord : int    number of chordwise nodes

    Returns
    -------
    u_nodes   : (n_span, n_chord, 6)  nodal displacements [m] and rotations [rad]
    delta_tip : float  maximum translational displacement magnitude at the tip [m]
    sigma_max : float  maximum von Mises stress across all elements, top surface [Pa]

    Notes
    -----
    Uses jnp.linalg.solve (dense LU).  For n_span×n_chord = 20×40 (4800 DOFs)
    on a GPU this is ~0.4 s per call; the backward pass (gradient) is ~2×.

    The function is fully differentiable w.r.t. both pts and F_nodes.
    """
    N_nodes = n_span * n_chord
    N_dofs  = 6 * N_nodes

    # Element connectivity (static — does not depend on pts or F_nodes)
    conn = np.array([
        [s * n_chord + c,
         s * n_chord + c + 1,
         (s + 1) * n_chord + c + 1,
         (s + 1) * n_chord + c]
        for s in range(n_span  - 1)
        for c in range(n_chord - 1)
    ], dtype=np.int32)   # (N_elem, 4)

    pts_flat = pts.reshape(N_nodes, 3)

    # Load vector: translational DOFs only (moments remain zero)
    F_flat   = F_nodes.reshape(N_nodes, 3)
    F_padded = jnp.concatenate([F_flat, jnp.zeros_like(F_flat)], axis=-1)  # (N,6)
    f        = F_padded.ravel()                                              # (N_dofs,)

    # Assemble and apply BCs
    K = assemble_K(pts_flat, conn, E, nu, t)

    root_nodes = np.arange(n_chord, dtype=np.int32)
    fixed_dofs = np.array([6 * n + d for n in root_nodes for d in range(6)],
                           dtype=np.int32)
    K_mod, f_mod = apply_dirichlet(K, f, fixed_dofs)

    # Solve (dense LU; differentiable via implicit function theorem)
    u_flat  = jnp.linalg.solve(K_mod, f_mod)
    u_nodes = u_flat.reshape(N_nodes, 6)

    # Outputs
    tip_nodes  = jnp.arange((n_span - 1) * n_chord, n_span * n_chord)
    delta_tip  = jnp.linalg.norm(u_nodes[tip_nodes, :3], axis=-1).max()
    sigma_max  = recover_stress(u_nodes, pts_flat, conn, E, nu, t).max()

    return u_nodes.reshape(n_span, n_chord, 6), delta_tip, sigma_max
