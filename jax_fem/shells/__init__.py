"""
jax_fem.shells — shell finite element implementations in pure JAX
=================================================================

Currently provides:
  mitc4   MITC4 flat-shell element (Bathe & Dvorkin 1984)

Why a separate sub-module?
--------------------------
The existing jax_fem framework assembles stiffness matrices through the
FiniteElement / Problem abstraction backed by Basix shape functions.  Shell
elements cannot use that path for two reasons:

1. Mixed interpolation.  MITC4 evaluates transverse shear strains at four
   edge-midpoint "tying points" and interpolates them to the Gauss points
   (Bathe & Dvorkin 1984).  This mixed interpolation is not expressible as a
   standard Lagrangian basis available in Basix.

2. Rotational DOFs.  Shell elements carry 5–6 DOFs per node (translations +
   rotations), whereas the solid elements currently in jax_fem use only
   translational DOFs.  The local-to-global frame transformation for rotations
   requires element-specific orthonormal frame construction.

The implementations here are standalone JAX modules: fully differentiable
(via jnp.linalg.solve + implicit function theorem), vmappable, and compatible
with jax.jit.  They can be used independently of jax_fem's Problem/solver
infrastructure, or wrapped to produce force vectors consumable by it.
"""

from .mitc4 import (
    element_stiffness_global,
    assemble_K,
    apply_dirichlet,
    recover_stress,
    run_shell_fem,
)

__all__ = [
    'element_stiffness_global',
    'assemble_K',
    'apply_dirichlet',
    'recover_stress',
    'run_shell_fem',
]
