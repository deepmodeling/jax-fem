"""Force-controlled arc-length buckling (hand Crisfeld via ``jax_fem.solver``)."""

import os
import shutil

import jax.numpy as np
import numpy as onp

from applications.arc_length.hyperelastic_models import (
    BeamHyperelastic,
    Lx,
    location_left,
    location_right,
    make_mesh,
)
from jax_fem.problem import Problem
from jax_fem.solver import get_q_vec, solver
from jax_fem.utils import save_sol

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'arc_length_force')

END_TRACTION = 5.0


def location_midspan_bottom(point):
    return np.isclose(point[1], 0., atol=1e-5) & (point[0] > Lx / 2. - 2.) & (point[0] < Lx / 2. + 2.)


def clamped_left_dirichlet_bc_info():
    def zero(point):
        return 0.

    return [
        [location_left, location_left],
        [0, 1],
        [zero, zero],
    ]


class BeamHyperelasticWithImperfection(BeamHyperelastic):
    """Main problem: design traction on the right edge (+x)."""

    def get_surface_maps(self):
        def imperfection(u, x):
            return np.array([0., 1e-5])

        def end_traction(u, x):
            return np.array([END_TRACTION, 0.])

        return [imperfection, end_traction]


class BeamCounterLoad(Problem):
    """Auxiliary problem for force-controlled arc-length (``q_vec_aux``).

    Surface traction is the opposite of the main problem's right-edge load
    (``-END_TRACTION``). ``get_q_vec`` turns this into the reference load
    vector ``q_aux``. The arc-length residual uses ``R + (1 - λ) q_aux``:

    - **λ = 0:** counter-traction cancels the main traction at ``u = 0``,
      so continuation starts from the undeformed reference state.
    - **λ = 1:** ``(1 - λ) q_aux = 0``; only the main problem's full design
      traction remains (the target forward problem).
    """

    def get_tensor_map(self):
        def first_PK_stress(u_grad):
            return np.zeros((self.dim, self.dim))
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([-END_TRACTION, 0.])
        return [surface_map]


def main():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)

    mesh = make_mesh()
    problem = BeamHyperelasticWithImperfection(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=clamped_left_dirichlet_bc_info(),
        location_fns=[location_midspan_bottom, location_right],
    )
    q_vec_aux = get_q_vec(BeamCounterLoad(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        location_fns=[location_right],
    ))

    def on_step(step, u_vec, lam):
        print(f"step {step:4d}  lambda = {lam:.6f}")
        if step % 10 == 0:
            sol = problem.unflatten_fn_sol_list(u_vec)[0]
            vtk_path = os.path.join(OUTPUT_DIR, f'u{step:05d}.vtu')
            save_sol(problem.fes[0], np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)

    sol_list, info = solver(problem, solver_options={
        'arc_length': {
            'control': 'force',
            'return_info': True,
            'q_vec_aux': q_vec_aux,
            'Delta_l': 0.1,
            'Delta_l_late': 1.0,
            'Delta_l_switch_step': 200,
            'psi': 0.5,
            'max_continuation_steps': 600,
            'step_callback': on_step,
            'linear': {'petsc_solver': {}},
        },
    })

    sol = sol_list[0]
    right = onp.isclose(mesh.points[:, 0], Lx, atol=1e-5)
    print(f"\ncontinuation: lambda = {info['lam']:.6f},  steps = {len(info['history'])}")
    if info['polished']:
        print(f"Newton polish: full-load equilibrium (lambda = {info['lambda_target']:.6f})")
    print("final solution:")
    print(f"  right edge ux = {sol[right, 0]}")
    print(f"  right edge uy = {sol[right, 1]}")
    print(f"  max |uy| = {float(onp.max(onp.abs(sol[:, 1]))):.6f}")


if __name__ == '__main__':
    main()
