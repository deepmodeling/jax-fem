"""Force-controlled arc-length buckling (hand Crisfeld via ``jax_fem.solver``)."""

import glob
import os

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
    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 1e-5])
        return [surface_map]


class BeamReferenceLoad(Problem):
    """Auxiliary problem: +x traction on right edge for ``get_q_vec``."""

    def get_tensor_map(self):
        def first_PK_stress(u_grad):
            return np.zeros((self.dim, self.dim))
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([10., 0.])
        return [surface_map]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(OUTPUT_DIR, '*.vtu')):
        os.remove(path)

    mesh = make_mesh()
    problem = BeamHyperelasticWithImperfection(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=clamped_left_dirichlet_bc_info(),
        location_fns=[location_midspan_bottom],
    )
    q_vec = get_q_vec(BeamReferenceLoad(
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
        'arc_length': True,
        'arc_length_control': 'force',
        'return_arc_length_info': True,
        'q_vec': q_vec,
        'Delta_l': 0.1,
        'Delta_l_late': 1.0,
        'Delta_l_switch_step': 200,
        'psi': 0.5,
        'lambda_max': 1.0,
        'max_continuation_steps': 500,
        'step_callback': on_step,
    })

    sol = sol_list[0]
    lam = info['lam']
    right = onp.isclose(mesh.points[:, 0], Lx, atol=1e-5)
    print(f"\nlambda = {lam:.6f},  steps = {len(info['history'])}")
    print(f"right edge ux = {sol[right, 0]}")
    print(f"right edge uy = {sol[right, 1]}")
    print(f"max |uy| = {float(onp.max(onp.abs(sol[:, 1]))):.6f}")


if __name__ == '__main__':
    main()
