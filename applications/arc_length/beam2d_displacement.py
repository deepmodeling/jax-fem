"""Displacement-controlled arc-length (buckling; λ=1 → END_COMPRESSION)."""

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
from jax_fem.solver import solver
from jax_fem.utils import save_sol

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'arc_length_displacement')

END_COMPRESSION = -0.05 * Lx


def location_midspan_bottom(point):
    return np.isclose(point[1], 0., atol=1e-5) & (point[0] > Lx / 2. - 2.) & (point[0] < Lx / 2. + 2.)


def compression_dirichlet_bc_info():
    def zero(point):
        return 0.

    def end_compression(point):
        return END_COMPRESSION

    return [
        [location_left, location_left, location_right, location_right],
        [0, 1, 0, 1],
        [zero, zero, end_compression, zero],
    ]


class BeamHyperelasticWithImperfection(BeamHyperelastic):
    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 1e-5])
        return [surface_map]


def main():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)

    mesh = make_mesh()
    problem = BeamHyperelasticWithImperfection(
        mesh, vec=2, dim=2, ele_type='QUAD4',
        dirichlet_bc_info=compression_dirichlet_bc_info(),
        location_fns=[location_midspan_bottom],
    )

    fe = problem.fes[0]
    for label, node_inds in zip(
        ('left ux', 'left uy', 'right ux', 'right uy'), fe.node_inds_list,
    ):
        assert len(node_inds) == 3, f'{label}: expected 3 edge nodes, got {len(node_inds)}'

    def on_step(step, u_vec, lam):
        print(f"step {step:4d}  lambda = {lam:.6f}")
        if step % 10 == 0:
            sol = problem.unflatten_fn_sol_list(u_vec)[0]
            right = onp.isclose(mesh.points[:, 0], Lx, atol=1e-5)
            vtk_path = os.path.join(OUTPUT_DIR, f'u{step:05d}.vtu')
            save_sol(problem.fes[0], np.hstack((sol, np.zeros((len(sol), 1)))), vtk_path)
            assert onp.max(onp.abs(sol[right, 1])) < 1e-6, (
                f'step {step}: right-edge uy BC violated, max |uy|={onp.max(onp.abs(sol[right, 1])):.3e}'
            )

    sol_list, info = solver(problem, solver_options={
        'arc_length': {
            'control': 'displacement',
            'return_info': True,
            'Delta_l': 0.1,
            'psi': 1.,
            'max_continuation_steps': 600,
            'step_callback': on_step,
        },
    })

    sol = sol_list[0]
    right = onp.isclose(mesh.points[:, 0], Lx, atol=1e-5)
    print(f"\ncontinuation: lambda = {info['lam']:.6f},  steps = {len(info['history'])}")
    if info['polished']:
        print(f"Newton polish: full BC at lambda = {info['lambda_target']:.6f}")
    print("final solution:")
    print(f"  right edge ux = {sol[right, 0]}  (target {END_COMPRESSION:.4f})")
    print(f"  right edge uy = {sol[right, 1]}  (must be ~0)")
    print(f"  max |uy| (interior) = {float(onp.max(onp.abs(sol[:, 1]))):.6f}")


if __name__ == '__main__':
    main()
