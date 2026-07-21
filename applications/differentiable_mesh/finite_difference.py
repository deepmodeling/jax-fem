import numpy as onp

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver


def central(J_plus: float, J_minus: float, epsilon: float) -> float:
    return (J_plus - J_minus) / (2.0 * epsilon)


def gold_fd_two_independent_problems(
    points0,
    cells,
    ele_type: str,
    dirichlet_bc_info,
    location_fns,
    problem_cls,
    objective_fn,
    perturb_node: int,
    perturb_axis: int,
    fd_eps: float,
    *,
    vec: int = 1,
    dim: int = 2,
):
    pts_m = onp.array(points0, dtype=onp.float64)
    pts_p = onp.array(points0, dtype=onp.float64)
    pts_m[perturb_node, perturb_axis] -= fd_eps
    pts_p[perturb_node, perturb_axis] += fd_eps

    def solve(pts):
        mesh = Mesh(pts, cells)
        prob = problem_cls(
            mesh=mesh,
            vec=vec,
            dim=dim,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns,
        )
        sol = solver(prob)
        return sol, prob

    sol_m, problem_m = solve(pts_m)
    sol_p, problem_p = solve(pts_p)

    Jm = float(objective_fn(sol_m))
    Jp = float(objective_fn(sol_p))
    fd_gold = central(Jp, Jm, fd_eps)
    n_m = int(problem_m.boundary_inds_list[0].shape[0])
    n_p = int(problem_p.boundary_inds_list[0].shape[0])
    return fd_gold, Jm, Jp, n_m, n_p
