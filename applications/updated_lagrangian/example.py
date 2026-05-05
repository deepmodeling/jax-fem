import os
import shutil

import jax
import jax.numpy as jnp
import numpy as onp

from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol


jax.config.update("jax_enable_x64", True)

MU = 1.0
LAMBDA = 2.0
LOAD_SCHEDULE_STEPS = 7
N_STEPS = 7
RIGHT_PULL = -0.1
SIMP_P = 3.0
RHO_MIN = 0.02
EPS = 1e-5

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")


def clear_output_dir():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def boundary_node_tags(points0, x_left=0.0, x_right=1.0, atol=1e-7):
    """
    Assigns each reference-mesh node a tag: 0 for interior nodes,
    1 for left boundary (x = x_left), and 2 for right boundary (x = x_right).
    """
    p = jnp.asarray(points0)
    x = p[:, 0]
    tags = jnp.zeros(p.shape[0], dtype=jnp.int32)
    tags = tags.at[jnp.isclose(x, x_left, atol=atol)].set(1)
    tags = tags.at[jnp.isclose(x, x_right, atol=atol)].set(2)
    return tags


def nodal_displacement_total_target(points0, lam, tags):
    """Total displacement useful for imposing Dirichlet boundary conditions."""
    p = jnp.asarray(points0)
    th = lam * jnp.pi
    y0 = p[:, 1]
    z0 = p[:, 2]
    left = tags == 1
    right = tags == 2
    ux_r = lam * RIGHT_PULL
    uy_r = y0 * (jnp.cos(th) - 1.0) - z0 * jnp.sin(th)
    uz_r = y0 * jnp.sin(th) + z0 * (jnp.cos(th) - 1.0)
    c0 = jnp.where(left, 0.0, jnp.where(right, ux_r, 0.0))
    c1 = jnp.where(left, 0.0, jnp.where(right, uy_r, 0.0))
    c2 = jnp.where(left, 0.0, jnp.where(right, uz_r, 0.0))
    u_total = jnp.stack([c0, c1, c2], axis=1)
    return u_total


def build_dirichlet_bc_by_tag(tags):
    def loc_left(point, ind):
        return tags[ind] == 1

    def loc_right(point, ind):
        return tags[ind] == 2

    def _zero(point):
        return 0.0

    return [
        [loc_left, loc_left, loc_left, loc_right, loc_right, loc_right],
        [0, 1, 2, 0, 1, 2],
        [_zero, _zero, _zero, _zero, _zero, _zero],
    ]


def push_F_prev(fe0, x_frozen, delta_u, F_prev):
    shape_grads = fe0.get_shape_grads(x_frozen)[0]
    cells = fe0.cells
    du_c = delta_u[cells]

    def one_cell(d_u, s_g, F_p):
        grad_du = jnp.einsum("ni,qnj->qij", d_u, s_g)
        I3 = jnp.eye(3)
        F_inc = I3 + grad_du
        return jnp.einsum("qij,qjk->qik", F_inc, F_p)

    return jax.vmap(one_cell)(du_c, shape_grads, F_prev)


class UpdatedLagrangian(Problem):
    def custom_init(self, bnd_tags):
        self._bnd_tags = bnd_tags
        self.points0 = self.fes[0].points
        nc, nq = self.fes[0].shape_grads.shape[:2]
        self.internal_vars = [
            jnp.broadcast_to(jnp.eye(3), (nc, nq, 3, 3)),
            jnp.ones((nc, nq)),
        ]

    def material_kirchhoff(self, F):
        I = jnp.eye(3)
        J = jnp.linalg.det(F)
        J = jnp.maximum(J, 1e-14)
        b = F @ jnp.swapaxes(F, -1, -2)
        lnJ = jnp.log(J)
        return MU * (b - I) + LAMBDA * lnJ[:, None, None] * I

    def get_universal_kernel(self):
        def universal_kernel(
            cell_sol_flat,
            physical_quad_points,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            *cell_internal_vars,
        ):
            del physical_quad_points, cell_JxW
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_du = cell_sol_list[0]
            cell_shape_grads = cell_shape_grads[:, : self.fes[0].num_nodes, :]
            cell_v_grads_JxW = cell_v_grads_JxW[:, : self.fes[0].num_nodes, :, :]
            F_prev, rho_q = cell_internal_vars[0], cell_internal_vars[1]
            simp = rho_q ** SIMP_P

            grad_du = jnp.einsum("ni,qnj->qij", cell_du, cell_shape_grads)
            I = jnp.eye(3)
            F_inc = I + grad_du
            F = jnp.einsum("qij,qjk->qik", F_inc, F_prev)

            J = jnp.linalg.det(F)
            J = jnp.maximum(J, 1e-14)
            tau = self.material_kirchhoff(F)
            # Per-cell vmap: rho_q is (n_quad,), tau is (n_quad, 3, 3).
            tau = tau * simp[:, None, None]
            cauchy = tau / J[:, None, None]
            J_inc = jnp.linalg.det(F_inc)
            J_inc = jnp.maximum(J_inc, 1e-14)
            F_inc_inv_T = jnp.linalg.inv(F_inc).swapaxes(-1, -2)
            T_pull = J_inc[:, None, None] * jnp.einsum("qij,qjk->qik", cauchy, F_inc_inv_T)

            val = jnp.sum(T_pull[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            return jax.flatten_util.ravel_pytree(val)[0]

        return universal_kernel

    def set_params(self, params):
        points = params["points"]
        lam = params["lam"]
        lam_prev = params["lam_prev"]
        F_prev = params["F_prev"]
        rho_el = params.get("rho")
        if rho_el is None:
            rho_el = jnp.ones((self.fes[0].num_cells,))
        rho_q = jnp.broadcast_to(rho_el[:, None], F_prev.shape[:2])
        self.initialize_geometric_quantities([points])
        self.internal_vars = [F_prev, rho_q]
        u_new = nodal_displacement_total_target(self.points0, jnp.asarray(lam), self._bnd_tags)
        u_old = nodal_displacement_total_target(self.points0, jnp.asarray(lam_prev), self._bnd_tags)
        delta_u = u_new - u_old
        fe = self.fes[0]
        fe.vals_list = [
            delta_u[fe.node_inds_list[i], fe.vec_inds_list[i]] for i in range(len(fe.node_inds_list))
        ]


def objective_ul(u_cum):
    return jnp.sum(u_cum ** 2)


def build_mesh_and_problem():
    ele_type = "HEX8"
    meshio_mesh = box_mesh(9, 2, 2, 1.0, 0.2, 0.2)
    cell_type = get_meshio_cell_type(ele_type)
    pts = onp.array(meshio_mesh.points, dtype=onp.float64)
    pts[:, 1] -= 0.1
    pts[:, 2] -= 0.1
    mesh = Mesh(pts, meshio_mesh.cells_dict[cell_type])
    bnd_tags = boundary_node_tags(mesh.points)
    dbc = build_dirichlet_bc_by_tag(bnd_tags)
    problem = UpdatedLagrangian(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dbc,
        location_fns=None,
        additional_info=(bnd_tags,),
    )
    return problem


def end_to_end_ul(rho_design, problem, fe0, fwd_pred, aut=False):
    rho = jnp.clip(jnp.asarray(rho_design), RHO_MIN, 1.0)
    nc, nq = fe0.shape_grads.shape[:2]
    F_prev = jnp.broadcast_to(jnp.eye(3), (nc, nq, 3, 3))
    u_cum = jnp.zeros_like(problem.points0)
    lam_prev = jnp.asarray(0.0)
    delta_us = []

    for step in range(N_STEPS):
        lam = jnp.asarray((step + 1.0) / LOAD_SCHEDULE_STEPS)
        x_frozen = problem.points0 + u_cum
        params = {
            "points": x_frozen,
            "lam": lam,
            "lam_prev": lam_prev,
            "F_prev": F_prev,
            "rho": rho,
        }
        sol_list = fwd_pred(params)
        delta_u = sol_list[0]
        delta_us.append(delta_u)
        F_prev = push_F_prev(fe0, x_frozen, delta_u, F_prev)
        u_cum = u_cum + delta_u
        lam_prev = lam

    J = objective_ul(u_cum)
    if not aut:
        return J

    aux = {"delta_us": jnp.stack(delta_us, axis=0)}
    return J, aux


def verify_grad_directional_fd(rho0, grad_rho, problem, fe0, fwd_pred, key, eps):
    """Compare AD ``g·v`` to one-sided FD ``(J(ρ+εv)-J(ρ))/ε`` with one random unit ``v`` over all ρ."""


def write_step_vtus(fe0, delta_stack, points0):
    """Write per-step VTU files with step-start mesh, ``sol``=Δu."""
    for step in range(N_STEPS):
        lam = float(step + 1) / LOAD_SCHEDULE_STEPS
        u_before = onp.zeros_like(points0) if step == 0 else onp.sum(delta_stack[:step], axis=0)
        x_frozen = points0 + u_before
        delta_u = delta_stack[step]
        print(
            f"load step {step + 1}/{N_STEPS} (λ=({step + 1})/{LOAD_SCHEDULE_STEPS}={lam:.4f}), "
            f"max|Δu|={float(onp.max(onp.abs(delta_u))):.6e}, "
        )
        tag = f"{step:04d}"
        path_vtu = os.path.join(OUTPUT_DIR, f"delta_u_{tag}.vtu")
        fe0.points = x_frozen
        save_sol(fe0, delta_u, path_vtu)

    print(f"Wrote VTU under: {OUTPUT_DIR}")


def main():
    clear_output_dir()
    problem = build_mesh_and_problem()
    fe0 = problem.fes[0]
    nc = fe0.num_cells
    fwd_pred = ad_wrapper(problem, solver_options={'spsolve_solver': {}}, 
        adjoint_solver_options={'spsolve_solver': {}})

    # AD-based derivatives
    def J_with_aux(rho_design):
        return end_to_end_ul(rho_design, problem, fe0, fwd_pred, aut=True)

    print("\n=== AD-based derivatives ===")
    rho0 = jnp.ones((nc,)) * 0.65 + 0.05 * jnp.sin(jnp.linspace(0.0, jnp.pi, nc))
    (J, aux), grad_rho = jax.value_and_grad(J_with_aux, has_aux=True)(rho0)
    v = jax.random.normal(jax.random.PRNGKey(0), rho0.shape)
    v = v / jnp.linalg.norm(v)
    ad_dir = jnp.dot(grad_rho, v)

    # FD-based derivatives
    def J_only(r):
        return end_to_end_ul(r, problem, fe0, fwd_pred, aut=False)

    print("\n=== FD-based derivatives ===")
    J0 = J_only(rho0)
    Jp = J_only(rho0 + EPS * v)
    fd_dir = (Jp - J0) / EPS

    # Error comparison
    abs_err = jnp.abs(ad_dir - fd_dir)
    rel_err = abs_err / (jnp.abs(fd_dir) + 1e-20)
    print("\n=== Directional derivative: AD vs finite difference (random unit v over all ρ) ===")
    print(f"ε = {EPS:.2e},  AD: g·v = {float(ad_dir):.8e},  FD: (J(ρ+εv)-J(ρ))/ε = {float(fd_dir):.8e}")
    print(f"absolute error = {float(abs_err):.4e},  relative error = {float(rel_err):.4e}")

    # Write VTU files
    delta_stack = onp.asarray(aux["delta_us"])
    points0 = onp.asarray(problem.points0)
    write_step_vtus(fe0, delta_stack, points0)


if __name__ == "__main__":
    main()
