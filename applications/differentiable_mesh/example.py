"""
Purpose of this tutorial:
    To deomstrate taking the derivatives of the FEM solutions with respect to the mesh points.

Governing equation (nonlinear Poisson / semilinear elliptic):
    -∇²u + u³ = g  on Ω = (0,1)²,
    weak form: ∫ ∇u·∇v dΩ + ∫ u³v dΩ = ∫ gv dΩ.

Manufactured solution (exact):
    u = sin(π x) (1 + y).  Then u|_{y=0} = sin(π x) ≠ 0 (except endpoints), so J = ∑ u_h² is
    sensitive to Neumann-boundary motion.  Here -∇²u = π² sin(π x) (1 + y), u³ = sin³(π x) (1 + y)³.

Boundary conditions:
    Dirichlet u = 0 on {x=0} and {x=1}; u = 2 sin(π x) on {y=1}.
    Neumann on {y=0}: n = (0,-1), h = ∂u/∂n = -∂u/∂y = -sin(π x); surface_map = -h in code.

Comparison of derivatives:
    AD (Automatic Differentiation) uses ``ad_wrapper`` (implicit adjoint).
    FD (Finite Difference) solves two independent problems per side.

Data flow:
    (1) Numpy hanldes the initial mesh and points since meshio is used to generate the mesh.
    (2) JAX hanldes the core computational procedures related to the FEM solutions and the derivatives.
    (3) Numpy handles the VTU file saving after the FEM solutions are obtained.
"""

import os
import sys
import jax
import jax.numpy as jnp

from jax_fem.problem import Problem
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol

import applications.differentiable_mesh.finite_difference as fd

crt_file_path = os.path.dirname(__file__)
output_dir = os.path.join(crt_file_path, "output")
vtk_dir = os.path.join(output_dir, "vtk")
os.makedirs(vtk_dir, exist_ok=True)


def left(point):
    return jnp.isclose(point[0], 0.0, atol=1e-5)

def right(point):
    return jnp.isclose(point[0], 1.0, atol=1e-5)

def bottom(point):
    return jnp.isclose(point[1], 0.0, atol=1e-3)

def top(point):
    return jnp.isclose(point[1], 1.0, atol=1e-5)

def u0(point):
    return 0.0

def u0_top(point):
    """Dirichlet on y=1: u_exact = sin(π x) (1 + 1)."""
    return jnp.sin(jnp.pi * point[0]) * 2.0

def u_exact_fn(x, y):
    return jnp.sin(jnp.pi * x) * (1.0 + y)

def objective_sum_u_squared(sol_list):
    """J = ∑ u² over all nodes"""
    u = sol_list[0]
    return jnp.sum(u**2)

def pick_fd_node_on_neumann_edge(points):
    """Select some node on the bottom Neumann edge to perturb."""
    target_xy = jnp.array([0.35, 0.0])
    p = points
    on_bottom = jnp.isclose(p[:, 1], 0.0, atol=1e-3)
    not_corner = (p[:, 0] > 1e-4) & (p[:, 0] < 1.0 - 1e-4)
    mask = on_bottom & not_corner
    dist = jnp.sum((p - target_xy) ** 2, axis=1)
    dist = jnp.where(mask, dist, jnp.inf)
    return int(jnp.argmin(dist))


class NonlinearPoisson(Problem):
    def get_tensor_map(self):
        """Include contributions from -∇²u."""
        return lambda grad_u: grad_u

    def get_mass_map(self):
        def mass_map(u, x):
            """Include contributions from u³ and -g, where 
            g is -∇²u + u³ = π² sin(π x) (1 + y) + sin³(π x) (1 + y)³.
            """
            sx = jnp.sin(jnp.pi * x[0])
            opy = 1.0 + x[1]
            u_ex = sx * opy
            lap_u = jnp.pi**2 * sx * opy
            g = lap_u + u_ex**3
            return jnp.array([u[0] ** 3 - g])
        return mass_map

    def get_surface_maps(self):
        """surface_map = −h with h = ∂u/∂n; u = sin(πx)(1+y) on y=0 ⇒ h = −sin(πx)."""
        def neumann_flux(u, x):
            return jnp.array([jnp.sin(jnp.pi * x[0])])
        return [neumann_flux]

    def set_params(self, points):
        """Effects of points will be reflected in the geometric quantities."""
        self.initialize_geometric_quantities([points])


def main():
    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    Nx, Ny = 32, 32
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=1.0, domain_y=1.0)
    cells = meshio_mesh.cells_dict[cell_type]
    mesh = Mesh(meshio_mesh.points, cells)
    dirichlet_bc_info = [[left, right, top], [0, 0, 0], [u0, u0, u0_top]]

    problem = NonlinearPoisson(
        mesh=mesh,
        vec=1,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[bottom],
    )

    points0 = mesh.points
    perturb_node = pick_fd_node_on_neumann_edge(points0)
    perturb_axis = 1
    fd_eps = 5e-6

    fwd_pred = ad_wrapper(problem)

    def forward_and_probe(pts):
        sol = fwd_pred(pts)
        return objective_sum_u_squared(sol), sol

    (loss_val, sol_list), grad_ad = jax.value_and_grad(forward_and_probe, has_aux=True)(points0)

    fd_gold, Jm, Jp, n_neu_m, n_neu_p = fd.gold_fd_two_independent_problems(
        points0,
        cells,
        ele_type,
        dirichlet_bc_info,
        [bottom],
        NonlinearPoisson,
        objective_sum_u_squared,
        perturb_node,
        perturb_axis,
        fd_eps,
    )

    # Since one node is perturbed, the number of Neumann faces must be the same.
    # Otherwise, the AD vs FD would not be comparable due to the different number of Neumann faces.
    n_neu_base = int(problem.boundary_inds_list[0].shape[0])
    if n_neu_m != n_neu_base or n_neu_p != n_neu_base:
        raise RuntimeError(
            "Neumann face count changed on perturbed meshes (baseline "
            f"{n_neu_base}, mesh− {n_neu_m}, mesh+ {n_neu_p}); "
            "AD vs gold FD would not be comparable — widen bottom atol or ε."
        )

    # Check if the solution is close to the exact solution.
    print("=== Nonlinear Poisson + Neumann (y=0) vs u = sin(πx)(1+y) ===")
    print(f"nodes = {len(sol_list[0].ravel())}, QUAD4 mesh {Nx}×{Ny}")
    u_ex = u_exact_fn(points0[:, 0], points0[:, 1])
    err = sol_list[0].ravel() - u_ex
    linf = float(jnp.max(jnp.abs(err)))
    print(f"‖u_h - u_exact‖_∞ = {linf:.6e}")

    # Check if the AD gradient is close to the FD gradient.
    _ax = ("x", "y")[perturb_axis]
    g_ad_ij = float(grad_ad[perturb_node, perturb_axis])
    J0 = float(loss_val)
    print(
        "=== Mesh sensitivity: J = ∑u², AD vs gold FD (two fresh solves per side) ===\n"
        f"Node {perturb_node} ({points0[perturb_node][0]:.17g}, {points0[perturb_node][1]:.17g}), ∂/∂{_ax}, ε = {fd_eps}; \n"
        f"J₀ = {J0:.12e}; J− = {Jm:.12e}; J+ = {Jp:.12e}\n"
        f"∂J/∂{_ax}: AD = {g_ad_ij:.12e}, \n"
        f"∂J/∂{_ax}: FD = {fd_gold:.12e}, \n"
        f"|AD − FD| = {abs(g_ad_ij - fd_gold):.3e}"
    )

    # Save the solution to a VTU file.
    vtk_path = os.path.join(vtk_dir, "u.vtu")
    save_sol(
        problem.fes[0],
        sol_list[0],
        vtk_path,
        point_infos=[
            ("u_exact", u_ex.reshape(-1, 1)),
            ("abs_err", jnp.abs(err).reshape(-1, 1)),
            ("dJ_dxy", grad_ad),
        ],
    )
    print(f"Wrote ParaView: {vtk_path}")


if __name__ == "__main__":
    main()
