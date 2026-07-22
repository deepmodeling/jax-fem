import argparse
import os
from collections import deque

import jax.numpy as np
import meshio
import numpy as onp

from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol


class SmallStrainElasticity(Problem):
    def custom_init(self, young=1.0, poisson=0.3):
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lmbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    def get_tensor_map(self):
        def stress(u_grad):
            strain = 0.5 * (u_grad + u_grad.T)
            return self.lmbda * np.trace(strain) * np.eye(self.dim) + 2.0 * self.mu * strain

        return stress


def connected_cell_ids(cells, max_cells):
    if max_cells is None or max_cells <= 0 or max_cells >= len(cells):
        return onp.arange(len(cells), dtype=onp.int64)

    node_to_cells = {}
    for cell_id, cell in enumerate(cells):
        for node_id in cell:
            node_to_cells.setdefault(int(node_id), []).append(cell_id)

    selected = {0}
    queue = deque([0])
    while queue and len(selected) < max_cells:
        cell_id = queue.popleft()
        for node_id in cells[cell_id]:
            for next_cell_id in node_to_cells[int(node_id)]:
                if next_cell_id not in selected:
                    selected.add(next_cell_id)
                    queue.append(next_cell_id)
                    if len(selected) >= max_cells:
                        break
            if len(selected) >= max_cells:
                break

    return onp.array(sorted(selected), dtype=onp.int64)


def compact_mesh(points, cells, max_cells):
    cell_ids = connected_cell_ids(cells, max_cells)
    sub_cells_old = cells[cell_ids]
    used_nodes = onp.unique(sub_cells_old.reshape(-1))
    old_to_new = -onp.ones(len(points), dtype=onp.int64)
    old_to_new[used_nodes] = onp.arange(len(used_nodes), dtype=onp.int64)
    sub_points = points[used_nodes]
    sub_cells = old_to_new[sub_cells_old]
    return sub_points, orient_tet4(sub_points, sub_cells), len(cell_ids)


def orient_tet4(points, cells):
    pts = points[cells]
    signed = onp.einsum(
        "ij,ij->i",
        onp.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]),
        pts[:, 3] - pts[:, 0],
    )
    neg = signed < 0.0
    if onp.any(neg):
        cells = cells.copy()
        tmp = cells[neg, 1].copy()
        cells[neg, 1] = cells[neg, 2]
        cells[neg, 2] = tmp
    return cells


def read_tet4_inp(path, max_cells):
    meshio_mesh = meshio.read(path)
    if "tetra" not in meshio_mesh.cells_dict:
        available = ", ".join(sorted(meshio_mesh.cells_dict))
        raise ValueError(f"Expected tetra cells in {path}; available cells: {available}")

    points = onp.asarray(meshio_mesh.points, dtype=onp.float64)
    cells = onp.asarray(meshio_mesh.cells_dict["tetra"], dtype=onp.int64)
    return compact_mesh(points, cells, max_cells)


def make_dirichlet_bc(points, displacement):
    xmin = float(points[:, 0].min())
    xmax = float(points[:, 0].max())
    span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1.0)
    atol = 1e-8 * span
    anchor0_id = int(onp.argmin(points[:, 0]))
    anchor0 = points[anchor0_id]
    dist0 = onp.linalg.norm(points - anchor0, axis=1)
    anchor1_id = int(onp.argmax(dist0))
    anchor1 = points[anchor1_id]
    axis = anchor1 - anchor0
    axis_norm = max(float(onp.linalg.norm(axis)), 1e-12)
    cross_dist = onp.linalg.norm(onp.cross(points - anchor0, axis / axis_norm), axis=1)
    anchor2_id = int(onp.argmax(cross_dist))
    anchor2 = points[anchor2_id]

    def at_node(target):
        target = np.asarray(target)

        def location(point):
            return np.linalg.norm(point - target) <= atol

        return location

    def right(point):
        return np.isclose(point[0], xmax, atol=atol)

    def zero(_point):
        return 0.0

    def ux(_point):
        return displacement

    location_fns = [
        at_node(anchor0),
        at_node(anchor0),
        at_node(anchor0),
        at_node(anchor1),
        at_node(anchor1),
        at_node(anchor2),
        right,
    ]
    vecs = [0, 1, 2, 1, 2, 2, 0]
    value_fns = [zero, zero, zero, zero, zero, zero, ux]
    return [location_fns, vecs, value_fns]


def make_initial_guess(points, displacement):
    xmin = float(points[:, 0].min())
    xmax = float(points[:, 0].max())
    xspan = max(xmax - xmin, 1e-12)
    guess = onp.zeros((len(points), 3), dtype=onp.float64)
    guess[:, 0] = displacement * (points[:, 0] - xmin) / xspan
    return [np.asarray(guess)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp",
        default="/home/user/work/159/schema/0119_c3d4_only.inp",
        help="Path to a meshio-readable Abaqus inp file containing C3D4/tetra cells.",
    )
    parser.add_argument("--max-cells", type=int, default=200)
    parser.add_argument("--displacement", type=float, default=1e-4)
    parser.add_argument("--young", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.3)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/home/user/work/159/output/inp_initial_guess_smoke",
    )
    args = parser.parse_args()

    points, cells, selected_cells = read_tet4_inp(args.inp, args.max_cells)
    mesh = Mesh(points, cells, ele_type="TET4")
    problem = SmallStrainElasticity(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type="TET4",
        dirichlet_bc_info=make_dirichlet_bc(points, args.displacement),
        additional_info=(args.young, args.poisson),
    )
    initial_guess = make_initial_guess(points, args.displacement)

    os.makedirs(args.output_dir, exist_ok=True)
    initial_vtu = os.path.join(args.output_dir, "initial_guess.vtu")
    save_sol(problem.fes[0], initial_guess[0], initial_vtu)

    print("inp:", args.inp)
    print("selected_cells:", selected_cells)
    print("points:", len(points))
    print("dofs:", problem.num_total_dofs_all_vars)
    print("bc_node_counts:", [len(x) for x in problem.fes[0].node_inds_list])
    print("initial_guess_shape:", initial_guess[0].shape)
    print("initial_guess_vtu:", initial_vtu)

    if args.solve:
        sol_list = solver(
            problem,
            solver_options={
                "newton": {
                    "initial_guess": initial_guess,
                    "linear": {"spsolve_solver": {}},
                    "tol": 1e-8,
                    "rel_tol": 1e-8,
                }
            },
        )
        solved_vtu = os.path.join(args.output_dir, "solved.vtu")
        save_sol(problem.fes[0], sol_list[0], solved_vtu)
        print("solved_vtu:", solved_vtu)


if __name__ == "__main__":
    main()
