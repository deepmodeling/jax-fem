"""Shared 3D Neo-Hookean cube setup (axial stretch + lateral traction)."""

import jax
import jax.numpy as np

from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.problem import Problem

Lx, Ly, Lz = 1.0, 1.0, 1.0
STRETCH = 0.02
T_LATERAL = 1.0e-3
E_BASE = 10.0


class HyperElasticityClassic(Problem):
    """Unit cube: x=0 fixed, x=Lx stretched, uniform ty on y=Ly."""

    def get_tensor_map(self):
        def psi(F):
            E = E_BASE
            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            kappa = E / (3.0 * (1.0 - 2.0 * nu))
            J = np.linalg.det(F)
            Jinv = J ** (-2.0 / 3.0)
            I1 = np.trace(F.T @ F)
            return (mu / 2.0) * (Jinv * I1 - 3.0) + (kappa / 2.0) * (J - 1.0) ** 2

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            F = u_grad + np.eye(self.dim)
            return P_fn(F)

        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            del u, x
            return np.array([0.0, T_LATERAL, 0.0])

        return [surface_map]


class HyperElasticityInverse(Problem):
    """Same BCs as classic; ``set_params(rho)`` scales modulus only."""

    def custom_init(self):
        self.fe = self.fes[0]
        self.E = E_BASE

    def get_tensor_map(self):
        def psi(F, rho):
            E = self.E * rho
            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            kappa = E / (3.0 * (1.0 - 2.0 * nu))
            J = np.linalg.det(F)
            J = np.maximum(J, 1e-14)
            Jinv = J ** (-2.0 / 3.0)
            I1 = np.trace(F.T @ F)
            return (mu / 2.0) * (Jinv * I1 - 3.0) + (kappa / 2.0) * (J - 1.0) ** 2

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, rho):
            F = u_grad + np.eye(self.dim)
            return P_fn(F, rho)

        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            del u, x
            return np.array([0.0, T_LATERAL, 0.0])

        return [surface_map]

    def set_params(self, rho):
        self.internal_vars = [rho]


def _uniaxial_dirichlet_bc_info():
    def left(point):
        return np.isclose(point[0], 0.0, atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def zero_dirichlet_val(point):
        del point
        return 0.0

    def stretch_val(point):
        del point
        return STRETCH

    return [
        [left] * 3 + [right] * 3,
        [0, 1, 2] * 2,
        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]
        + [stretch_val, zero_dirichlet_val, zero_dirichlet_val],
    ]


def _y_max(point):
    return np.isclose(point[1], Ly, atol=1e-5)


def build_hyperelastic3d_problem_classic(Nx, Ny, Nz):
    ele_type = "HEX8"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    problem = HyperElasticityClassic(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=_uniaxial_dirichlet_bc_info(),
        location_fns=[_y_max],
    )
    return problem, mesh


def build_hyperelastic3d_problem_inverse(Nx, Ny, Nz):
    ele_type = "HEX8"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    problem = HyperElasticityInverse(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=_uniaxial_dirichlet_bc_info(),
        location_fns=[_y_max],
    )
    return problem, mesh
