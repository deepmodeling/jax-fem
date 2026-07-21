import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import modify_vtu_file, save_sol


class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def get_mass_map(self):
        def mass_map(u, x):
            val = -np.array([1e3*x[0], 2e3*x[1], 3e3*x[2]])
            return val
        return mass_map

    def get_surface_maps(self):
        def surface_map(u, x):
            return -np.array([1e3*x[0]**2 + 1e3*x[1]**2, 0., 0.])

        return [surface_map]

    def surface_integral(self, location_fn, surface_fn, sol):
        """Compute surface integral specified by surface_fn: f(u_grad) * ds
        For post-processing only.
        Example usage: compute the total force on a certain surface.

        Parameters
        ----------
        location_fn: callable
            A function that inputs a point (ndarray) and returns if the point satisfies the location condition.
        surface_fn: callable
            A function that inputs a point (ndarray) and returns the value.
        sol: ndarray
            (num_total_nodes, vec)

        Returns
        -------
        int_val: ndarray
            (vec,)
        """
        boundary_inds = self.fes[0].get_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.fes[0].get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = sol[self.fes[0].cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
        traction = surface_fn(u_grads_face) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
        int_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))
        return int_val

    def compute_surface_area(self, location_fn, sol):
        """For post-processing only
        """
        def unity_fn(u_grads):
            unity = np.ones_like(u_grads)[:, :, :, 0]
            return unity
        unity_integral_val = self.surface_integral(location_fn, unity_fn, sol)
        return unity_integral_val


class Test(unittest.TestCase):
    """Test linear elasticity with cylinder mesh
    """
    # @unittest.skip("Temporarily skip")
    def test_solve_problem(self):
        """Compare FEniCSx solution with JAX-FEM
        """
        problem_name = "linear_elasticity_cylinder"
        crt_dir = os.path.dirname(__file__)
        fenicsx_vtu_path_raw = os.path.join(crt_dir, "fenicsx/sol_p0_000000.vtu")
        fenicsx_vtu_path = os.path.join(crt_dir, "fenicsx/sol.vtu")
        modify_vtu_file(fenicsx_vtu_path_raw, fenicsx_vtu_path)
        fenicsx_vtu = meshio.read(fenicsx_vtu_path)
        cells = fenicsx_vtu.cells_dict['VTK_LAGRANGE_HEXAHEDRON'] # 'hexahedron'
        points = fenicsx_vtu.points
        mesh = Mesh(points, cells)
        R = 5.
        H = 10.

        def top(point):
            return np.isclose(point[2], H, atol=1e-5)

        def bottom(point):
            return np.isclose(point[2], 0., atol=1e-5)

        def dirichlet_val(point):
            return 0.

        location_fns = [bottom, bottom, bottom]
        value_fns = [dirichlet_val, dirichlet_val, dirichlet_val]
        vecs = [0, 1, 2]
        dirichlet_bc_info = [location_fns, vecs, value_fns]

        location_fns = [top]
        problem = LinearElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
        sol_list = solver(problem)

        jax_vtu_path = os.path.join(crt_dir, "jax_fem/sol.vtu")
        save_sol(problem.fes[0], sol_list[0], jax_vtu_path)
        jax_fem_vtu = meshio.read(jax_vtu_path)

        fenicsx_sol = fenicsx_vtu.point_data['sol']
        jax_fem_sol = jax_fem_vtu.point_data['sol']

        print(f"Solution absolute value differs by {np.max(np.absolute(jax_fem_sol - fenicsx_sol))} between FEniCSx and JAX-FEM")
        onptest.assert_array_almost_equal(fenicsx_sol, jax_fem_sol, decimal=3)

        # Compute the top surface area of the cylinder with FEniCSx and JAX-FEM
        fenicsx_surface_area = np.load(os.path.join(crt_dir, "fenicsx/surface_area.npy"))
        jax_fem_area = problem.compute_surface_area(top, sol_list[0])[0]
        print(f"Circle area is {np.pi*R**2}")
        print(f"FEniCSx computes approximate area to be {fenicsx_surface_area}")
        print(f"JAX-FEM computes approximate area to be {jax_fem_area}")

        onptest.assert_almost_equal(fenicsx_surface_area, jax_fem_area, decimal=4)


if __name__ == '__main__':
    unittest.main()
