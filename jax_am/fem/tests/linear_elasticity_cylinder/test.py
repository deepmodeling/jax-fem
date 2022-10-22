import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import unittest
from jax_am.fem.jax_fem import Mesh, LinearElasticity
from jax_am.fem.solver import solver
from jax_am.fem.utils import modify_vtu_file, save_sol


class Test(unittest.TestCase):
    """Test linear elasticity with cylinder mesh
    """
    # @unittest.skip("Temporarily skip")
    def test_solve_problem(self):
        """Compare FEniCSx solution with JAX-FEM
        """
        problem_name = "linear_elasticity_cylinder"
        fenicsx_vtu_path_raw = f"jax_am/fem/tests/{problem_name}/fenicsx/sol_p0_000000.vtu"
        fenicsx_vtu_path = f"jax_am/fem/tests/{problem_name}/fenicsx/sol.vtu"
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

        def neumann_val(point):
            return np.array([1e3*point[0]**2 + 1e3*point[1]**2, 0., 0.])

        def body_force(point):
            return np.array([1e3*point[0], 2e3*point[1], 3e3*point[2]])

        location_fns = [bottom, bottom, bottom]
        value_fns = [dirichlet_val, dirichlet_val, dirichlet_val]
        vecs = [0, 1, 2]
        dirichlet_bc_info = [location_fns, vecs, value_fns]

        neumann_bc_info = [[top], [neumann_val]]

        problem = LinearElasticity(f"{problem_name}", mesh, dirichlet_bc_info=dirichlet_bc_info, 
                                   neumann_bc_info=neumann_bc_info, source_info=body_force)

        sol = solver(problem)

        jax_vtu_path = f"jax_am/fem/tests/{problem_name}/jax_fem/sol.vtu"
        save_sol(problem, sol, jax_vtu_path)
        jax_fem_vtu = meshio.read(jax_vtu_path)

        fenicsx_sol = fenicsx_vtu.point_data['sol']
        jax_fem_sol = jax_fem_vtu.point_data['sol']

        print(f"Solution absolute value differs by {np.max(np.absolute(jax_fem_sol - fenicsx_sol))} between FEniCSx and JAX-FEM")
        onptest.assert_array_almost_equal(fenicsx_sol, jax_fem_sol, decimal=3)

        # Compute the top surface area of the cylinder with FEniCSx and JAX-FEM
        fenicsx_surface_area = np.load(f"jax_am/fem/tests/{problem_name}/fenicsx/surface_area.npy")
        jax_fem_area = problem.compute_surface_area(top, sol)[0]
        print(f"Circle area is {np.pi*R**2}")
        print(f"FEniCSx computes approximate area to be {fenicsx_surface_area}")
        print(f"JAX-FEM computes approximate area to be {jax_fem_area}")

        onptest.assert_almost_equal(fenicsx_surface_area, jax_fem_area, decimal=4)


if __name__ == '__main__':
    unittest.main()
