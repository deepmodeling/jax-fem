import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.models import LinearElasticity
from jax_am.fem.solver import solver
from jax_am.fem.utils import modify_vtu_file, save_sol


class Test(unittest.TestCase):
    """Test linear elasticity with cubic mesh
    """
    def test_solve_problem(self):
        """Compare FEniCSx solution with JAX-FEM
        """
        problem_name = "linear_elasticity_cube"
        crt_dir = os.path.dirname(__file__)
        fenicsx_vtu_path_raw = os.path.join(crt_dir, "fenicsx/sol_p0_000000.vtu")
        fenicsx_vtu_path = os.path.join(crt_dir, "fenicsx/sol.vtu")
        modify_vtu_file(fenicsx_vtu_path_raw, fenicsx_vtu_path)
        fenicsx_vtu = meshio.read(fenicsx_vtu_path)
        cells = fenicsx_vtu.cells_dict['VTK_LAGRANGE_HEXAHEDRON'] # 'hexahedron'
        points = fenicsx_vtu.points
        mesh = Mesh(points, cells)
        L = 1.

        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)

        def right(point):
            return np.isclose(point[0], L, atol=1e-5)

        def dirichlet_val(point):
            return 1.

        def neumann_val(point):
            return np.array([10., 0., 0.])

        def body_force(point):
            return np.array([0., 10., 10.])

        location_fns = [left, left, left]
        value_fns = [dirichlet_val, dirichlet_val, dirichlet_val]
        vecs = [0, 1, 2]
        dirichlet_bc_info = [location_fns, vecs, value_fns]

        neumann_bc_info = [[right], [neumann_val]]

        problem = LinearElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, 
                                   neumann_bc_info=neumann_bc_info, source_info=body_force)
        sol = solver(problem)

        jax_vtu_path = os.path.join(crt_dir, "jax_fem/sol.vtu")
        save_sol(problem, sol, jax_vtu_path)
        jax_fem_vtu = meshio.read(jax_vtu_path)

        jax_fem_sol = jax_fem_vtu.point_data['sol']
        fenicsx_sol = fenicsx_vtu.point_data['sol'].reshape(jax_fem_sol.shape)

        print(f"Solution absolute value differs by {np.max(np.absolute(jax_fem_sol - fenicsx_sol))} between FEniCSx and JAX-FEM")
        onptest.assert_array_almost_equal(fenicsx_sol, jax_fem_sol, decimal=5)


if __name__ == '__main__':
    unittest.main()
