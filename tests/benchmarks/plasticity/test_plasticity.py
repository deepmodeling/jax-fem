import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import unittest
import os

from jax_fem.problem import Problem
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from jax_fem.utils import modify_vtu_file, save_sol



class Plasticity(Problem):
    def custom_init(self):
        """Override base class method.
        Initializing total strain and stress.
        """
        self.fe = self.fes[0]
        self.epsilons_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        self.sigmas_old = np.zeros_like(self.epsilons_old)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def get_tensor_map(self):
        """Override base class method.
        """
        _, stress_return_map = self.get_maps()
        return stress_return_map

    def get_maps(self):
        def safe_sqrt(x):  
            """np.sqrt is not differentiable at 0.
            """
            safe_x = np.where(x > 0., np.sqrt(x), 0.)
            return safe_x

        def safe_divide(x, y):
            return np.where(y == 0., 0., x/y)

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress(epsilon):
            E = 70.e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            sig0 = 250.
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev, s_norm)
            return sigma
        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = [self.sigmas_old, self.epsilons_old]

    def compute_avg_stress(self):
        """For post-processing only: Compute volume averaged stress.
        """
        # (num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1) -> (vec, dim)
        sigma = np.sum(self.sigmas_old.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.fe.JxW)
        avg_sigma = sigma/vol
        return avg_sigma


class Test(unittest.TestCase):
    """Test J2-plasticity
    """
    def test_solve_problem(self):
        """Compare FEniCSx solution with JAX-FEM
        """
        problem_name = "plasticity"
        crt_dir = os.path.dirname(__file__)
        fenicsx_vtu_path_raw = os.path.join(crt_dir, "fenicsx/sol_p0_000000.vtu")
        fenicsx_vtu_path = os.path.join(crt_dir, "fenicsx/sol.vtu")
        modify_vtu_file(fenicsx_vtu_path_raw, fenicsx_vtu_path)
        fenicsx_vtu = meshio.read(fenicsx_vtu_path)
        cells = fenicsx_vtu.cells_dict['VTK_LAGRANGE_HEXAHEDRON'] # 'hexahedron'
        points = fenicsx_vtu.points
        mesh = Mesh(points, cells)
        H = 10.

        def top(point):
            return np.isclose(point[2], H, atol=1e-5)

        def bottom(point):
            return np.isclose(point[2], 0., atol=1e-5)

        def dirichlet_val_bottom(point):
            return 0.

        def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn

        disps = np.load(os.path.join(crt_dir, "fenicsx/disps.npy"))

        location_fns = [bottom, bottom, bottom, top, top, top]
        value_fns = [dirichlet_val_bottom, dirichlet_val_bottom, dirichlet_val_bottom,
                     dirichlet_val_bottom, dirichlet_val_bottom, get_dirichlet_top(disps[0])]
        vecs = [0, 1, 2, 0, 1, 2]
        dirichlet_bc_info = [location_fns, vecs, value_fns]

        problem = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
        avg_stresses = []

        for i, disp in enumerate(disps):
            print(f"\nStep {i} in {len(disps)}, disp = {disp}")
            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
            problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
            sol_list = solver(problem)
            problem.update_stress_strain(sol_list[0])
            avg_stress = problem.compute_avg_stress()
            print(avg_stress)
            avg_stresses.append(avg_stress)

        avg_stresses = np.array(avg_stresses)

        jax_vtu_path = os.path.join(crt_dir, "jax_fem/sol.vtu")
        save_sol(problem.fe, sol_list[0], jax_vtu_path)
        jax_fem_vtu = meshio.read(jax_vtu_path)

        jax_fem_sol = jax_fem_vtu.point_data['sol']
        fenicsx_sol = fenicsx_vtu.point_data['sol'].reshape(jax_fem_sol.shape)

        print(f"Solution absolute value differs by {np.max(np.absolute(jax_fem_sol - fenicsx_sol))} between FEniCSx and JAX-FEM")
        onptest.assert_array_almost_equal(fenicsx_sol, jax_fem_sol, decimal=5)

        fenicsx_avg_stresses = np.load(os.path.join(crt_dir, "fenicsx/avg_stresses.npy"))
        jax_fem_avg_stresses = avg_stresses[:, 2, 2]
        np.save(os.path.join(crt_dir, "jax_fem/avg_stresses.npy"), jax_fem_avg_stresses)

        print(f"FEniCSx computes stresses to be {fenicsx_avg_stresses}")
        print(f"JAX-FEM computes stresses to be {jax_fem_avg_stresses}")

        onptest.assert_array_almost_equal(fenicsx_avg_stresses, jax_fem_avg_stresses, decimal=3)


if __name__ == '__main__':
    unittest.main()
