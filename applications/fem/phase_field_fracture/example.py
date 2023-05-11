import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.core import FEM
from jax_am.fem.utils import save_sol


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')


class PhaseField(FEM):
    def custom_init(self):
        # self.G_c = 100.
        # self.l = 0.01
        self.G_c = 2.7e-3
        self.l = 0.02

    def get_tensor_map(self):
        def fn(d_grad):
            return self.G_c*self.l*d_grad
        return fn
 
    def get_mass_map(self):
        def fn(d, history):
            return self.G_c/self.l*d - 2.*(1 - d)*history
        return fn

    def set_params(self, history):
        self.internal_vars['mass'] = [history]


class Elasticity(FEM):
    def custom_init(self):
        # self.E = 70e9
        # self.nu = 0.3
        self.E = 210
        self.nu = 0.3

    def get_tensor_map(self):
        _, stress_fn = self.get_maps()
        return stress_fn

    def get_maps(self):
        mu = self.E/(2.*(1. + self.nu))
        lmbda = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        bulk_mod = lmbda + 2.*mu/self.dim

        def psi_plus(epsilon):
            eps_dev = epsilon - 1./self.dim*np.trace(epsilon)*np.eye(self.dim)
            tr_epsilon_plus = np.maximum(np.trace(epsilon), 0.)
            return bulk_mod/2.*tr_epsilon_plus**2 + mu*np.sum(eps_dev*eps_dev)
            
        def psi_minus(epsilon):
            tr_epsilon_minus = np.minimum(np.trace(epsilon), 0.)
            return bulk_mod/2.*tr_epsilon_minus**2

        def psi(epsilon):
            eps_dev = epsilon - 1./self.dim*np.trace(epsilon)*np.eye(self.dim)
            return bulk_mod/2.*np.trace(epsilon)**2 + mu*np.sum(eps_dev*eps_dev)

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress_fn(u_grad, d):
            epsilon = strain(u_grad)
            sigma = ((1 - d[0])**2 + 1e-3) * jax.grad(psi_plus)(epsilon) + jax.grad(psi_minus)(epsilon) 
            # sigma = jax.grad(psi_plus)(epsilon) + jax.grad(psi_minus)(epsilon) 
            # sigma = jax.grad(psi)(epsilon)
            return sigma

        def psi_plus_fn(u_grad):
            epsilon = strain(u_grad)
            return psi_plus(epsilon)

        return psi_plus_fn, stress_fn

    def compute_history(self, sol_u, history_old):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol_u, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        psi_plus_fn, _ = self.get_maps()
        vmap_psi_plus_fn = jax.vmap(jax.vmap(psi_plus_fn))
        psi_plus = vmap_psi_plus_fn(u_grads)
        history = np.maximum(psi_plus, history_old)
        return history

    def set_params(self, sol_d):
        d = self.convert_from_dof_to_quad(sol_d)
        self.internal_vars['laplace'] = [d]
    

def simulation():
    vtk_dir = os.path.join(data_dir, 'vtk')
    problem_name = 'example'
    Nx, Ny, Nz = 50, 50, 1
    Lx, Ly, Lz = 1., 1., 0.02
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def y_max(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def y_min(point):
        return np.isclose(point[1], 0., atol=1e-5)
 
    def dirichlet_val(point):
        return 0.

    def get_dirichlet_load(disp):
        def val_fn(point):
            return disp
        return val_fn

    problem_d = PhaseField(mesh, vec=1, dim=3)
    sol_d = onp.zeros((len(mesh.points), 1))
    flag1 = onp.logical_and(mesh.points[:, 1] > 0.5*Ly - 0.01*Ly, mesh.points[:, 1] < 0.5*Ly + 0.01*Ly)
    flag2 = onp.logical_and(mesh.points[:, 0] > 0.5*Lx, flag1)
    sol_d[flag2] = 1.
    sol_d_old = onp.array(sol_d)

    disps = np.linspace(0., 0.01*Ly, 101)
    # disps = np.linspace(0., 0.1*Ly, 11)
    location_fns = [y_min, y_min, y_min, y_max, y_max, y_max]
    vecs = [0, 1, 2, 0, 1, 2]
    value_fns = [dirichlet_val, dirichlet_val, dirichlet_val, 
                 dirichlet_val, get_dirichlet_load(disps[0]), dirichlet_val]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem_u = Elasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    sol_u = onp.zeros((len(mesh.points), 3))
    sol_u_old = onp.array(sol_u)
    history = onp.zeros((problem_u.num_cells, problem_u.num_quads))
    history_old = onp.array(history)

    files = glob.glob(os.path.join(vtk_dir, f'{problem_name}/*'))
    for f in files:
        os.remove(f)

    vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{0:05d}.vtu")
    save_sol(problem_d, sol_d, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('history', np.mean(history, axis=1))])

    for i, disp in enumerate(disps[1:]):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")
        dirichlet_bc_info[-1][-2] = get_dirichlet_load(disp)
        problem_u.update_Dirichlet_boundary_conditions(dirichlet_bc_info)

        err = 1.
        tol = 1e-5
        while err > tol:

            print(f"####### max history = {np.max(history)}")
            problem_d.set_params(history)
            problem_u.set_params(sol_d)
            sol_u = solver(problem_u, use_petsc=True)
            sol_d = solver(problem_d, use_petsc=True)
            history = problem_u.compute_history(sol_u, history_old)

            sol_d = onp.maximum(sol_d, sol_d_old)
 
            err_u = onp.linalg.norm(sol_u - sol_u_old)
            err_d = onp.linalg.norm(sol_d - sol_d_old)
            err = onp.maximum(err_u, err_d)
            sol_u_old = onp.array(sol_u)
            sol_d_old = onp.array(sol_d)
            print(f"####### err = {err}, tol = {tol}")

            if True:
                break

        history_old = onp.array(history)

        if (i + 1) % 1 == 0:
            vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{i + 1:05d}.vtu")
            save_sol(problem_d, sol_d, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('history', np.mean(history, axis=1))])
 

if __name__ == "__main__":
    simulation()
