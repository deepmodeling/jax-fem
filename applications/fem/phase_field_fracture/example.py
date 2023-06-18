import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio
import matplotlib.pyplot as plt
import time

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.core import FEM
from jax_am.fem.utils import save_sol

from applications.fem.phase_field_fracture.eigen import get_eigen_f_custom

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
vtk_dir = os.path.join(data_dir, 'vtk')
numpy_dir = os.path.join(data_dir, 'numpy')
os.makedirs(numpy_dir, exist_ok=True)


safe_plus = lambda x: 0.5*(x + np.abs(x))
safe_minus = lambda x: 0.5*(x - np.abs(x))


class PhaseField(FEM):
    def get_tensor_map(self):
        def fn(d_grad):
            return G_c*l*d_grad
        return fn
 
    def get_mass_map(self):
        def fn(d, history):
            return G_c/l*d - 2.*(1 - d)*history
        return fn

    def set_params(self, history):
        self.internal_vars['mass'] = [history]


class Elasticity(FEM):
    def get_tensor_map(self):
        _, stress_fn = self.get_maps()
        return stress_fn

    def get_maps(self):
        return self.get_maps_1()

    def get_maps_1(self):
        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def psi_plus(epsilon):
            eigen_vals, eigen_evecs = np.linalg.eigh(epsilon)
            tr_epsilon_plus = safe_plus(np.trace(epsilon))
            return lmbda/2.*tr_epsilon_plus**2 + mu*np.sum(safe_plus(eigen_vals)**2)

        def psi_minus(epsilon):
            eigen_vals, eigen_evecs = np.linalg.eigh(epsilon)
            tr_epsilon_minus = safe_minus(np.trace(epsilon))
            return lmbda/2.*tr_epsilon_minus**2 + mu*np.sum(safe_minus(eigen_vals)**2) 

        def g(d):
            return (1 - d[0])**2 + 1e-3

        def stress_fn_opt1(u_grad, d):
            epsilon = strain(u_grad)

            def fn(x):
                return 2*mu*(g(d) * safe_plus(x) + safe_minus(x))
            eigen_f = get_eigen_f_custom(fn)

            tr_epsilon_plus = safe_plus(np.trace(epsilon))
            tr_epsilon_minus = safe_minus(np.trace(epsilon))
            sigma1 = lmbda*(g(d)*tr_epsilon_plus + tr_epsilon_minus)*np.eye(self.dim) 

            sigma2 = eigen_f(epsilon)
            sigma = sigma1 + sigma2

            return sigma        

        key = jax.random.PRNGKey(0)
        noise = jax.random.uniform(key, shape=(self.dim, self.dim), minval=-1e-8, maxval=1e-8)
        noise = np.diag(np.diag(noise))

        def stress_fn_opt2(u_grad, d):
            epsilon = strain(u_grad)
            epsilon += noise
            sigma = g(d)*jax.grad(psi_plus)(epsilon) + jax.grad(psi_minus)(epsilon) 
            return sigma

        stress_fn = stress_fn_opt2

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

    def set_params(self, params):
        sol_d, disp = params
        d = self.convert_from_dof_to_quad(sol_d)
        self.internal_vars['laplace'] = [d]
        dirichlet_bc_info[-1][-2] = get_dirichlet_load(disp)
        self.update_Dirichlet_boundary_conditions(dirichlet_bc_info)

    def compute_traction(self, location_fn, sol_u, sol_d):
        """For post-processing only
        """
        stress = self.get_tensor_map()
        vmap_stress = jax.vmap(stress)
        def traction_fn(u_grads, d_face):
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
            # (num_selected_faces, num_face_quads, vec) -> (num_selected_faces*num_face_quads, vec)
            d_face_reshape = d_face.reshape(-1, d_face.shape[-1])
            sigmas = vmap_stress(u_grads_reshape, d_face_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.])
            traction = (sigmas @ normals[None, None, :, None])[:, :, :, 0]
            return traction

        boundary_inds = self.get_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = sol_u[self.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)

        selected_cell_sols_d = sol_d[self.cells][boundary_inds[:, 0]] # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) -> (num_selected_faces, num_face_quads, vec) 
        d_face = np.sum(selected_cell_sols_d[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
 
        traction = traction_fn(u_grads_face, d_face) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
        traction_integral_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))

        return traction_integral_val


# Units are in [kN], [mm] and [s]
G_c = 2.7e-3 # Critical energy release rate [kN/mm] 
E = 210 # Young's modulus [kN/mm^2]
nu = 0.3 # Poisson's ratio
l = 0.02 # Length-scale parameters= [mm]
mu = E/(2.*(1. + nu)) 
lmbda = E*nu/((1+nu)*(1-2*nu))

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
flag = (mesh.points[:, 1] > 0.5*Ly - 0.01*Ly) & (mesh.points[:, 1] < 0.5*Ly + 0.01*Ly) & (mesh.points[:, 0] > 0.5*Lx) 
sol_d[flag] = 1.
sol_d_old = onp.array(sol_d)

# disps = np.linspace(0., 0.01*Ly, 101)

disps = 0.01*Ly*np.hstack((np.linspace(0, 0.6, 21),
                           np.linspace(0.6, 0.8, 121),
                           np.linspace(0.8, -0.4, 61),
                           np.linspace(-0.4, 0.8, 61),
                           np.linspace(0.8, 1., 121)))

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


simulation_flag = False
if simulation_flag:
    start = time.time()

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
    save_sol(problem_d, sol_d, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('history', np.mean(history, axis=1))])

    tractions = [0.]
    for i, disp in enumerate(disps[1:]):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")

        err = 1.
        tol = 1e-5
        while err > tol:
            print(f"####### max history = {np.max(history)}")
            problem_u.set_params([sol_d, disp])
            sol_u = solver(problem_u, use_petsc=False)

            problem_d.set_params(history)
            sol_d = solver(problem_d, use_petsc=False)

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
     
        traction = problem_u.compute_traction(y_max, sol_u, sol_d)/Lz
        tractions.append(traction[-1])
        print(f"Traction force = {traction}")
        vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
        save_sol(problem_d, sol_d, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('history', np.mean(history, axis=1))])

    tractions = np.array(tractions)

    results = np.stack((disps, tractions))
    np.save(os.path.join(numpy_dir, 'results.npy'), results)

    print(np.stack((np.arange(len(tractions)), tractions)).T)
    end = time.time()
    print(f"Wall time = {end - start} for this simulation.")
else:
    results = np.load(os.path.join(numpy_dir, 'results.npy'), )

    fig = plt.figure(figsize=(10, 8))
    plt.plot(results[0], results[1], color='red', marker='o', markersize=4, linestyle='-') 
    plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
    plt.ylabel(r'Force on top surface [kN]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()