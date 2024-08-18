# Import some generally useful packages.
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio
import matplotlib.pyplot as plt
import time


# Import JAX-FEM specific modules.
from jax_fem.generate_mesh import box_mesh_gmsh, Mesh
from jax_fem.solver import solver
from jax_fem.problem import Problem
from jax_fem.utils import save_sol

from demos.phase_field_fracture.eigen import get_eigen_f_custom


# If you have multiple GPUs, set the one to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# Define some useful directory paths.
crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
vtk_dir = os.path.join(data_dir, 'vtk')
numpy_dir = os.path.join(data_dir, 'numpy')
os.makedirs(numpy_dir, exist_ok=True)


# The bracket operator
# One may define something like 'lambda x: np.maximum(x, 0.)' 
# and 'lambda x: np.minimum(x, 0.)', but it turns out that they may lead to 
# unexpected behaviors. See more discussions and tests in the file 'eigen.py'.
safe_plus = lambda x: 0.5*(x + np.abs(x))
safe_minus = lambda x: 0.5*(x - np.abs(x))


# Define the phase field variable class. 
class PhaseField(Problem):
    # Note how 'get_tensor_map' and 'get_mass_map' specify the corresponding terms 
    # in the weak form. Since the displacement variable u affects the phase field 
    # variable d through the history variable H, we need to set this using 'set_params'.
    def get_tensor_map(self):
        def fn(d_grad, history):
            return G_c*l*d_grad
        return fn

    def get_mass_map(self):
        def fn(d, x, history):
            return G_c/l*d - 2.*(1 - d)*history
        return fn
    
    def set_params(self, history):
        # Override base class method.
        self.internal_vars = [history]


# Define the displacement variable class. 
class Elasticity(Problem):
    # As we previously discussed, native JAX AD may return NaN in the cases 
    # with repeated eigenvalues. We provide two workarounds and users can choose 
    # either one to use. The first option adds a small noise to the strain tensor, 
    # while the second option defines custom derivative rules to properly handle 
    # repeated eigenvalues.
    def get_tensor_map(self):
        _, stress_fn = self.get_maps()
        return stress_fn

    def get_maps(self):
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
    
        key = jax.random.PRNGKey(0)
        noise = jax.random.uniform(key, shape=(self.dim, self.dim), minval=-1e-8, maxval=1e-8)
        noise = np.diag(np.diag(noise))
    
        def stress_fn_opt1(u_grad, d):
            epsilon = strain(u_grad)
            epsilon += noise
            sigma = g(d)*jax.grad(psi_plus)(epsilon) + jax.grad(psi_minus)(epsilon) 
            return sigma
    
        def stress_fn_opt2(u_grad, d):
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

        # Replace stress_fn_opt1 with stress_fn_opt2 will use the second option 
        stress_fn = stress_fn_opt1
    
        def psi_plus_fn(u_grad):
            epsilon = strain(u_grad)
            return psi_plus(epsilon)
    
        return psi_plus_fn, stress_fn
    
    def compute_history(self, sol_u, history_old):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol_u, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        psi_plus_fn, _ = self.get_maps()
        vmap_psi_plus_fn = jax.vmap(jax.vmap(psi_plus_fn))
        psi_plus = vmap_psi_plus_fn(u_grads)
        history = np.maximum(psi_plus, history_old)
        return history
    
    def set_params(self, params):
        # Override base class method.
        sol_d, disp = params
        d = self.fes[0].convert_from_dof_to_quad(sol_d)
        self.internal_vars = [d]
        dirichlet_bc_info[-1][-2] = get_dirichlet_load(disp)
        self.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    
    def compute_traction(self, location_fn, sol_u, sol_d):
        # For post-processing only
        stress = self.get_tensor_map()
        vmap_stress = jax.vmap(stress)
        def traction_fn(u_grads, d_face):
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.fes[0].vec, self.dim)
            # (num_selected_faces, num_face_quads, vec) -> (num_selected_faces*num_face_quads, vec)
            d_face_reshape = d_face.reshape(-1, d_face.shape[-1])
            sigmas = vmap_stress(u_grads_reshape, d_face_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.])
            traction = (sigmas @ normals[None, None, :, None])[:, :, :, 0]
            return traction
    
        boundary_inds = self.fes[0].get_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.fes[0].get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = sol_u[self.fes[0].cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
        selected_cell_sols_d = sol_d[self.fes[0].cells][boundary_inds[:, 0]] # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.fes[0].face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) -> (num_selected_faces, num_face_quads, vec) 
        d_face = np.sum(selected_cell_sols_d[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        traction = traction_fn(u_grads_face, d_face) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
        traction_integral_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))
    
        return traction_integral_val


# Define some material parameters.
# Units are in [kN], [mm] and [s]
G_c = 2.7e-3 # Critical energy release rate [kN/mm] 
E = 210 # Young's modulus [kN/mm^2]
nu = 0.3 # Poisson's ratio
l = 0.02 # Length-scale parameter [mm]
mu = E/(2.*(1. + nu)) # First Lamé parameter
lmbda = E*nu/((1+nu)*(1-2*nu)) # Second Lamé parameter


# Specify mesh-related information (first-order hexahedron element)
Nx, Ny, Nz = 50, 50, 1 
Lx, Ly, Lz = 1., 1., 0.02
meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])


# Define boundary locations.
def y_max(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def y_min(point):
    return np.isclose(point[1], 0., atol=1e-5)


# Create an instance of the phase field problem.
problem_d = PhaseField(mesh, vec=1, dim=3)
sol_d_list = [onp.zeros((len(mesh.points), 1))]
flag = (mesh.points[:, 1] > 0.5*Ly - 0.01*Ly) & (mesh.points[:, 1] < 0.5*Ly + 0.01*Ly) & (mesh.points[:, 0] > 0.5*Lx) 
sol_d_list[0][flag] = 1. # Specify initial crack
sol_d_old = onp.array(sol_d_list[0])


# Create an instance of the displacement problem.
def dirichlet_val(point):
    return 0.

def get_dirichlet_load(disp):
    def val_fn(point):
        return disp
    return val_fn


# disps = 0.01*Ly*np.linspace(0., 1., 101)
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
sol_u_list = [onp.zeros((len(mesh.points), 3))]
sol_u_old = onp.array(sol_u_list[0])
history = onp.zeros((problem_u.fes[0].num_cells, problem_u.fes[0].num_quads))
history_old = onp.array(history)


# Start the major loop for loading steps.
simulation_flag = True
if simulation_flag:
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)
    
    vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
    save_sol(problem_d.fes[0], sol_d_list[0], vtk_path, point_infos=[('u', sol_u_list[0])], cell_infos=[('history', np.mean(history, axis=1))])

    tractions = [0.]
    for i, disp in enumerate(disps[1:]):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")
    
        err = 1.
        tol = 1e-5
        while err > tol:
            print(f"####### max history = {np.max(history)}")
            problem_u.set_params([sol_d_list[0], disp])
            sol_u_list = solver(problem_u)
    
            problem_d.set_params(history)
            sol_d_list = solver(problem_d)
    
            history = problem_u.compute_history(sol_u_list[0], history_old)
            sol_d_list = [onp.maximum(sol_d_list[0], sol_d_old)]
    
            err_u = onp.linalg.norm(sol_u_list[0] - sol_u_old)
            err_d = onp.linalg.norm(sol_d_list[0] - sol_d_old)
            err = onp.maximum(err_u, err_d)
            sol_u_old = onp.array(sol_u_list[0])
            sol_d_old = onp.array(sol_d_list[0])
            print(f"####### err = {err}, tol = {tol}")
            
            # Technically, we are not doing the real 'staggered' scheme. This is an early stop strategy.
            # Comment the following two lines out to get the real staggered scheme, which is more  computationally demanding.
            if True:
                break
    
        history_old = onp.array(history)
     
        traction = problem_u.compute_traction(y_max, sol_u_list[0], sol_d_list[0])/Lz
        tractions.append(traction[-1])
        print(f"Traction force = {traction}")
        vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
        save_sol(problem_d.fes[0], sol_d_list[0], vtk_path, point_infos=[('u', sol_u_list[0])], cell_infos=[('history', np.mean(history, axis=1))])
    
    tractions = np.array(tractions)
    
    results = np.stack((disps, tractions))
    np.save(os.path.join(numpy_dir, 'results.npy'), results)
    
else:
    results = np.load(os.path.join(numpy_dir, 'results.npy'))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(results[0], results[1], color='red', marker='o', markersize=4, linestyle='-') 
    plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
    plt.ylabel(r'Force on top surface [kN]', fontsize=20)
    plt.tick_params(labelsize=18)
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(1e3*onp.hstack((0., onp.cumsum(np.abs(np.diff(results[0]))))), results[0], color='blue', marker='o', markersize=4, linestyle='-') 
    plt.xlabel(r'Time [s]', fontsize=20)
    plt.ylabel(r'Displacement of top surface [mm]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()