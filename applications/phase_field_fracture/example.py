# Import some generally useful packages.
import os
import glob
import time

import jax
import jax.numpy as np

import numpy as onp
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem import logger
logger.setLevel('ERROR')
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.solver import solver
from jax_fem.problem import Problem
from jax_fem.utils import save_sol

from applications.phase_field_fracture.eigen import get_eigen_f_custom

# If you have multiple GPUs, set the one to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Define some useful directory paths.
crt_file_path = os.path.dirname(__file__)
input_dir = os.path.join(crt_file_path, 'input')
output_dir = os.path.join(crt_file_path, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vtk_dir, exist_ok=True)


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
            return (1 - d[0])**2
    
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
        dirichlet_bc_info[-1][-1] = get_dirichlet_load(disp)
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
            normals = np.array([0., 1.])
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
l = 0.0075 # Length-scale parameter [mm]
mu = E/(2.*(1. + nu)) # First Lamé parameter
lmbda = E*nu/((1+nu)*(1-2*nu)) # Second Lamé parameter


# Specify mesh-related information (bilinear quadrilateral element)
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
npz_data = onp.load(os.path.join(input_dir, 'mesh.npz'))
points = npz_data['points']
cells = npz_data['cells']
mesh = Mesh(points, cells, ele_type)


# Define boundary locations (for u).
def top(point):
    return np.isclose(point[1], 0.5, atol=1e-5)

def bot(point):
    return np.isclose(point[1], -0.5, atol=1e-5)

def bot_left(point):
    return np.logical_and(np.isclose(point[0], -0.5, atol=1e-5),
                          np.isclose(point[1], -0.5, atol=1e-5))

def zero_dirichlet_val(point):
    return 0.

def get_dirichlet_load(disp):
    def val_fn(point):
        return disp
    return val_fn

dirichlet_bc_info = [[bot_left, bot, top], [0, 1, 1], 
                  [zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_load(0.)]]


# Create an instance of the phase field problem.
problem_d = PhaseField(mesh, vec=1, dim=2, ele_type=ele_type)
sol_d_list = [onp.zeros((len(mesh.points), 1))]
sol_d_old = onp.array(sol_d_list[0])


# Create an instance of the displacement problem.
problem_u = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
sol_u_list = [onp.zeros((len(mesh.points), 2))]
sol_u_old = onp.array(sol_u_list[0])
history = onp.zeros((problem_u.fes[0].num_cells, problem_u.fes[0].num_quads))
history_old = onp.array(history)


# Start the major loop for loading steps.
simulation_flag = True
if simulation_flag:
    # clear
    files = glob.glob(os.path.join(vtk_dir, './*'))
    for f in files:
        os.remove(f)
    
    # initial state
    vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtk")
    save_sol(problem_d.fes[0], sol_d_list[0], vtk_path, point_infos=[('u', sol_u_list[0])], cell_infos=[('history', np.mean(history, axis=1))])
    
    # displacement loadings
    totaltime = 0.8
    dt = 0.01
    du = 0.01
    times = onp.arange(0, totaltime+dt, dt)
    disps = times * du
    
    # solve
    tractions = [0.]
    start = time.time()
    for i, disp in enumerate(disps[1:], start=1):
        print(f"\nStep {i} in {len(disps)-1}, disp = {disp:.4e}")
        err = 1.
        tol = 1e-5
        while err > tol:
            logger.debug(f"####### max history = {np.max(history)}")
            # solve for u
            problem_u.set_params([sol_d_list[0], disp])
            sol_u_list = solver(problem_u, solver_options={'umfpack_solver':{}})
            # history
            history = problem_u.compute_history(sol_u_list[0], history_old)
            # solve for d
            problem_d.set_params(history)
            sol_d_list = solver(problem_d, solver_options={'umfpack_solver':{}})
            # error
            err_u = onp.linalg.norm(sol_u_list[0] - sol_u_old)/onp.linalg.norm(sol_u_list[0])
            err_d = onp.linalg.norm(sol_d_list[0] - sol_d_old)/onp.linalg.norm(sol_d_list[0])
            err = onp.maximum(err_u, err_d)
            # update previous state
            sol_u_old = onp.array(sol_u_list[0])
            sol_d_old = onp.array(sol_d_list[0])
            logger.debug(f"####### err = {err:.4e}, tol = {tol}")
        # update history
        history_old = onp.array(history)
        # compute tractions
        traction = problem_u.compute_traction(top, sol_u_list[0], sol_d_list[0])
        tractions.append(traction[-1])
        print(f"Traction force = {traction[-1]:.4e}")
        vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtk")
        save_sol(problem_d.fes[0], sol_d_list[0], vtk_path, point_infos=[('u', sol_u_list[0])], cell_infos=[('history', np.mean(history, axis=1))])
    end = time.time()
    print(f"Time cost: {end-start:.2f} seconds")
    
    # save data
    tractions = np.array(tractions)
    np.savez(os.path.join(output_dir, 'sol.npz'), disps=disps, forces=tractions)


# Force-displacement curve
sol = np.load(os.path.join(output_dir, 'sol.npz'))
sol_ref = onp.load(os.path.join(input_dir, 'sol_ref.npz'))
fig = plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 2
plt.plot(sol_ref['disps'], sol_ref['forces'], 'r-', marker='o', ms=6, label='Ref [4]')
plt.plot(sol['disps'], sol['forces'], 'b--', marker='o', ms=6, label='JAX-FEM') 
plt.xlabel(r'Displacement of top surface [mm]')
plt.ylabel(r'Force on top surface [kN]')
plt.legend(frameon=False, loc='upper left')
plt.tick_params()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ForceVsDisp.png'), dpi=600, format='png')
plt.show()