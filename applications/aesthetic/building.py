import sys
sys.path.append("../../..")

import numpy as onp
import jax
import jax.numpy as np
from jax.config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import glob
import matplotlib.pyplot as plt

from jax_fem.core import FEM
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.mma import optimize


from applications.fem.aesthetic.arguments import args, bcolors
from applications.fem.aesthetic.style_loss import style_transfer


# args.Nx = 40
# args.Ny = 200

args.Nx = 100
args.Ny = 500

args.Lx = 1.
args.Ly = 5.


class Elasticity(FEM):
    def custom_init(self):
        self.cell_centroids = onp.mean(onp.take(self.points, self.cells, axis=0), axis=1)

    def get_tensor_map(self):
        """Override base class method.
        """
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 1e5
            Emin = 1e-3*Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*theta[0]**penal
            epsilon = 0.5*(u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma
        return stress

    def set_params(self, params):
        """Override base class method.
        """
        full_params = np.ones((self.num_cells, params.shape[1]))
        full_params = full_params.at[self.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars['laplace'] = [thetas]

    def compute_compliance(self, neumann_fn, sol):
        """Surface integral
        """
        boundary_inds = self.neumann_boundary_inds_list[0]
        _, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :] * self.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
        traction = jax.vmap(jax.vmap(neumann_fn))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val

problem_name = 'building'
data_path = args.output_path
vtk_path = os.path.join(data_path, f'vtk/{problem_name}')

files_vtk = glob.glob(os.path.join(vtk_path, f'*'))
files_jpg = glob.glob(os.path.join(data_path, f'jpg/*'))
for f in files_vtk + files_jpg:
    os.remove(f)

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
meshio_mesh = rectangle_mesh(Nx=args.Nx, Ny=args.Ny, domain_x=args.Lx, domain_y=args.Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def fixed_location(point):
    return np.isclose(point[1], 0., atol=1e-5)
    
def load_location(point):
    flag1 = np.isclose(point[0], 0., atol=1e-5)
    flag2 = np.isclose(point[0], args.Lx, atol=1e-5)
    return flag1 | flag2

def dirichlet_val(point):
    return 0.

def neumann_val(point):
    base_force = 10.
    # traction = base_force*point[1]/args.Ly
    traction = base_force*point[1]/args.Ly
    return np.array([traction, 0.])


def flex_location(point):
    flag1 = point[0] > 0.025*args.Lx
    flag2 = point[0] < args.Lx - 0.025*args.Lx
    flag3 = point[1] > 0.025*args.Lx
    flag4 = point[1] < args.Ly - 0.025*args.Lx 
    return flag1 & flag2 & flag3 & flag4 | True



dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
neumann_bc_info = [[load_location], [neumann_val]]
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

problem.flex_inds = np.argwhere(jax.vmap(flex_location)(problem.cell_centroids)).reshape(-1)
 
fwd_pred = ad_wrapper(problem, linear=True, use_petsc=False)

def J_total(params):
    """J(u(theta), theta)
    """     
    sol = fwd_pred(params)
    compliance = problem.compute_compliance(neumann_val, sol)
    return compliance

outputs = []
def output_sol(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol = fwd_pred(params)
    vtu_path = os.path.join(vtk_path, f'sol_{output_sol.counter:03d}.vtu')
    save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, cell_infos=[('theta', 1. - problem.full_params[:, 0])])
    print(f"{bcolors.HEADER}Case = {problem_name}, compliance = {obj_val}{bcolors.ENDC}")
    outputs.append(obj_val)
    output_sol.counter += 1
output_sol.counter = 0

def rho_full2flex(rho_full):
    return rho_full[problem.flex_inds]

def rho_flex2full(rho_flex):
    rho_full = np.ones((problem.num_cells, 1))
    rho_full = rho_full.at[problem.flex_inds].set(rho_flex)
    return rho_full

vf = 0.5
rho_ini = vf*np.ones((len(problem.flex_inds), 1))
numConstraints = 1

config.update("jax_enable_x64", False)
style_value_and_grad, initial_loss = style_transfer(problem, rho_flex2full(rho_ini), image_path='styles/moha.png', reverse=True)
config.update("jax_enable_x64", True)

 
def objectiveHandleCompliance(rho):
    """MMA solver requires (J, dJ) as inputs
    J has shape ()
    dJ has shape (...) = rho.shape
    """
    J_to, dJ_to = jax.value_and_grad(J_total)(rho)
    output_sol(rho, J_to)
    return J_to, dJ_to

def objectiveHandleStyle(rho):
    """MMA solver requires (J, dJ) as inputs
    J has shape ()
    dJ has shape (...) = rho.shape
    """
    J_style, dJ_style = style_value_and_grad(rho, output_sol.counter)
    output_sol(rho, J_style)
    return J_style, dJ_style


def consHandle(rho, epoch):
    """MMA solver requires (c, dc) as inputs
    c should have shape (numConstraints,)
    gradc should have shape (numConstraints, ...)
    """
    def computeGlobalVolumeConstraint(rho):
        g = np.mean(rho)/vf - 1.
        return g
    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc


optimizationParamsCompliance = {'maxIters':5, 'movelimit':0.05}
optimizationParamsStyle = {'maxIters':10, 'movelimit':0.1}

rho_flex = rho_ini
for i in range(3):
    rho_flex = optimize(problem, rho_flex, optimizationParamsCompliance, objectiveHandleCompliance, consHandle, numConstraints)
    # rho_full = rho_flex2full(rho_flex)
    # rho_full = optimize(problem, rho_full, optimizationParamsStyle, objectiveHandleStyle, consHandle, numConstraints)
    # rho_flex = rho_full2flex(rho_full)

for i in range(10):
    rho_flex = optimize(problem, rho_flex, {'maxIters':10, 'movelimit':0.05}, objectiveHandleCompliance, consHandle, numConstraints)
 

print(f"As a reminder, compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")

