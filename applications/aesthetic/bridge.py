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



args.Nx = 300
args.Ny = 100

args.Lx = 3.
args.Ly = 1.


class Elasticity(FEM):
    def custom_init(self):
        self.cell_centroids = onp.mean(onp.take(self.points, self.cells, axis=0), axis=1)

    def get_tensor_map(self):
        """Override base class method.
        """
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 1e3
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

    def compute_compliance(self, sol):
        body_force_fn = self.source_info
        physical_quad_points = self.get_physical_quad_points()  # (num_cells, num_quads, dim)
        body_force = jax.vmap(jax.vmap(body_force_fn))(physical_quad_points)  # (num_cells, num_quads, vec)
        sol_quads = self.convert_from_dof_to_quad(sol) # (num_cells, num_quads, vec)
        return np.sum(body_force*sol_quads*self.JxW[:, :, None])


problem_name = 'bridge'
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
    flag1 = np.isclose(point[0], 0., atol=1e-5) & (point[1] < 0.05*args.Ly)
    flag2 = np.isclose(point[0], args.Lx, atol=1e-5) & (point[1] < 0.05*args.Ly)
    flag3 = np.isclose(point[1], 0., atol=1e-5) & (point[0] < 0.05*args.Lx)
    flag4 = np.isclose(point[1], 0., atol=1e-5) & (point[0] > (1 - 0.05)*args.Lx)
    return flag1 | flag2 | flag3 | flag4

def body_force(point):
    flag = (point[1] >= 0.3*args.Ly) & (point[1] <= 0.4*args.Ly)
    base_force = 1e2
    force = np.where(flag, -base_force, 0.)
    return np.array([0., force])

def dirichlet_val(point):
    return 0.

def flex_location(point):
    return ~((point[1] >= 0.3*args.Ly) & (point[1] <= 0.4*args.Ly))

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, source_info=body_force)

problem.flex_inds = np.argwhere(jax.vmap(flex_location)(problem.cell_centroids)).reshape(-1)
 
fwd_pred = ad_wrapper(problem, linear=True, use_petsc=False)

def J_total(params):
    """J(u(theta), theta)
    """     
    sol = fwd_pred(params)
    compliance = problem.compute_compliance(sol)
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


vf = 0.3
rho_ini = vf*np.ones((len(problem.flex_inds), 1))
numConstraints = 1

def objectiveHandleCompliance(rho):
    """MMA solver requires (J, dJ) as inputs
    J has shape ()
    dJ has shape (...) = rho.shape
    """
    J_to, dJ_to = jax.value_and_grad(J_total)(rho)
    output_sol(rho, J_to)
    return J_to, dJ_to

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


optimize(problem, rho_ini, {'maxIters':150, 'movelimit':0.1}, objectiveHandleCompliance, consHandle, numConstraints)
 

print(f"As a reminder, compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")

