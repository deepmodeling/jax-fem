import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, ad_wrapper
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.fem.mma import optimize
from jax_am.common import rectangle_mesh


class Elasticity(FEM):
    def custom_init(self):
        """Override base class method.
        Modify self.flex_inds so that location-specific TO can be realized.
        """
        self.flex_inds = np.arange(len(self.cells))

    def get_tensor_map(self):
        """Override base class method.
        """
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 70.e3
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

data_path = os.path.join(os.path.dirname(__file__), 'data') 
files = glob.glob(os.path.join(data_path, f'vtk/*'))
for f in files:
    os.remove(f)

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.
meshio_mesh = rectangle_mesh(Nx=60, Ny=30, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)
    
def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

def dirichlet_val(point):
    return 0.

def neumann_val(point):
    return np.array([0., -100.])

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
neumann_bc_info = [[load_location], [neumann_val]]
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

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
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"compliance = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1
output_sol.counter = 0

def objectiveHandle(rho):
    """MMA solver requires (J, dJ) as inputs
    J has shape ()
    dJ has shape (...) = rho.shape
    """
    J, dJ = jax.value_and_grad(J_total)(rho)
    output_sol(rho, J)
    return J, dJ

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

vf = 0.5
optimizationParams = {'maxIters':51, 'movelimit':0.1}
rho_ini = vf*np.ones((len(problem.flex_inds), 1))
numConstraints = 1
optimize(problem, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints)
print(f"As a reminder, compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")

obj = onp.array(outputs)
plt.figure(figsize=(10, 8))
plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
plt.show()
