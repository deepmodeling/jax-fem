"""Try the idea of using numba for FEM global matrix assembly.
The sparse matrix in the form of (i, j, value) tuples can be obtained.
Still not fast enough, can't deal with problem with dofs of 50x50x50x3.
Though assembly is slow, linear solver is pretty fast.

Tianju Xue
09/13/2022
At Northwestern U
"""
import numpy as onp
import jax
import jax.numpy as np
import os
import sys
import time
import meshio
import matplotlib.pyplot as plt
from functools import partial
import gc
from jax_am.fem.generate_mesh import box_mesh, cylinder_mesh
from jax_am.fem.jax_fem import LinearElasticity, FEM, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

from numba import jit, njit, prange

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


@njit
def stress(u_grad):
    E = 70e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))
    epsilon = 0.5*(u_grad + u_grad.T)
    sigma = lmbda*onp.trace(epsilon)*onp.eye(3) + 2*mu*epsilon
    return sigma


@njit
def to_dof_ind(i, v, vec):
    return vec*i + v


@njit
def construct_phi_grad(phi_grad, v):
    phi_grad_tensor = onp.zeros((3, 3))
    phi_grad_tensor[v] = phi_grad
    return phi_grad_tensor


@njit()
def assemble(cells, shape_grads, JxW, num_nodes, vec, num_quads):
    inds_I = []
    inds_J = []
    vals = []
    for c in range(len(cells)):
        if c % 1000 == 0:
            print(f"c = {c}, len(cells) = {len(cells)}")
        for i in range(num_nodes):
            for v in range(vec):
                index_I = to_dof_ind(cells[c, i], v, vec)
                for j in range(num_nodes):
                    for w in range(vec):
                        index_J = to_dof_ind(cells[c, j], w, vec)
                        val = 0.
                        for q in range(num_quads):
                            phi_grad_i = construct_phi_grad(shape_grads[c, q, i], v)
                            phi_grad_j = construct_phi_grad(shape_grads[c, q, j], w)
                            physics_j = stress(phi_grad_j)
                            val += onp.sum(phi_grad_i * physics_j) * JxW[c, q]
                        inds_I.append(index_I)
                        inds_J.append(index_J)
                        vals.append(val)
    inds_I = onp.array(inds_I)
    inds_J = onp.array(inds_J)
    vals = onp.array(vals)
    print(f"Finished assembling")
    return inds_I, inds_J, vals


class NumbdaLaplace(FEM):
    def __init__(self, mesh, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info) 
        # Some pre-computations   
        self.body_force = self.compute_body_force(source_info)
        self.neumann = self.compute_Neumann_integral(neumann_bc_info)
        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]
        self.inds_I, self.inds_J, self.vals = assemble(onp.array(self.cells), onp.array(self.shape_grads), onp.array(self.JxW), self.num_nodes, self.vec, self.num_quads)
        print(f"inds_I.shape = {self.inds_I.shape}")
   
    def compute_residual(self, sol):
        dofs = sol.reshape(-1)
        weak_form = np.zeros_like(dofs)
        weak_form = weak_form.at[self.inds_I].add(dofs[self.inds_J] * self.vals).reshape(sol.shape)
        res = weak_form - self.body_force - self.neumann
        return res 

    def compute_body_force(self, source_info):
        rhs = np.zeros((self.num_total_nodes, self.vec))
        if source_info is not None:
            body_force_fn = source_info
            physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim) 
            body_force = jax.vmap(jax.vmap(body_force_fn))(physical_quad_points) # (num_cells, num_quads, vec) 
            assert len(body_force.shape) == 3
            v_vals = np.repeat(self.shape_vals[None, :, :, None], self.num_cells, axis=0) # (num_cells, num_quads, num_nodes, 1)
            v_vals = np.repeat(v_vals, self.vec, axis=-1) # (num_cells, num_quads, num_nodes, vec)
            # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
            rhs_vals = np.sum(v_vals * body_force[:, :, None, :] * self.JxW[:, :, None, None], axis=1).reshape(-1, self.vec) 
            rhs = rhs.at[self.cells.reshape(-1)].add(rhs_vals) 
        return rhs

    def compute_Neumann_integral(self, neumann_bc_info):
        integral = np.zeros((self.num_total_nodes, self.vec))
        if neumann_bc_info is not None:
            location_fns, value_fns = neumann_bc_info
            integral = np.zeros((self.num_total_nodes, self.vec))
            boundary_inds_list = self.Neuman_boundary_conditions_inds(location_fns)
            traction_list = self.Neuman_boundary_conditions_vals(value_fns, boundary_inds_list)
            for i, boundary_inds in enumerate(boundary_inds_list):
                traction = traction_list[i]
                _, nanson_scale = self.get_face_shape_grads(boundary_inds) # (num_selected_faces, num_face_quads)
                # (num_faces, num_face_quads, num_nodes) ->  (num_selected_faces, num_face_quads, num_nodes)
                v_vals = np.take(self.face_shape_vals, boundary_inds[:, 1], axis=0)
                v_vals = np.repeat(v_vals[:, :, :, None], self.vec, axis=-1) # (num_selected_faces, num_face_quads, num_nodes, vec)
                subset_cells = np.take(self.cells, boundary_inds[:, 0], axis=0) # (num_selected_faces, num_nodes)
                # (num_selected_faces, num_nodes, vec) -> (num_selected_faces*num_nodes, vec)
                int_vals = np.sum(v_vals * traction[:, :, None, :] * nanson_scale[:, :, None, None], axis=1).reshape(-1, self.vec) 
                integral = integral.at[subset_cells.reshape(-1)].add(int_vals)   
        return integral


class NumbaLinearElasticity(NumbdaLaplace):
    def __init__(self, name, mesh, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info) 
    

def linear_elasticity_problem():
    problem_name = 'numba_test'
    meshio_mesh = box_mesh(20, 20, 20)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
 
    problem = NumbaLinearElasticity(problem_name, mesh, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, use_linearization_guess=False)
 

if __name__ == "__main__":
    linear_elasticity_problem()

