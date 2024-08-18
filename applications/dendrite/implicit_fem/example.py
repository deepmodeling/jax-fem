"""
Implicit finite element solver
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem

from jax import config
config.update("jax_enable_x64", True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)


def safe_arctan2(y, x):
    """When y and x are both zero, gradient needs special care.

    For reference, analytical gradient of function np.arctan2(y, x) is
    grad tan^-1 (y/x) = [-y/(x^2 + y^2), x/(x^2 + y^2)]

    The following tests will produce 

    x, y = 0., 0.
    jax.jacrev(np.arctan2, argnums=0)(y, x) # nan
    jax.jacfwd(np.arctan2, argnums=0)(y, x) # nan
    jax.jacrev(safe_atan2, argnums=0)(y, x) # nan
    jax.jacfwd(safe_atan2, argnums=0)(y, x) # 0. (This is what we want. Used by jax_fem/problem.py)
    """
    safe_theta = np.where((y == 0.) & (x == 0.), 0., np.arctan2(y, x))
    return safe_theta
 

class Solidification(Problem):
    def custom_init(self, params):
        self.fe_p = self.fes[0]
        self.fe_T = self.fes[1]
        self.params = params

    def get_universal_kernel(self):
        def eps_fn(theta):
            eps_bar = self.params['eps_bar']
            delta = self.params['delta']
            J = self.params['J']
            theta0 = np.pi/2
            return eps_bar*(1 + delta*np.cos(J*(theta - theta0)))

        eps_grad = jax.grad(eps_fn)
        vmap_eps_fn = jax.vmap(eps_fn)
        vmap_eps_grad = jax.vmap(eps_grad)
        vmap_safe_arctan2 = jax.vmap(safe_arctan2)

        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_sol_p_old, cell_sol_T_old, p_old, T_old, chi):
            """
            Handles the weak form with one cell.
            Assume trial function (p, T), test function (q, S)

            cell_sol_flat: (num_nodes*vec + ...,)
            cell_sol_list: [(num_nodes, vec), ...]
            x: (num_quads, dim)
            cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_JxW: (num_vars, num_quads)
            cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            p_old: (num_quads,)
            T_old: (num_quads,)
            chi: (num_quads,)

            You may define fully implicit weak form, but that doesn't converge.
            Therefore, some of the terms are changed to explicit for good reason.
            In summary, only the "diffusion" like term is kept implict, and other terms are explicit.
            The entire weak form will be linear.
            """
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            # cell_sol_p: (num_nodes_p, vec), cell_sol_T: (num_nodes, vec)
            cell_sol_p, cell_sol_T = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p, cell_shape_grads_T = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p, cell_v_grads_JxW_T = cell_v_grads_JxW_list
            cell_JxW_p, cell_JxW_T = cell_JxW[0], cell_JxW[1]

            # Handles the term `inner(..., grad(q)*dx` [Hybrid implicit/explicit]
            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p_grads_old = np.sum(cell_sol_p_old[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p_grads_x = p_grads[:, 0, 0] # (num_quads,)
            p_grads_y = p_grads[:, 0, 1] # (num_quads,)
            p_grads_x_old = p_grads_old[:, 0, 0] # (num_quads,)
            p_grads_y_old = p_grads_old[:, 0, 1] # (num_quads,)
            # The coefficient before the "diffusion" term is explicit, but the "diffusion" term itself is implicit
            thetas = vmap_safe_arctan2(p_grads_y_old, p_grads_x_old) # (num_quads,)
            epsilons = vmap_eps_fn(thetas) # (num_quads,)
            epsilons_p = vmap_eps_grad(thetas) # (num_quads,)
            tmp11 = epsilons[:, None] * epsilons_p[:, None] * np.stack((-p_grads_y, p_grads_x)).T # (num_quads, dim)
            tmp12 = epsilons[:, None, None]**2 * p_grads # (num_quads, vec_p, dim)
            tmp1 = tmp11[:, None, :] + tmp12 # (num_quads, vec_p, dim)
            # (num_quads, 1, vec_p, dim) * (num_quads, num_nodes_p, 1, dim)
            # (num_quads, num_nodes_p, vec_p, dim) -> (num_nodes_p, vec_p)
            val1 = np.sum(tmp1[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))

            # Handles the term `-p*(1-p)*(p-1/2+m)*q*dx` [Explicit]
            # (1, num_nodes_p, vec_p) * (num_quads, num_nodes_p, 1) -> (num_quads, num_nodes_p, vec_p) -> (num_quads, vec_p/vec_T)
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            T = np.sum(cell_sol_T[None, :, :] * self.fe_T.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            alpha = self.params['alpha']
            gamma = self.params['gamma']
            T_eq = self.params['T_eq']
            m = alpha / np.pi * np.arctan(gamma*(T_eq - T_old)) # (num_quads,)
            tmp2 = -p_old * (1 - p_old) * (p_old - 0.5 + m) # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val2 = np.sum(tmp2[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)

            # Handles the term `tau*(p - p_old)*q*dx` [Left hand side]
            dt = self.params['dt']
            tau = self.params['tau']
            tmp3 = tau*(p - p_old)/dt # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val3 = np.sum(tmp3[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)

            # Handles the term `-a*p*(1-p)*chi*q*dx` [Explicit]
            a = self.params['a']
            tmp4 = -a * p_old * (1 - p_old) * chi # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val4 = np.sum(tmp4[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)

            # Handles the term `inner(grad(T), grad(S)*dx` [Implicit]
            # (1, num_nodes_T, vec_T, 1) * (num_quads, num_nodes_T, 1, dim) -> (num_quads, num_nodes_T, vec_T, dim)
            T_grads = np.sum(cell_sol_T[None, :, :, None] * cell_shape_grads_T[:, :, None, :], axis=1) # (num_quads, vec_T, dim)   
            # (num_quads, 1, vec_T, dim) * (num_quads, num_nodes_T, 1, dim) -> (num_quads, num_nodes_T, vec_T, dim) -> (num_nodes_T, vec_T)
            val5 = np.sum(T_grads[:, None, :, :] * cell_v_grads_JxW_T, axis=(0, -1))

            # Handles the term `(T - T_old)*S*dx` [Left hand side]
            tmp6 = (T - T_old)/dt # (num_quads,)
            # (num_quads, 1, vec_T) * (num_quads, num_nodes_T, 1) * (num_quads, 1, 1) -> (num_nodes_T, vec_T)
            val6 = np.sum(tmp6[:, None, None] * self.fe_T.shape_vals[:, :, None] * cell_JxW_T[:, None, None], axis=0)

            # Handles the term `-K*(p - p_old)*S*dx` [Left hand side]
            K = self.params['K']
            tmp7 = -K*(p - p_old)/dt # (num_quads,)
            # (num_quads, 1, vec_T) * (num_quads, num_nodes_T, 1) * (num_quads, 1, 1) -> (num_nodes_T, vec_T)
            val7 = np.sum(tmp7[:, None, None] * self.fe_T.shape_vals[:, :, None] * cell_JxW_T[:, None, None], axis=0)

            weak_form = [val1 + val2 + val3 + val4, val5 + val6 + val7] # [(num_nodes_p, vec_p), (num_nodes_T, vec_T)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel

    def set_params(self, params):
        # Override base class method.
        sol_p_old, sol_T_old, noise = params
        self.internal_vars = [sol_p_old[self.fe_p.cells],
                              sol_T_old[self.fe_T.cells],
                              self.fe_p.convert_from_dof_to_quad(sol_p_old)[:, :, 0], 
                              self.fe_T.convert_from_dof_to_quad(sol_T_old)[:, :, 0],
                              np.repeat(noise[:, None], self.fe_p.num_quads, axis=1)]


def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:05d}.vtu')
    vtk_path_T = os.path.join(vtk_dir, f'T_{step:05d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)
    # save_sol(problem.fe_T, sol_list[1], vtk_path_T)


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)

    dt = params['dt']
    t_OFF = params['t_OFF']
    hx = params['hx']
    hy = params['hy']
    nx = params['nx']
    ny = params['ny']
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=nx*hx, domain_y=ny*hy)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    problem = Solidification(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type], additional_info=[params])

    points = problem.fe_p.points
    mask = np.argwhere(((points[:, 0] - nx*hx/2.)**2 + (points[:, 1] - ny*hy/2.)**2) < 20.*hx**2).reshape(-1)

    sol_p = np.zeros((problem.fe_p.num_total_nodes, problem.fe_p.vec))
    sol_p = sol_p.at[mask].set(1.)
    sol_T = np.zeros((problem.fe_T.num_total_nodes, problem.fe_T.vec))
    sol_list = [sol_p, sol_T]
    save_sols(problem, sol_list, 0)

    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
        chi = jax.random.uniform(jax.random.PRNGKey(0), shape=(problem.fe_p.num_cells,)) - 0.5
        problem.set_params([sol_list[0], sol_list[1], chi])
        sol_list = solver(problem, solver_options={'petsc_solver': {}, 'initial_guess': sol_list})   

        if (i + 1) % 10 == 0:
            save_sols(problem, sol_list, i + 1)


if __name__ == '__main__':
    simulation()
