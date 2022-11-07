import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import scipy.optimize as opt
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import meshio
import time

from jax_am.fem.generate_mesh import Mesh, box_mesh
from jax_am.fem.solver import solver, adjoint_method
from jax_am.fem.utils import save_sol

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_boundary_info(jax_mesh):
    left_cx = -21.
    left_cy = 20.
    left_r = 5.

    right_up_cx = 10.
    right_up_cy = 40.
    right_up_r = 5.

    right_down_cx = 10.
    right_down_cy = 0.
    right_down_r = 5.

    def load_location(point):
        return np.isclose(np.sqrt((point[0] - left_cx)**2 + (point[1] - left_cy)**2) - left_r, 0., atol=1e-2)

    def fixed_location(point):
        up = np.isclose(np.sqrt((point[0] - right_up_cx)**2 + (point[1] - right_up_cy)**2) - right_up_r, 0., atol=1e-2)
        down = np.isclose(np.sqrt((point[0] - right_down_cx)**2 + (point[1] - right_down_cy)**2) - right_down_r, 0., atol=1e-2)
        return np.logical_or(up, down)

    def flex_location(point):
        lower_bound_point = np.array([-15., 10., 5.])
        upper_bound_point = np.array([20., 30., 25.])
        lower_flag = np.all(point > lower_bound_point)
        upper_flag = np.all(point < upper_bound_point)
        return np.logical_and(lower_flag, upper_flag)

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., 1., 0.])

    return load_location, fixed_location, flex_location, dirichlet_val, neumann_val


def human_design():
    """Take a human design, run the forward problem.
    """
    linear_flag = True
    problem_name = 'human_design'
    root_path = os.path.join(os.path.dirname(__file__), 'data') 

    files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    mesh_file = os.path.join(root_path, f"abaqus/designed_fine.inp")
    meshio_mesh = meshio.read(mesh_file)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    load_location, fixed_location, flex_location, dirichlet_val, neumann_val = get_boundary_info(jax_mesh)

    dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(jax_mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info, additional_info=(linear_flag,))

    sol = solver(problem, linear=linear_flag)
    compliance = problem.compute_compliance(neumann_val, sol)
    print(f"Human design compliance = {compliance}")

    volume_avg_vm_stress = problem.compute_von_mises_stress(sol)

    vtu_path = os.path.join(root_path, f'vtk/{problem_name}/sol.vtu')
    save_sol(problem, sol, vtu_path, cell_infos=[('vm_stress', volume_avg_vm_stress)])

    flex_inds = np.argwhere(jax.vmap(flex_location)(problem.cell_centroids)).reshape(-1)
    V = np.sum(problem.JxW)
    V_design = np.sum(problem.JxW[flex_inds])
    print(f"Total V = {V}, design area V = {V_design}")

    return V_design, compliance


def computer_design():
    """Inverse design with topology optimization.
    """
    linear_flag = True
    problem_name = 'computer_design'
    root_path = os.path.join(os.path.dirname(__file__), 'data') 
    files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    mesh_file = os.path.join(root_path, f"abaqus/undesigned_fine.inp")
    meshio_mesh = meshio.read(mesh_file)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    load_location, fixed_location, flex_location, dirichlet_val, neumann_val = get_boundary_info(jax_mesh)

    dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(problem_name, jax_mesh, linear_flag, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

    problem.flex_inds = np.argwhere(jax.vmap(flex_location)(problem.cell_centroids)).reshape(-1)

    def J_fn(dofs, params):
        """J(u, p)
        """
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        compliance = problem.compute_compliance(neumann_val, sol)
        return compliance

    outputs = []
    def output_sol(params, dofs, obj_val):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        vtu_path = os.path.join(root_path, f'vtk/{problem_name}/sol_{fn.counter:03d}.vtu')
        design_box = np.ones(problem.num_cells).at[problem.flex_inds].set(0.)
        save_sol(problem, sol, vtu_path, cell_infos=[('theta', problem.full_params), ('design', design_box)])
        print(f"compliance = {obj_val}, compliance_human = {compliance_human}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        outputs.append(obj_val)

    fn, fn_grad = adjoint_method(problem, J_fn, output_sol, linear=linear_flag)

    V_design, compliance_human = human_design()
    V_design_box = np.sum(problem.JxW[problem.flex_inds])
    vf = V_design/V_design_box
    print(f"Computer design should use the same vf = {vf} with human design.")

    def objectiveHandle(rho):
        J = fn(rho)
        dJ = fn_grad(rho)
        return J, dJ

    def computeConstraints(rho, epoch):
        def computeGlobalVolumeConstraint(rho):
            g = np.mean(rho)/vf - 1.
            return g
        c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho);
        c, gradc = c.reshape((1, 1)), gradc.reshape((1, -1))
        return c, gradc

    optimizationParams = {'maxIters':30, 'minIters':30, 'relTol':0.05}
    rho_ini = vf*np.ones(len(problem.flex_inds))
    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=1)
    onp.save(os.path.join(root_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    print(f"Compliance = {fn(np.ones(len(problem.flex_inds)))} for full material")


if __name__ == "__main__":
    human_design()
    # computer_design()
