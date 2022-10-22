"""The example is not used and should be deprecated.
"""
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

from jax_am.fem.generate_mesh import box_mesh
from jax_am.fem.jax_fem import Mesh
from jax_am.fem.solver import solver, adjoint_method
from jax_am.fem.utils import save_sol

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def define_circle(p1, p2, p3):
    """Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).

    Copied from https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = onp.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)


def get_boundary_locations(jax_mesh):
    """Stupid STEP file doesn't define circles with integer value locations...
    """
    left_id1 = 1874
    left_id2 = 1811
    left_id3 = 1854
    (left_cx, left_cy), left_r = define_circle(jax_mesh.points[left_id1], 
                                               jax_mesh.points[left_id2], 
                                               jax_mesh.points[left_id3])
    right_up_id1 = 907
    right_up_id2 = 908
    right_up_id3 = 914
    (right_up_cx, right_up_cy), right_up_r = define_circle(jax_mesh.points[right_up_id1], 
                                                           jax_mesh.points[right_up_id2], 
                                                           jax_mesh.points[right_up_id3])
    right_down_id1 = 1076
    right_down_id2 = 29
    right_down_id3 = 1059
    (right_down_cx, right_down_cy), right_down_r = define_circle(jax_mesh.points[right_down_id1], 
                                                                 jax_mesh.points[right_down_id2], 
                                                                 jax_mesh.points[right_down_id3])

    print(f"Manually find out the circle positions:")
    print(f"Left circle: {(left_cx, left_cy), left_r}")
    print(f"Right up circle: {(right_up_cx, right_up_cy), right_up_r}")
    print(f"Right down circle: {(right_down_cx, right_down_cy), right_down_r}")

    def load_location(point):
        return np.isclose(np.sqrt((point[0] - left_cx)**2 + (point[1] - left_cy)**2) - left_r, 0., atol=1e-5)

    def fixed_location(point):
        up = np.isclose(np.sqrt((point[0] - right_up_cx)**2 + (point[1] - right_up_cy)**2) - right_up_r, 0., atol=1e-5)
        down = np.isclose(np.sqrt((point[0] - right_down_cx)**2 + (point[1] - right_down_cy)**2) - right_down_r, 0., atol=1e-5)
        return np.logical_or(up, down)

    def handle_location(point):
        tol = 2.
        left = np.isclose(np.sqrt((point[0] - left_cx)**2 + (point[1] - left_cy)**2) - left_r, 0., atol=tol)
        right_up = np.isclose(np.sqrt((point[0] - right_up_cx)**2 + (point[1] - right_up_cy)**2) - right_up_r, 0., atol=tol)
        right_down = np.isclose(np.sqrt((point[0] - right_down_cx)**2 + (point[1] - right_down_cy)**2) - right_down_r, 0., atol=tol)
        return np.logical_or(np.logical_or(left, right_up), right_down)

    return load_location, fixed_location, handle_location


def topology_optimization():
    linear_flag = True
    problem_name = 'bracket'
    root_path = f'applications/fem/top_opt/data'
    files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    mesh_file = os.path.join(root_path, f"abaqus/bracket.inp")
    meshio_mesh = meshio.read(mesh_file)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., 1., 0.])

    load_location, fixed_location, handle_location = get_boundary_locations(jax_mesh)

    dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(problem_name, jax_mesh, linear_flag, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

    inds_all = set(onp.arange(problem.num_cells))
    inds_handle = set(onp.argwhere(jax.vmap(handle_location)(problem.cell_centroids)).reshape(-1))
    problem.flex_inds = np.array(list(inds_all - inds_handle))

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
        save_sol(problem, sol, vtu_path, cell_infos=[('theta', problem.full_params)])
        print(f"compliance = {obj_val}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        outputs.append(obj_val)

    fn, fn_grad = adjoint_method(problem, J_fn, output_sol, linear=linear_flag)

    vf = 0.5

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

    optimizationParams = {'maxIters':50, 'minIters':50, 'relTol':0.05}
    rho_ini = vf*np.ones(len(problem.flex_inds))
    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=1)
    onp.save(os.path.join(root_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    print(f"Compliance = {fn(np.ones(len(problem.flex_inds)))} for full material")


if __name__ == "__main__":
    topology_optimization()
