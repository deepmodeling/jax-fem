import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import meshio
import time

from jax_am.fem.generate_mesh import Mesh, box_mesh
from jax_am.fem.solver import ad_wrapper
from jax_am.fem.utils import save_sol

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
 
def topology_optimization():
    problem_name = 'plate'
    root_path = os.path.join(os.path.dirname(__file__), 'data') 

    files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    meshio_mesh = box_mesh(50, 30, 1, 50., 30., 1., root_path)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load_location(point):
        return np.logical_and(np.isclose(point[0], 50., atol=1e-5), np.isclose(point[1], 15., atol=1.5))

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., -10., 0.])

    dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(jax_mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info, additional_info=(problem_name,))
    fwd_pred = ad_wrapper(problem, linear=False)

    def J_fn(dofs, params):
        """J(u, p)
        """
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        compliance = problem.compute_compliance(neumann_val, sol)
        return compliance

    def J_total(params):
        """J(u(p), p)
        """     
        sol = fwd_pred(params)
        dofs = sol.reshape(-1)
        obj_val = J_fn(dofs, params)
        return obj_val

    outputs = []
    def output_sol(params, obj_val):
        print(f"\nOutput solution - need to solve the forward problem again...")
        sol = fwd_pred(params)
        vtu_path = os.path.join(root_path, f'vtk/{problem_name}/sol_{output_sol.counter:03d}.vtu')
        save_sol(problem, sol, vtu_path, cell_infos=[('theta', problem.full_params[:, 0])])
        print(f"compliance = {obj_val}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        outputs.append(obj_val)
        output_sol.counter += 1

    output_sol.counter = 0
        
    vf = 0.5

    def objectiveHandle(rho):
        J, dJ = jax.value_and_grad(J_total)(rho)
        output_sol(rho, J)
        return J, dJ

    def computeConstraints(rho, epoch):
        def computeGlobalVolumeConstraint(rho):
            g = np.mean(rho)/vf - 1.
            return g
        c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
        c, gradc = c.reshape((1,)), gradc[None, ...]
        return c, gradc

    optimizationParams = {'maxIters':7, 'minIters':7, 'relTol':0.05}
    rho_ini = vf*np.ones((len(problem.flex_inds), 1))
    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=1)
    onp.save(os.path.join(root_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    print(f"Compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")


if __name__=="__main__":
    topology_optimization()
