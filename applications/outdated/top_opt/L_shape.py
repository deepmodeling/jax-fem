'''
Generally following
Yang, Dixiong, et al. "Stress-constrained topology optimization based on maximum stress measures." Computers & Structures 198 (2018): 23-39.


'''
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import meshio
import time

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh

from applications.fem.top_opt.fem_model import Elasticity
from applications.fem.top_opt.mma import optimize

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
 
def topology_optimization():
    p_name = 'L_shape'
    problem_name = p_name + '_w_cstr'
    root_path = os.path.join(os.path.dirname(__file__), 'data') 

    files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
    for f in files:
        os.remove(f)

    mesh_file = os.path.join(root_path, f"abaqus/L_1600.inp")
    meshio_mesh = meshio.read(mesh_file)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'])

    # vtk_dir = os.path.join(root_path, f'vtk/{problem_name}')
    # os.makedirs(vtk_dir, exist_ok=True)
    # meshio_mesh.write(os.path.join(vtk_dir, 'L_6400.vtk'))

    def fixed_location(point):
        return np.isclose(point[1], 1., atol=1e-5)

    def load_location(point):
        return np.logical_and(np.isclose(point[0], 1., atol=1e-5), np.isclose(point[1], 0.4, atol=0.06 + 1e-5))

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., -1e6])

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(jax_mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=dirichlet_bc_info, 
        neumann_bc_info=neumann_bc_info, additional_info=(p_name,))
    fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)

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
    max_vm_stresses = []
    def output_sol(params, obj_val):
        print(f"\nOutput solution - need to solve the forward problem again...")
        sol = fwd_pred(params)
        volume_avg_vm_stress = problem.compute_von_mises_stress(sol)
        vtu_path = os.path.join(root_path, f'vtk/{problem_name}/sol_{output_sol.counter:03d}.vtu')
        save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, 
            cell_infos=[('theta', problem.full_params[:, 0]), ('vm_stress', volume_avg_vm_stress)])
        print(f"compliance = {obj_val}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        outputs.append(obj_val)
        max_vm_stresses.append(onp.max(volume_avg_vm_stress))
        output_sol.counter += 1

    output_sol.counter = 0
        
    vf = 0.5
    max_allowed_stress = 3.5e6

    def objectiveHandle(rho):
        J, dJ = jax.value_and_grad(J_total)(rho)
        output_sol(rho, J)
        return J, dJ

    def computeConstraints(rho, epoch):
        def computeGlobalVolumeConstraintFull(rho):
            g1 = np.mean(rho)/vf - 1.
            sol = fwd_pred(rho)
            volume_avg_vm_stress = problem.compute_von_mises_stress(sol)
            max_vm_stress = np.max(volume_avg_vm_stress)
            p = 12
            vm_stress_PN = np.sum((volume_avg_vm_stress/max_allowed_stress)**p)**(1./p)
            qn = 0.5
            cp = qn*computeConstraints.max_vm_stress/(max_allowed_stress*computeConstraints.vm_stress_PN) + (1 - qn)*computeConstraints.cp
            g2 = cp*vm_stress_PN - 1.
            return np.array([g1, g2]), (cp, max_vm_stress, vm_stress_PN)
            # If no constraint, set the following
            # return np.array([g1, 0]), (cp, max_vm_stress, vm_stress_PN)

        computeGlobalVolumeConstraint = lambda *args: computeGlobalVolumeConstraintFull(*args)[0]
        c, (computeConstraints.cp, computeConstraints.max_vm_stress, computeConstraints.vm_stress_PN) = computeGlobalVolumeConstraintFull(rho)
        print(f"### computeConstraints.cp = {computeConstraints.cp}, computeConstraints.max_vm_stress = {computeConstraints.max_vm_stress/1e6}, \
            PN approx stress = {computeConstraints.vm_stress_PN*max_allowed_stress/1e6}, max_allowed_stress = {max_allowed_stress/1e6}")
        gradc = jax.jacrev(computeGlobalVolumeConstraint)(rho)
        return c, gradc

    computeConstraints.cp = 1.
    computeConstraints.max_vm_stress = max_allowed_stress
    computeConstraints.vm_stress_PN = 1.

    optimizationParams = {'maxIters':201, 'minIters':201, 'relTol':0.05}
    rho_ini = vf*np.ones((len(problem.flex_inds), 1))
    optimize(problem, rho_ini, optimizationParams, objectiveHandle, computeConstraints, numConstraints=2, movelimit=0.1)
    onp.save(os.path.join(root_path, f"numpy/{problem_name}_outputs.npy"), onp.array(outputs))
    onp.save(os.path.join(root_path, f"numpy/{problem_name}_max_vm_stresses.npy"), onp.array(max_vm_stresses))
    print(f"Compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")


if __name__=="__main__":
    topology_optimization()
