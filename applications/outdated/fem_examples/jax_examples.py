import numpy as onp
import jax
import jax.numpy as np
import meshio
import time
import os

from jax_fem.models import LinearElasticity, HyperElasticity, Plasticity
from jax_fem.solver import solver
from jax_fem.utils import modify_vtu_file, save_sol
from jax_fem.generate_mesh import Mesh, cylinder_mesh_gmsh

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def linear_elasticity_dogbone(disp, index):
    abaqus_root = os.path.join(data_dir, f'abaqus')
    abaqus_files = ['DogBone_mesh6_disp10.inp',
                    'DogBone_mesh2_disp10.inp',
                    'DogBone_mesh1_disp10.inp',
                    'DogBone_mesh05_disp10.inp',
                    'DogBone_mesh03_disp10.inp',
                    'DogBone_mesh025_disp10.inp',
                    'DogBone_mesh02_disp10.inp']

    mesh_file = os.path.join(abaqus_root, abaqus_files[index])
    meshio_mesh = meshio.read(mesh_file)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    min_x = np.min(jax_mesh.points[:, 0])
    max_x = np.max(jax_mesh.points[:, 0])
    print(f"max_x = {max_x}, min_x = {min_x}")
 
    def min_x_loc(point):
        return np.isclose(point[0], min_x, atol=1e-5)

    def max_x_loc(point):
        return np.isclose(point[0], max_x, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        # return 0.1*(max_x - min_x)
        return disp

    dirichlet_bc_info = [[min_x_loc, min_x_loc, min_x_loc, max_x_loc, max_x_loc, max_x_loc], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]

    start_time = time.time()
    problem_name = "dogbone"
    problem = LinearElasticity(jax_mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, linear=True, precond=True)
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Wall time elapsed {solve_time}")
    vtu_path = os.path.join(data_dir, f'vtk/{problem_name}/u_{index}.vtu')
    save_sol(problem, sol, vtu_path)
    num_total_dofs = problem.num_total_nodes*problem.vec
    return solve_time, num_total_dofs


def linear_elasticity_cylinder(disps):
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], 10., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]
 
    problem = LinearElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    tractions = []
    for i, disp in enumerate(disps):
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        sol = solver(problem, linear=True)
        traction = problem.compute_traction(top, sol)
        tractions.append(traction[2])
    tractions = np.array(tractions)
    np.save(os.path.join(data_dir, "numpy/linear_elasticity/jax_fem/forces.npy"), tractions)


def hyperelasticity_cylinder(disps):
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], 10., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]             
    problem = HyperElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    sol = np.zeros((problem.num_total_nodes, problem.vec)) 
    tractions = []
    for i, disp in enumerate(disps):
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        sol = solver(problem)
        traction = problem.compute_traction(top, sol)
        tractions.append(traction[2])
    tractions = np.array(tractions)
    np.save(os.path.join(data_dir, f'numpy/hyperelasticity/jax_fem/forces.npy'), tractions)


def plasticity():
    problem_name = "plasticity"
    fenicsx_vtu_path_raw = os.path.join(data_dir, 'vtk', problem_name, 'fenicsx/sol_p0_000000.vtu')
    fenicsx_vtu_path = os.path.join(data_dir, 'vtk', problem_name, 'fenicsx/sol.vtu')
    modify_vtu_file(fenicsx_vtu_path_raw, fenicsx_vtu_path)
    fenicsx_vtu = meshio.read(fenicsx_vtu_path)
    cells = fenicsx_vtu.cells_dict['VTK_LAGRANGE_HEXAHEDRON'] # 'hexahedron'
    points = fenicsx_vtu.points
    mesh = Mesh(points, cells)
    H = 10.

    def top(point):
        return np.isclose(point[2], H, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def dirichlet_val_bottom(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    disps_path = os.path.join(plasticity_path, 'numpy', problem_name, 'fenicsx/disps.npy')
    disps = np.load(disps_path)

    location_fns = [bottom, bottom, bottom, top, top, top]
    value_fns = [dirichlet_val_bottom, dirichlet_val_bottom, dirichlet_val_bottom, 
                 dirichlet_val_bottom, dirichlet_val_bottom, get_dirichlet_top(disps[0])]
    vecs = [0, 1, 2, 0, 1, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    avg_stresses = []

    for i, disp in enumerate(disps):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")
        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
        problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        sol = solver(problem)
        problem.update_stress_strain(sol)
        avg_stress = problem.compute_avg_stress()
        print(avg_stress)
        avg_stresses.append(avg_stress)

    avg_stresses = np.array(avg_stresses)

    jax_vtu_path = os.path.join(plasticity_path, 'vtk', problem_name, 'jax_fem/sol.vtu')

    save_sol(problem, sol, jax_vtu_path)

    jax_fem_avg_stresses_path = os.path.join(plasticity_path, 'numpy', problem_name, 'jax_fem/forces.npy')
    jax_fem_avg_stresses = avg_stresses[:, 2, 2]
    np.save(jax_fem_avg_stresses_path, jax_fem_avg_stresses)


def performance_test():
    solve_time = []
    num_dofs = []
    # for i in range(6, -1, -1):
    for i in range(3, -1, -1):
        wall_time, n_dofs = linear_elasticity_dogbone(10., i)
        solve_time.append(wall_time)
        num_dofs.append(n_dofs)
    solve_time = np.array(solve_time)
    num_dofs = np.array(num_dofs)
    platform = jax.lib.xla_bridge.get_backend().platform
    onp.savetxt(os.path.join(data_dir, f'/txt/jax_fem_{platform}_time.txt'), solve_time[::-1], fmt='%.3f')
    onp.savetxt(os.path.join(data_dir, f'txt/jax_fem_{platform}_dof.txt'), num_dofs[::-1], fmt='%.3f')


def generate_fem_examples():
    plasticity()
    linear_elasticity_disps = np.load(os.path.join(data_dir, f'numpy/linear_elasticity/fenicsx/disps.npy'))
    linear_elasticity_cylinder(linear_elasticity_disps)
    hyperelasticity_disps = np.load(os.path.join(data_dir, f'numpy/hyperelasticity/fenicsx/disps.npy'))
    hyperelasticity_cylinder(hyperelasticity_disps)


def exp():
    linear_elasticity_dogbone(10., 6)

if __name__ == "__main__":
    # generate_fem_examples()
    performance_test()
    # exp()
