# Import some useful modules.
import jax
import jax.numpy as np
import os
import glob
import meshio
import numpy as onp
import os

# Import JAX-FEM specific modules.
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.solver import solver
from jax_fem.problem import Problem
from jax_fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define some useful directory paths.
crt_file_path = os.path.dirname(__file__)
input_dir = os.path.join(crt_file_path, 'input')
output_dir = os.path.join(crt_file_path, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
files = glob.glob(os.path.join(vtk_dir, f'*'))
for f in files:
    os.remove(f)

# Unit are in SI unit system
T0 = 300.
Cp = 500.
rho = 8440.
rb = 1e-3
eta = 0.4
h = 50.
P = 500.

class Thermal(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        def fn(u_grad, T_old, dt, active_cell_marker):
            """If active_cell_marker is False, this cell (or quad points in it) should NOT participate in FEM computations
            """
            k = 15.
            val = k*u_grad
            return np.where(active_cell_marker, val, np.zeros_like(val))
        return fn

    def get_mass_map(self):
        def T_map(T, x, T_old, dt, active_cell_marker):
            """If active_cell_marker is False, this cell (or quad points in it) should NOT participate in FEM computations
            """
            val = rho*Cp*(T - T_old)/dt
            return np.where(active_cell_marker, val, np.zeros_like(val))
        return T_map

    def get_surface_maps(self):
        def thermal_flux(u, point, old_T, laser_center, switch, active_face_markers):
            """If active_face_marker is False, this face (or quad points in it) should NOT participate in FEM computations
            """
            active_face_external, active_face_top = active_face_markers
            # q is the heat flux INTO the domain
            d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
            q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2) * switch
            q_laser = np.where(active_face_top, q_laser, 0.)
            q_conv = h*(T0 - old_T[0])
            q_conv = np.where(active_face_external, q_conv, 0.)
            q = q_laser + q_conv
            return -np.array([q])
        return [thermal_flux]

    def set_params(self, params):
        sol_T_old, dt, laser_center, switch, quiet_point_inds_set, active_cell_truth_tab, active_face_truth_tab = params

        sol_T_old_surface = self.fe.convert_from_dof_to_face_quad(sol_T_old, self.boundary_inds_list[0])
        sol_T_old_body = self.fe.convert_from_dof_to_quad(sol_T_old)
        dt_quad = dt * np.ones((self.fe.num_cells, self.fe.num_quads))

        # (num_selected_faces, num_face_quads, dim)
        laser_center_quad = laser_center[None, None, :] * np.ones((len(self.boundary_inds_list[0]), self.fe.num_face_quads))[:, :, None]
        # (num_selected_faces, num_face_quads)
        switch_quad = switch * np.ones((len(self.boundary_inds_list[0]), self.fe.num_face_quads))

        active_cell_truth_tab_quad = np.repeat(active_cell_truth_tab[:, None], self.fe.num_quads, axis=1)
        active_face_truth_tab_quad = np.repeat(active_face_truth_tab[:, None], self.fe.num_face_quads, axis=1)

        self.internal_vars = [sol_T_old_body, dt_quad, active_cell_truth_tab_quad]
        self.internal_vars_surfaces = [[sol_T_old_surface, laser_center_quad, switch_quad, active_face_truth_tab_quad]]

        self.fe.dirichlet_bc_info[0][0] = get_dirichlet_location_fn(quiet_point_inds_set)
        self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)


def hash_map_for_faces(active_cell_truth_tab, cells_face, hash_map, inner_faces, all_faces, cell_inds):
    """Use a hash table to store faces

    Parameters
    ----------
    active_cell_truth_tab : onp.ndarray
        (num_cells,)
        active_cell_truth_tab[i] is True if the ith cell is active; it is False if the cell is quiet
    cells_face : onp.ndarray
        (num_cells, num_faces, num_face_vertices)
    hash_map : Dictionary
        e.g., {(node1, node2, node3, node4): (cell_id, face_id),...} 
        Note that a sorted (node1, node2, node3, node4) defines one face, but the face can be shared from two cells, i.e., (cell_id, face_id)
    inner_faces : List[Tuple]
        e.g., [(cell_id, face_id),...]
        faces shared by two cells
    all_faces : List[Tuple]
        e.g., [(cell_id, face_id),...]
        all faces
    cell_inds : onp.ndarray
        number of cells considered

    Returns
    -------
    external_faces : List[Tuple]
        e.g., [(cell_id, face_id),...]
        faces NOT shared by two cells, i.e., external faces
    """
    for i, cell_id in enumerate(cell_inds):
        if active_cell_truth_tab[cell_id]:
            for face_id in range(len(cells_face[cell_id])):
                key = tuple(cells_face[cell_id, face_id].tolist())
                if key in hash_map.keys():
                    inner_faces.append(hash_map[key])
                    inner_faces.append((cell_id, face_id))
                else:
                    hash_map[key] = (cell_id, face_id)
                all_faces.append((cell_id, face_id))

    external_faces = onp.array(list((set(all_faces) - set(inner_faces))))
    return external_faces


def initialize_hash_map(problem, active_cell_truth_tab):
    """Use a hash table to store faces

    Parameters
    ----------
    problem : Problem object
    active_cell_truth_tab : onp.ndarray
    """
    print(f"Initializing hash map for external faces...")
    face_inds = problem.fe.face_inds
    cells_face = problem.fe.cells[:, face_inds] # (num_cells, num_faces, num_face_vertices)
    cells_face = onp.sort(cells_face)
    hash_map = {}
    inner_faces = []
    all_faces = []
    cell_inds = onp.arange(len(cells_face))
    external_faces = hash_map_for_faces(active_cell_truth_tab, cells_face, hash_map, inner_faces, all_faces, cell_inds)
    return external_faces, cells_face, hash_map, inner_faces, all_faces


def update_hash_map(active_cell_truth_tab_old, active_cell_truth_tab_new, cells_face, hash_map, inner_faces, all_faces):
    """Update hash_map, inner_faces and external_faces 

    """    
    print(f"Updating hash map for external faces...")
    assert onp.sum(active_cell_truth_tab_new) >= onp.sum(active_cell_truth_tab_old), "Number of new born cells must be non-negative!"
    new_born_cell_inds = onp.argwhere(active_cell_truth_tab_old != active_cell_truth_tab_new).reshape(-1)
    external_faces = hash_map_for_faces(active_cell_truth_tab_new, cells_face, hash_map, inner_faces, all_faces, new_born_cell_inds)
    return external_faces


def get_active_face_truth_tab(problem, external_faces, laser_center_z):
    """Get truth tables for active faces

    Parameters
    ----------   
    external_faces : onp.ndarray
        (num_external_faces, 2) 

    Returns
    -------
    active_face_truth_tab: onp.ndarray
        (num_selected_faces, 2)
    """
    cell_points = onp.take(problem.fe.points, problem.fe.cells, axis=0)  # (num_cells, num_nodes, dim)
    cell_face_points = onp.take(cell_points, problem.fe.face_inds, axis=1)  # (num_cells, num_faces, num_face_vertices, dim)
    all_boundary_faces = problem.boundary_inds_list[0]

    all_boundary_faces_dict = {}
    for index, face in enumerate(all_boundary_faces):
        all_boundary_faces_dict[tuple(face.tolist())] = index

    active_face_truth_tab_external = onp.zeros(len(all_boundary_faces), dtype=bool)
    active_face_truth_tab_top = onp.zeros(len(all_boundary_faces), dtype=bool)
    for i, face in enumerate(external_faces):
        key = tuple(face.tolist())
        if key in all_boundary_faces_dict.keys():
            index = all_boundary_faces_dict[key]
            active_face_truth_tab_external[index] = True

            face_points = cell_face_points[face[0], face[1]]
            if np.all(np.isclose(face_points[:, 2], laser_center_z, atol=1e-5)):
                active_face_truth_tab_top[index] = True
        else:
            raise ValueError(f"An external face should always be in the set of all faces")

    active_face_truth_tab = onp.stack((active_face_truth_tab_external, active_face_truth_tab_top)).T

    return active_face_truth_tab


def get_quiet_point_inds_set(problem, active_cell_truth_tab):
    """Use a hash table to store faces

    Parameters
    ----------
    problem : Problem object
    active_cell_truth_tab : onp.ndarray

    Returns
    -------
    quiet_point_inds_set : onp.ndarray
        (num_active_points,)
        a collection of quiet point indices; Dirichlet B.C. will be applied to these points so that they are quiet
    """
    active_point_inds = set(problem.fe.cells[active_cell_truth_tab].reshape(-1))
    all_points_inds = set(onp.arange(len(problem.fe.points)))
    quiet_point_inds_set = onp.array(list(all_points_inds - active_point_inds), dtype=onp.int32)
    return quiet_point_inds_set


def get_dirichlet_location_fn(quiet_point_inds_set):
    def dirichlet_location_fn(point, ind):
        return np.isin(ind, quiet_point_inds_set)
    return dirichlet_location_fn


def quiet_element_simulation():
    """
    A cell is either active or quiet
    A face is either active or quiet
    A point is either active or quiet
    """
    base_plate_height = 20.*1e-3 # Use Paraview to check base_plate_height 
    path_resolution = 0.125*1e-3 # # Use Paraview to check element x size = 0.5*1e-3

    vec = 1
    dim = 3
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)

    abaqus_file = os.path.join(input_dir, f'abaqus/thinwall.inp')
    meshio_mesh = meshio.read(abaqus_file)
    mesh = Mesh(meshio_mesh.points/1e3, meshio_mesh.cells_dict[cell_type]) # Original Abaqus file is in [mm]

    vtk_file = os.path.join(vtk_dir, 'thinwall.vtu')
    meshio_mesh.write(vtk_file)

    toolpath = onp.loadtxt(os.path.join(input_dir, f'toolpath/thinwall_toolpath.crs'))
    toolpath[:, 1:4] = toolpath[:, 1:4]/1e3 # Original toolpath file is in [mm]

    def all_faces_fn(point):
        return True

    location_fns = [all_faces_fn]

    def dirichlet_val_fn(point):
        return T0

    dirichlet_bc_info = [[lambda x: False], [0], [dirichlet_val_fn]]
    problem = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

    active_cell_truth_tab = onp.zeros(len(problem.fe.cells), dtype=bool)
    centroids = onp.mean(problem.fe.points[problem.fe.cells], axis=1)
    active_cell_truth_tab[centroids[:, 2] <= base_plate_height] = True
    external_faces, cells_face, hash_map, inner_faces, all_faces = initialize_hash_map(problem, active_cell_truth_tab)

    sol_list = [T0*np.ones((len(problem.fe.points), vec))]
    active_cell_truth_tab_old = active_cell_truth_tab

    for i in range(2, toolpath.shape[0]):
        switch = toolpath[i, 4]
        if switch:
            direction = toolpath[i, 1:4] - toolpath[i - 1 , 1:4]
            d = np.linalg.norm(direction)
            num_dt_laser_on = round(d/path_resolution)
            t = onp.linspace(toolpath[i - 1, 0], toolpath[i, 0], num_dt_laser_on + 1)
            X = onp.interp(t, [toolpath[i - 1, 0], toolpath[i, 0]], [toolpath[i - 1, 1], toolpath[i, 1]])
            Y = onp.interp(t, [toolpath[i - 1, 0], toolpath[i, 0]], [toolpath[i - 1, 2], toolpath[i, 2]])

            for j in range(num_dt_laser_on):
                dt = t[j + 1] - t[j]
                print(f"\n############################################################")
                print(f"Laser on: i = {i} in {toolpath.shape[0]} , j = {j} in {num_dt_laser_on}")
                laser_center = np.array([X[j], Y[j], toolpath[i, 3] + base_plate_height])
                print(f"laser center = {laser_center}, dt = {dt}")
                flag_1 = centroids[:, 2] < laser_center[2]
                flag_2 = (centroids[:, 0] - laser_center[0])**2 + (centroids[:, 1] - laser_center[1])**2 <= rb**2
                active_cell_truth_tab = active_cell_truth_tab | (flag_1 & flag_2)

                if onp.all(active_cell_truth_tab == active_cell_truth_tab_old):
                    print(f"No element born")
                else:
                    print(f"New elements born: number = {onp.sum(active_cell_truth_tab) - onp.sum(active_cell_truth_tab_old)}")
                    external_faces = update_hash_map(active_cell_truth_tab_old, active_cell_truth_tab, cells_face, hash_map, inner_faces, all_faces)
                    active_face_truth_tab = get_active_face_truth_tab(problem, external_faces, laser_center[2])
                    quiet_point_inds_set = get_quiet_point_inds_set(problem, active_cell_truth_tab)

                problem.set_params([sol_list[0], dt, laser_center, switch, quiet_point_inds_set, active_cell_truth_tab, active_face_truth_tab])
                sol_list = solver(problem)
       
                if j % 10 == 0:
                    vtk_path = os.path.join(vtk_dir, f"u_{i:05d}_{j:05d}.vtu")
                    save_sol(problem.fe, sol_list[0], vtk_path, cell_infos=[('active', onp.array(active_cell_truth_tab, dtype=onp.int32))])

                active_cell_truth_tab_old = active_cell_truth_tab
        else:
            num_dt_laser_off = 10
            t = onp.linspace(toolpath[i - 1, 0], toolpath[i, 0], num_dt_laser_off + 1)
            for j in range(num_dt_laser_off):
                dt = t[j + 1] - t[j]
                print(f"\n############################################################")
                print(f"Laser off: i = {i} in {toolpath.shape[0]} , j = {j} in {num_dt_laser_off}")
                problem.set_params([sol_list[0], dt, laser_center, switch, quiet_point_inds_set, active_cell_truth_tab, active_face_truth_tab])
                sol_list = solver(problem, solver_options={'petsc_solver': {}})
                vtk_path = os.path.join(vtk_dir, f"u_{i:05d}_{j:05d}.vtu")
                save_sol(problem.fe, sol_list[0], vtk_path, cell_infos=[('active', onp.array(active_cell_truth_tab, dtype=onp.int32))])


if __name__=="__main__":
    quiet_element_simulation()
