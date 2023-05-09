import numpy as onp
import jax
import jax.numpy as np
import os
import os.path as osp
import glob
import meshio
import time
import json
from pathlib import Path
from timeit import default_timer
import pickle

from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

from applications.fem.thermal.models import Thermal, initialize_hash_map, update_hash_map, get_active_mesh

# from memory_profiler import profile
import tracemalloc
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# data_dir = os.path.join('data') 

crt_dir = os.path.dirname(__file__)
data_dir = os.path.join(crt_dir, 'data') 


#@profile
def ded_cad_model(parameter_json_file, problem_name):
    #### generate activation time list as well
    
    am_para = json.load(open(parameter_json_file,"rb")) 
    
    T0 = float(am_para["T0"])
    Cp = float(am_para["Cp"])
    L = float(am_para["L"])
    rho = float(am_para["rho"])
    h = float(am_para["h"])
    rb = float(am_para["rb"])
    eta = float(am_para["eta"])
    P = float(am_para["P"])
    base_plate_height = 0 

    

    vec = 1
    dim = 3
    ele_type = 'HEX8'
    
    femfile_dir = osp.join(data_dir,"meshes",problem_name)
    files = glob.glob(osp.join(femfile_dir, f'*'))
    
    t1 = default_timer()
    # for ffi,f in enumerate(files[:1]):
    for ffi,f in enumerate(files):

        print(f"\n\n\nffi = {ffi}, f = {f} len(files[:1]) = {len(files[:1])}")

        fem_file = pickle.load( open(f, "rb" ) )   
        path_dx = fem_file["dx"]
        toolpath = fem_file["toolpath"]
        fem_points = fem_file["extend_vertices"]
        #fem_points = fem_file["vertices"]
        hexahedron = fem_file["hexahedra"]
        dx = fem_file["dx"]
        dt = fem_file["dt"]
        #dt = 0.2
        sampled_deposits = fem_file["sampled_deposits"]

        vtk_dir = os.path.join(data_dir,"vtk",problem_name,Path(f).stem)
        os.makedirs(vtk_dir, exist_ok=True)
        vtk_files = glob.glob(os.path.join(vtk_dir, f'*'))
        for ff in vtk_files:
            os.remove(ff)
        full_mesh = Mesh(fem_points, hexahedron)
        active_cell_truth_tab = onp.zeros(len(full_mesh.cells), dtype=bool)
        centroids = onp.mean(full_mesh.points[full_mesh.cells], axis=1)
        active_cell_truth_tab[centroids[:, 2] <= base_plate_height + dx/4 ] = True
        active_mesh, points_map_active, cells_map_full = get_active_mesh(full_mesh, active_cell_truth_tab)
        base_plate_mesh = meshio.Mesh(points=active_mesh.points, cells={'hexahedron': active_mesh.cells})
        base_plate_mesh.write(os.path.join(vtk_dir, f"base_plate_mesh.vtu"))
        cad_mesh = meshio.Mesh(points=full_mesh.points, cells={'hexahedron': full_mesh.cells})
        cad_mesh.write(os.path.join(vtk_dir, f"cad_mesh.vtu"))
        active_cell_truth_tab_old = active_cell_truth_tab

        external_faces, cells_face, hash_map, inner_faces, all_faces = initialize_hash_map(full_mesh, 
            active_cell_truth_tab, cells_map_full, ele_type)

        def neumann_top(point, old_T):
            # q is the heat flux into the domain
            d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
            q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2)
            q = q_laser
            #return np.array([q])
            return np.expand_dims(q,axis=0)

        def neumann_walls(point, old_T):
            # q is the heat flux into the domain
            q_conv = h*(T0 - old_T[0])
            q = q_conv
            #return np.array([q])
            return np.expand_dims(q,axis=0)

        neumann_bc_info_laser_on = [None, [neumann_walls, neumann_top]]
        neumann_bc_info_laser_off = [None, [neumann_walls]]

        full_sol = T0*np.ones((len(full_mesh.points), vec))  
        
        activation_time = onp.zeros((toolpath.shape[0],1),dtype=np.float64)
                
        for i in range(0,toolpath.shape[0]):
        # for i in range(0, 3):

            print(f"########## i = {i}, toolpath.shape = {toolpath.shape}, ffi = {ffi}")

            if i == 0:
                n_step = 1
            else:
                direction = toolpath[i, 1:4] - toolpath[i - 1 , 1:4]
                d = np.linalg.norm(direction)
                n_step = round(d/path_dx)
                activation_time[i,0] = dt*n_step + activation_time[i-1,0]
            if n_step > 1:
                num_laser_off = n_step-1
                sol = full_sol[points_map_active]
                problem = Thermal(active_mesh, vec=vec, dim=dim, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info_laser_off, 
                                  additional_info=(sol, rho, Cp, dt, external_faces))
                for j in range(n_step-1):
                    #print(f"\n############################################################")
                    #print(f"Laser off: i = {i} in {toolpath.shape[0]} , j = {j} in {num_laser_off}")
                    # old_sol = full_sol[points_map_active]
                    # problem.old_sol = old_sol
                    sol = solver(problem, linear=True,use_petsc=True)
                    problem.update_int_vars(sol)
                    full_sol = full_sol.at[points_map_active].set(sol)
                    #vtk_path = os.path.join(vtk_dir, f"u_{i:05d}_inactive_{j:05d}.vtu")
                    #save_sol(problem, sol, vtk_path)
            num_laser_on = 1
            laser_center = np.array([toolpath[i,1], toolpath[i,2], toolpath[i,3]])
            #print(f"laser center = {laser_center}")   
            flag_1 = centroids[:, 2] < laser_center[2]+ dx/4
            flag_2 = (centroids[:, 0] - laser_center[0])**2 + (centroids[:, 1] - laser_center[1])**2 <= rb**2
            active_cell_truth_tab = onp.logical_or(active_cell_truth_tab, onp.logical_and(flag_1, flag_2))
            active_mesh, points_map_active, cells_map_full = get_active_mesh(full_mesh, active_cell_truth_tab)
            sol = full_sol[points_map_active]
            external_faces, hash_map, inner_faces, all_faces = update_hash_map(active_cell_truth_tab_old, 
                        active_cell_truth_tab, cells_map_full, cells_face, hash_map, inner_faces, all_faces)
            if onp.all(active_cell_truth_tab == active_cell_truth_tab_old):
                        print(f"No element born")
                        # problem.old_sol = old_sol
            else:
                #print(f"New elements born {i}")        
                problem = Thermal(active_mesh, vec=vec, dim=dim, dirichlet_bc_info=[[],[],[]], neumann_bc_info=neumann_bc_info_laser_on, 
                                  additional_info=(sol, rho, Cp, dt, external_faces))    
                sol = solver(problem, linear=True,use_petsc=True)
                problem.update_int_vars(sol)
                full_sol = full_sol.at[points_map_active].set(sol)
                j=0
                vtk_path = os.path.join(vtk_dir, f"u_{i:05d}_active_{j:05d}.vtu")
                if i in sampled_deposits:
                    save_sol(problem, sol, vtk_path)
                    print(f"save {i} deposition")

            active_cell_truth_tab_old = active_cell_truth_tab
            gc.collect()
            
        t2 = default_timer()
        print("{} sec to compute one model".format(t2-t1))      
        onp.savetxt(osp.join(vtk_dir, Path(f).stem+"_activation_time.txt"), activation_time, delimiter=',')
    

if __name__ == "__main__":
    parameter_json_file = os.path.join(crt_dir, "am_parameters.json")
    problem_name = "extend_small_10_base_20"
    
    
    
    tracemalloc.start()
    ded_cad_model(parameter_json_file,problem_name)
    
    
    print("No.of tracked objects before calling get method")
    print(len( gc.get_objects() ) )
    gc.collect()
 
    print("No.of tracked objects after removing non-referenced objects")
    print(len( gc.get_objects() ) )
 
    
    # displaying the memory
    print(tracemalloc.get_traced_memory())
     
    # stopping the library
    tracemalloc.stop()                                                                                                                                                                                                      
