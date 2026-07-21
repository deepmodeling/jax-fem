# Import standard modules.
import numpy as onp
import os
import pickle
import torch

# Import JAX modules.
import jax
import jax.numpy as np
from flax import linen as nn 
from jax import grad, jit, vmap
import flax

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh

# Import local modules
from applications.surrogate_model.train import SurrogateModel, Network

input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')


class NnbasedMetamaterial(Problem):
    # Use neural network surrogate model as the constitutive model for the metamaterial.
    def get_tensor_map(self):
        def P_fn(E):
            K2 = np.concatenate((E[0,0].reshape(-1), E[0,1].reshape(-1), E[1,1].reshape(-1)), 0)
            P = self.surrogate_model.compute_input_gradient(self.surrogate_model.params, K2) - self.p_initial
            PK2 = np.concatenate((P[0].reshape(-1), P[1].reshape(-1), P[1].reshape(-1), P[2].reshape(-1)), 0).reshape(2,2)
            return PK2
 
        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            C = F.T @ F
            E = 0.5*(C - I)
            Pk2 = P_fn(E)
            Pk1 = F @ Pk2
            return Pk1
        return first_PK_stress

def problem():
    # Specify mesh-related information
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)


    vtk_dir = os.path.join(output_dir, 'vtk')
    dataset_dir = os.path.join(input_dir, 'dataset')
    os.makedirs(vtk_dir, exist_ok=True)
    Lx, Ly = 30, 30
    meshio_mesh = rectangle_mesh(Nx=30,
                                Ny=30,
                                domain_x=Lx, 
                                domain_y=Ly)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Define boundary locations.
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def dirichlet_val_bottom(point):
        return 0.
    
    # Create an instance of the displacement problem.
    def get_dirichlet_top(disp):
        def val_fn(x):
            return disp
        return val_fn

    location_fns = [bottom, bottom, top, top]
    vecs = [0, 1, 0, 1]
    dirichlet_bc_info = [location_fns, vecs, [dirichlet_val_bottom]*3 + [get_dirichlet_top(-0.05*30)]]

    # Create an instance of the problem.
    surrogate_problem = NnbasedMetamaterial(mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info)

    # Create surrogate model
    surrogate_problem.surrogate_model = SurrogateModel(Network)
    model_file_path = os.path.join(input_dir, 'model.pth')
    if not os.path.exists(model_file_path):
        raise ValueError(f"Please run the train.py file to train the model first.")

    pkl_file = pickle.load(open(model_file_path, "rb"))
    surrogate_problem.surrogate_model.params = flax.serialization.from_state_dict(target=surrogate_problem.surrogate_model.model, state=pkl_file)
    
    # Stress of strain-free state 
    e_initial = np.concatenate((np.array([0.]).reshape(-1), np.array([0.]).reshape(-1), np.array([0.]).reshape(-1)), 0)
    surrogate_problem.p_initial = surrogate_problem.surrogate_model.compute_input_gradient(surrogate_problem.surrogate_model.params, e_initial)
    
    # Solve problem
    sol = solver(surrogate_problem, solver_options={'petsc_solver':{}})
    surrogate_problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    vtk_path = os.path.join(vtk_dir, 'displacement.vtk')
    save_sol(surrogate_problem.fes[0], sol[0], vtk_path)


if __name__ == "__main__":
    problem()
