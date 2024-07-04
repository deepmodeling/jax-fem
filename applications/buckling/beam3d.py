import jax
import jax.numpy as np
import os
import glob

from jax_fem.problem import Problem
from jax_fem.solver import solver, dynamic_relax_solve
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 10.
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress


data_dir = os.path.join(os.path.dirname(__file__), 'output')
files = glob.glob(os.path.join(data_dir, f'vtk/*'))
for f in files:
    os.remove(f)

ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 20., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=100, Ny=5, Nz=5, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def left_dirichlet_val_x1(point):
    return 0.

def left_dirichlet_val_x2(point):
    return 0.

def left_dirichlet_val_x3(point):
    return 0.
 
def right_dirichlet_val_x1(point):
    return -0.2*Lx

def right_dirichlet_val_x2(point):
    return 2.*Ly

def right_dirichlet_val_x3(point):
    return 0.*Lz


dirichlet_bc_info = [[left]*3 + [right]*3, 
                     [0, 1, 2]*2, 
                     [left_dirichlet_val_x1, left_dirichlet_val_x2, left_dirichlet_val_x3, 
                      right_dirichlet_val_x1, right_dirichlet_val_x2, right_dirichlet_val_x3]]


problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
sol = dynamic_relax_solve(problem)

vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol, vtk_path)
