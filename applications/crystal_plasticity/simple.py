import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol

from applications.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'calibration'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')


def problem():
    class CrystalPlasticityModified(CrystalPlasticity):
        def set_params(self, all_params):
            disp, params = all_params
            self.internal_vars = params
            self.fes[0].dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
            self.fes[0].update_Dirichlet_boundary_conditions(self.fes[0].dirichlet_bc_info)

        def targe_val(self, sol):
            tgt_volume = 1.01
            def det_fn(u_grad):
                F = u_grad + np.eye(self.dim)
                return np.linalg.det(F)

            u_grads = self.fes[0].sol_to_grad(sol)
            vmap_det_fn = jax.jit(jax.vmap(jax.vmap(det_fn)))
            crt_volume = np.sum(vmap_det_fn(u_grads) * self.fes[0].JxW)

            square_error = (crt_volume - tgt_volume)**2

            return square_error

    ele_type = 'HEX8'
    Nx, Ny, Nz = 1, 1, 1
    Lx, Ly, Lz = 1., 1., 1.

    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir, ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    disps = np.linspace(0., 0.005, 11)
    ts = np.linspace(0., 0.5, 11)

    def corner(point):
        flag_x = np.isclose(point[0], 0., atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], 0., atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn

    dirichlet_bc_info = [[corner, corner, bottom, top], 
                         [0, 1, 2, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    quat = onp.array([[1, 0., 0., 0.]])
    cell_ori_inds = onp.zeros(len(mesh.cells), dtype=onp.int32)
    problem = CrystalPlasticityModified(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                                additional_info=(quat, cell_ori_inds))
    fwd_pred = ad_wrapper(problem)

    def simulation(scale_d):
        params = problem.internal_vars
        results_to_save = []
        for i in range(2):
            print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}")
            problem.dt = ts[i + 1] - ts[i]
            sol_list = fwd_pred([scale_d*disps[i + 1], params])  
            params = problem.update_int_vars_gp(sol_list[0], params)
            obj_val = problem.targe_val(sol_list[0])
            print(f"obj_val = {obj_val}")
        return obj_val

    simulation(1.)

    grads = jax.grad(simulation)(1.)
    print(f"grads = {grads}")


if __name__ == "__main__":
    problem()
 