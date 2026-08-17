import jax
import jax.numpy as np
import numpy as onp
import glob
import os

from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol

from applications.crystal_plasticity.models import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'volume_sensitivity'
output_dir = os.path.join(os.path.dirname(__file__), 'output', case_name)
vtk_dir = os.path.join(output_dir, 'vtk')


def problem():
    class CrystalPlasticityModified(CrystalPlasticity):
        def set_params(self, all_params):
            disp, params = all_params
            self.internal_vars = params
            self.fes[0].dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
            self.fes[0].update_Dirichlet_boundary_conditions(self.fes[0].dirichlet_bc_info)

        def target_value(self, sol):
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
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, output_dir, ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    os.makedirs(vtk_dir, exist_ok=True)
    for vtk_path in glob.glob(os.path.join(vtk_dir, '*.vtu')):
        os.remove(vtk_path)

    num_steps = 2
    disps = np.linspace(0., 0.001, num_steps + 1)
    ts = np.linspace(0., 0.1, num_steps + 1)

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
    initial_params = problem.internal_vars
    fwd_pred = ad_wrapper(problem)

    def simulation(displacement_scale, save_results=False):
        params = initial_params
        for i in range(num_steps):
            print(f"\nStep {i + 1} in {num_steps}, disp = {disps[i + 1]}")
            problem.dt = ts[i + 1] - ts[i]
            sol_list = fwd_pred([displacement_scale*disps[i + 1], params])
            sol = sol_list[0]
            params = problem.update_int_vars_gp(sol, params)
            obj_val = problem.target_value(sol)
            print(f"obj_val = {obj_val}")

            if save_results:
                vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
                save_sol(problem.fes[0], sol, vtk_path)
        return obj_val

    simulation(1., save_results=True)

    grads = jax.grad(simulation)(1.)
    print(f"grads = {grads}")


if __name__ == "__main__":
    problem()
 
