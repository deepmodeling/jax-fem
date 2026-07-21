import numpy as onp
import jax
import jax.numpy as np
import meshio
import time
import os
import glob
import scipy.optimize as opt

from jax_fem.core import FEM
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import modify_vtu_file, save_sol
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh

onp.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# TODO
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LinearPoisson(FEM):
    def custom_init(self, name):
        self.name = name

    def get_tensor_map(self):
        return lambda x: x

    def get_body_map(self):
        return lambda x: x

    def compute_L2(self, sol):
        kernel = self.get_mass_kernel(lambda x: x**2)
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        val = jax.vmap(kernel)(cells_sol, self.JxW) # (num_cells, num_nodes, vec)
        return np.sum(val)

    def set_params(self, params):
        self.internal_vars['body'] = params.reshape((self.num_total_nodes, self.vec))

def taylor_tests(data_dir, m, fn, fn_grad):
    """See https://www.dolfin-adjoint.org/en/latest/documentation/verification.html
    """
    hs = onp.array([1e-4, 1e-3, 1e-2, 1e-1])
    direction = np.ones_like(m)
    J = fn(m)
    res_zero = []
    res_first = []
    for h in hs:
        m_perturb = direction*h + m
        r_zero = onp.absolute(fn(m_perturb) - J)
        r_first = onp.absolute(fn(m_perturb) - J - h*np.dot(fn_grad(m), direction))
        res_zero.append(r_zero)
        res_first.append(r_first)

    onp.save(os.path.join(data_dir, f"numpy/res.npy"), onp.stack((hs, res_zero, res_first)))


def param_id():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    files = glob.glob(os.path.join(data_dir, f'vtk/inverse/*'))
    for f in files:
        os.remove(f)

    Lx, Ly, Lz = 1., 1., 0.2
    Nx, Ny, Nz = 50, 50, 10
    meshio_mesh = box_mesh_gmsh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def min_x_loc(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def max_x_loc(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def min_y_loc(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def max_y_loc(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def min_z_loc(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def max_z_loc(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def body_force(point):
        center1 = np.array([Lx/4., Ly/4., Lz/2.])
        val1 = 10.*np.exp(-10*np.sum((point - center1)**2))
        center2 = np.array([Lx*3./4., Ly*3./4., Lz/2.])
        val2 = 10.*np.exp(-10*np.sum((point - center2)**2))
        return np.array([val1 + val2])

    dirichlet_bc_info = [[min_x_loc, max_x_loc, min_y_loc, max_y_loc, min_z_loc, max_z_loc], 
                         [0]*6, 
                         [zero_dirichlet_val]*6]

    problem_fwd_name = "forward"                 
    problem_fwd = LinearPoisson(jax_mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info, 
        source_info=body_force, additional_info=(problem_fwd_name,))
    true_sol = solver(problem_fwd, linear=True)
    true_body_force = jax.vmap(body_force)(problem_fwd.points)
    vtu_path = os.path.join(data_dir, f"vtk/{problem_fwd_name}/u.vtu")
    save_sol(problem_fwd, true_sol, vtu_path, point_infos=[('source', true_body_force)])
    print(f"True force L2 integral = {problem_fwd.compute_L2(true_body_force)}")
    
    num_obs_pts = 250

    observed_inds = onp.random.choice(onp.arange(len(jax_mesh.points)), size=num_obs_pts, replace=False)
    observed_points = jax_mesh.points[observed_inds]
    cells = [[i%num_obs_pts, (i + 1)%num_obs_pts, (i + 2)%num_obs_pts] for i in range(num_obs_pts)]
    mesh = meshio.Mesh(observed_points, [("triangle", cells)])
    mesh.write(os.path.join(data_dir, f"vtk/{problem_fwd_name}/points.vtu"))
    true_vals = true_sol[observed_inds]

    problem_inv_name = "inverse"
    problem_inv = LinearPoisson(jax_mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info, additional_info=(problem_inv_name,))
    fwd_pred = ad_wrapper(problem_inv, linear=True)

    files = glob.glob(os.path.join(data_dir, f'vtk/{problem_inv_name}/*'))
    for f in files:
        os.remove(f)

    def J_fn(dofs, params):
        """J(u, p)
        """
        sol = dofs.reshape((problem_inv.num_total_nodes, problem_inv.vec))
        params = params.reshape(sol.shape)
        pred_vals = sol[observed_inds]
        assert pred_vals.shape == true_vals.shape
        l2_loss = np.sum((pred_vals - true_vals)**2) 
        # l2_loss = problem_fwd.compute_L2(true_sol - sol)
        # reg = 1e-5*problem_fwd.compute_L2(params)
        # print(f"{bcolors.HEADER}Predicted force L2 integral = {problem_fwd.compute_L2(params)}{bcolors.ENDC}")
        return l2_loss


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
        vtu_path = os.path.join(data_dir, f"vtk/{problem_inv_name}/sol_{output_sol.counter:03d}.vtu")
        save_sol(problem_inv, sol, vtu_path, point_infos=[('source', params)])
        rel_error_sol = (np.sqrt(problem_fwd.compute_L2(true_sol - sol))/
                         np.sqrt(problem_fwd.compute_L2(true_sol)))
        rel_error_force = (np.sqrt(problem_fwd.compute_L2(params.reshape(sol.shape) - true_body_force))/
                           np.sqrt(problem_fwd.compute_L2(true_body_force)))
        print(f"loss = {obj_val}")
        print(f"max true source = {np.max(true_body_force)}, min source = {np.min(true_body_force)}, mean source = {np.mean(true_body_force)}")
        print(f"max opt source = {np.max(params)}, min source = {np.min(params)}, mean source = {np.mean(params)}")
        print(f"rel_error_sol = {rel_error_sol}, rel_error_force = {rel_error_force}")
        outputs.append([obj_val, rel_error_sol, rel_error_force])
        output_sol.counter += 1

    output_sol.counter = 0

    fn, fn_grad = J_total, jax.grad(J_total)
    params_ini = onp.zeros(problem_inv.num_total_nodes * problem_inv.vec)
    taylor_tests(data_dir, params_ini, fn, fn_grad)

    def objective_wrapper(x):
        obj_val, dJ = jax.value_and_grad(J_total)(x)
        objective_wrapper.dJ = dJ
        output_sol(x, obj_val)
        print(f"{bcolors.HEADER}obj_val = {obj_val}{bcolors.ENDC}")
        return obj_val

    def derivative_wrapper(x):
        grads = objective_wrapper.dJ
        print(f"grads.shape = {grads.shape}")
        # 'L-BFGS-B' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(grads, order='F', dtype=onp.float64)
    
    bounds = None
    options = {'maxiter': 20, 'disp': True, 'gtol': 1e-20}  # CG or L-BFGS-B or Newton-CG or SLSQP
    res = opt.minimize(fun=objective_wrapper,
                       x0=params_ini,
                       method='L-BFGS-B',
                       jac=derivative_wrapper,
                       bounds=bounds,
                       callback=None,
                       options=options)

    onp.save(os.path.join(data_dir, f"numpy/outputs.npy"), onp.array(outputs).T)


if __name__=="__main__":
    param_id()
