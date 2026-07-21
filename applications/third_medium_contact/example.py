"""
Third medium contact, generally following

Dahlberg, Vilmer, Filip Sjövall, Anna Dalklint, and Mathias Wallin. 
"A rotation-based approach to third medium contact regularization." 
Computer Methods in Applied Mechanics and Engineering 453 (2026): 118801.
"""

# Import some useful modules.
import jax
import jax.numpy as np
import os
import glob

# Import JAX-FEM specific modules.
from jax_fem.generate_mesh import rectangle_mesh, Mesh, get_meshio_cell_type
from jax_fem.solver import solver, ad_wrapper
from jax_fem.problem import Problem
from jax_fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

crt_file_path = os.path.dirname(__file__)
input_dir = os.path.join(crt_file_path, 'input')
output_dir = os.path.join(crt_file_path, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
length = 1000.
height = 500. 
arm = 100.

# files = glob.glob(os.path.join(vtk_dir, f'*'))
# for f in files:
#     os.remove(f)

# Unit system: [N], [mm]


class Contact(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        stress_return_map, _, _ = self.get_maps()
        return stress_return_map

    def get_maps(self):

        key = jax.random.PRNGKey(0)
        # may need to change noise level
        noise = jax.random.uniform(key, shape=(self.dim, self.dim), minval=-1e-6, maxval=1e-6)
        noise = np.diag(np.diag(noise))

        def deform_grad(u_grad):
            u_grad += noise
            F_2d = u_grad + np.eye(2)
            I = np.eye(3)
            F_3d = I.at[:2, :2].set(F_2d)
            return F_3d

        def rotation(F_3d):
            C = F_3d.T @ F_3d
            eigvals, eigvecs = np.linalg.eigh(C)
            sqrt_eigvals = np.sqrt(eigvals)
            Lambda_sqrt = np.diag(sqrt_eigvals)
            Q = eigvecs
            U = Q @ Lambda_sqrt @ Q.T
            R = F_3d @ np.linalg.inv(U)
            return R

        def psi(F_2d, theta, R_old):
            E = 1. # MPa
            nu = 0.4   
            K = E / (3. * (1. - 2. * nu))
            G = E / (2. * (1. + nu))
            I = np.eye(3)
            F_3d = I.at[:2, :2].set(F_2d)
            R = rotation(F_3d)
            # k_R = 5 * 1e-5 * 1e-6???
            k_R = 1e-2
            psi_reg = 0.5 * k_R * np.trace((I - R @ R_old.T) @ (I - R @ R_old.T).T)
            J = np.linalg.det(F_3d)
            I1 = np.trace(F_3d.T @ F_3d)
            Jinv = J**(-2.0/3.0)
            I1_bar = Jinv * I1
            log_J = np.log(J)
            psi_vol = 0.5 * K * log_J**2
            psi_iso = 0.5 * G * (I1_bar - 3.0)
            gamma_K = np.where(theta > 0.5, 1., 0.)
            gamma_G = np.where(theta > 0.5, 1., 1e-6)
            gamma_R = np.where(theta > 0.5, 0., 1.)
            energy = gamma_K*psi_vol + gamma_G*psi_iso + gamma_R*psi_reg
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, theta, R_old):
            u_grad += noise
            F_2d = u_grad + np.eye(2)
            P = P_fn(F_2d, theta, R_old)
            return P

        return first_PK_stress, deform_grad, rotation

    def post_processing(self, sol):
        u_grads = self.fe.sol_to_grad(sol)
        _, deform_grad, rotation = self.get_maps()
        vmap_deform_grad = jax.vmap(jax.vmap(deform_grad))
        vmap_rotation = jax.vmap(jax.vmap(rotation))
        Fs_3d = vmap_deform_grad(u_grads)
        Rs = vmap_rotation(Fs_3d)
        mean_Rs = np.mean(Rs, axis=1, keepdims=True)  # Shape is (num_cells, 1, 3, 3)
        mean_Rs = np.repeat(mean_Rs, repeats=Rs.shape[1], axis=1) # Shape is (num_cells, num_quads, 3, 3)
        return mean_Rs
        # return Rs

    def set_params(self, params):
        scale_d, thetas, Rs_old = params
        self.internal_vars = [thetas, Rs_old]
        self.fe.dirichlet_bc_info[-1][-1] = get_dirichlet_topright(scale_d)
        self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)


def get_dirichlet_topright(scale):
    def dirichlet_val_topright(point):
        z_disp = -scale*height
        return z_disp
    return dirichlet_val_topright


def problem():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    # h = 12.5
    h = 50
    meshio_mesh = rectangle_mesh(Nx=round(length/h) + 1, Ny=round(height/h), domain_x=length + h, domain_y=height)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def rightmost_node(point):
        return np.isclose(point[0], length, atol=1e-5) & np.isclose(point[1], height, atol=1e-5)

    def dirichlet_val_left(point):
        return 0.

    location_fns_dirichlet = [left]*2 + [rightmost_node]
    value_fns = [dirichlet_val_left]*2 + [get_dirichlet_topright(0)]
    vecs = [0, 1, 1]

    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    problem = Contact(mesh, vec=2, dim=2, ele_type=ele_type, quadrature_rule='Gauss-Lobatto-Legendre',
                      quadrature_order=2, dirichlet_bc_info=dirichlet_bc_info)

    physical_quad_points = problem.fe.get_physical_quad_points()
    cell_centroids = np.mean(physical_quad_points, axis=1, keepdims=True)
    cell_centroids_repeated = np.repeat(cell_centroids, repeats=physical_quad_points.shape[1], axis=1)

    thetas = np.ones((cell_centroids_repeated.shape[0], cell_centroids_repeated.shape[1]))
    thetas = np.where( (cell_centroids_repeated[:, :, 0] < length) &
                    ((cell_centroids_repeated[:, :, 0] < arm) |
                    (cell_centroids_repeated[:, :, 1] > height - arm) |
                    (cell_centroids_repeated[:, :, 1] < arm)), thetas, 0.)
    
    solver_options = {'petsc_solver': {'ksp_type': 'tfqmr', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def fwd_pred_seq():
        scale_ds = np.linspace(0., 1.2, 201) # h=50 will not work
        # scale_ds = np.linspace(0., 1.2, 501) # h=12.5 will work
        sol_list = [np.ones((problem.fe.num_total_nodes, problem.fe.vec))]
        for i in range(len(scale_ds)):
            print(f"\nStep {i + 1} in {len(scale_ds)}")
            Rs_old = problem.post_processing(sol_list[0])
            solver_options['initial_guess'] = sol_list
            sol_list = fwd_pred([scale_ds[i], thetas, Rs_old])
        return sol_list

    sol_list = fwd_pred_seq()
    vtk_path = os.path.join(vtk_dir, f'u.vtu')
    save_sol(problem.fes[0], np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), vtk_path, 
        cell_infos=[('theta', np.mean(thetas, axis=-1))])


if __name__ == "__main__":
    problem()
