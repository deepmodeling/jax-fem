import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.core import FEM
from jax_am.fem.utils import save_sol

# If you have multiple GPUs, set the one to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')


class Thermal(FEM):
    """We solve the following equation (weak form of FEM):
    (rho*Cp/dt*(T_crt-T_old), Q) * dx + (k*T_crt_grad, Q_grad) * dx - (heat_flux, Q) * ds = 0
    where T_crt is the trial function, and Q is the test function.
    """
    def get_tensor_map(self):
        def fn(u_grad):
            k = 15.
            return k*u_grad
        return fn
 
    def get_mass_map(self):
        def T_map(T, T_old):
            Cp = 500.
            rho = 8440.
            return rho*Cp*(T - T_old)/dt
        return T_map

    def set_params(self, params):
        sol_T_old, laser_center = params
        self.internal_vars['neumann'] = [[], [self.convert_neumann_from_dof(sol_T_old, 1)]]
        self.internal_vars['mass'] = [self.convert_from_dof_to_quad(sol_T_old)]
        self.neumann_value_fns[0] = get_thermal_neumann_top(laser_center)


class Plasticity(FEM):
    """We solve the following equation (weak form of FEM):
    (sigma(u_grad), v_grad) * dx = 0
    where u is the trial function, and v is the test function.
    """
    def custom_init(self):
        sigmas_old = onp.zeros((len(self.cells), self.num_quads, self.vec, self.dim))
        epsilons_old = onp.zeros_like(sigmas_old)
        dT =  onp.zeros((len(self.cells), self.num_quads, 1))
        self.internal_vars['laplace'] = [sigmas_old, epsilons_old, dT]
    
    def get_tensor_map(self):
        _, stress_return_map, _ = self.get_maps()
        return stress_return_map

    def get_maps(self):
        def safe_sqrt(x):  
            safe_x = np.where(x > 0., np.sqrt(x), 0.)
            return safe_x

        def safe_divide(x, y):
            return np.where(y == 0., 0., x/y)

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress(epsilon, E):
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def stress_return_maps(u_grad, sigma_old, epsilon_old, dT):
            E = 70.e9
            sig0 = 250.e6 
            alpha_V = 1e-5
            epsilon_inc_T = alpha_V*dT*np.eye(self.dim)
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc - epsilon_inc_T, E) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev, s_norm)
            return sigma, f_yield_plus

        stress_return_map = lambda *args: stress_return_maps(*args)[0]
        yield_val_fn = lambda *args: stress_return_maps(*args)[1]

        return strain, stress_return_map, yield_val_fn

    def convert_node_to_quad(self, sol):
        """sol is defined as nodal DOFs, which needs to be converted to quadrature values
        """
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        return u

    def vmap_stress_strain_fns(self):
        strain, stress_return_map, yield_val_fn = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        vmap_yield_val_fn = jax.vmap(jax.vmap(yield_val_fn))
        return vmap_strain, vmap_stress_return_map, vmap_yield_val_fn

    def update_stress_strain(self, sol, params):
        """Update sigmas and epsilons
        Keep dT unchanged
        Output f_yield_vals for debugging purpose: we hope to know if plastic deformation occurs
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        vmap_strain, vmap_stress_rm, vmap_yield_val_fn = self.vmap_stress_strain_fns()
        sigmas_old, epsilons_old, dT = params
        sigmas_update = vmap_stress_rm(u_grads, sigmas_old, epsilons_old, dT)
        epsilons_update = vmap_strain(u_grads)
        f_yield_vals = vmap_yield_val_fn(u_grads, sigmas_old, epsilons_old, dT)
        return [sigmas_update, epsilons_update, dT], f_yield_vals

    def update_dT(self, dT, params):
        """Update dT
        Keep sigmas and epsilons unchanged
        """
        sigmas, epsilons, _ = params
        dT_update = self.convert_node_to_quad(dT)
        return sigmas, epsilons, dT_update

    def set_params(self, params):
        sol_T_old, 
        self.internal_vars['laplace'] = params


dt = 1e-6

# Inconel 625 material
# All units in SI standard
vel = 0.5 # laser scanning velocity
T0 = 300. # ambient temperature
h = 100. # heat convection coefficient
rb = 0.1e-3 # laser beam size
eta = 0.4 # absorption rate
P = 100. # laser power
ele_type = 'HEX8'
vtk_dir = os.path.join(data_dir, 'vtk')
problem_name = 'example'
Nx, Ny, Nz = 100, 25, 25
Lx, Ly, Lz = 1.e-3, 0.25e-3, 0.25e-3 # domain size
meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Define boundaries
def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def walls(point):
    left = np.isclose(point[0], 0., atol=1e-5)
    right = np.isclose(point[0], Lx, atol=1e-5)
    front = np.isclose(point[1], 0., atol=1e-5)
    back = np.isclose(point[1], Ly, atol=1e-5)
    return np.logical_or(np.logical_or(np.logical_or(left, right), front), back)


def get_thermal_neumann_top(laser_center):
    # Neumann BC values for thermal problem
    def thermal_neumann_top(point):
        # q is the heat flux into the domain
        d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
        q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2)
        q = q_laser
        return np.array([q])
    return thermal_neumann_top

def thermal_neumann_walls(point, old_T):
    # q is the heat flux into the domain
    q_conv = h*(T0 - old_T[0])
    q = q_conv
    return np.array([q])

# Dirichlet BC values for thermal problem
def thermal_dirichlet_bottom(point):
    return T0

# Dirichlet BC values for mechanical problem
def displacement_dirichlet_bottom(point):
    return 0.

# Define thermal problem
dirichlet_bc_info_T = [[bottom], [0], [thermal_dirichlet_bottom]]
neumann_bc_info_T = [[top, walls], [None, thermal_neumann_walls]]
sol_T_old = T0*np.ones((len(mesh.points), 1))
problem_T = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info_T, neumann_bc_info=neumann_bc_info_T)

# Define mechanical problem
dirichlet_bc_info_u = [[bottom]*3, [0, 1, 2], [displacement_dirichlet_bottom]*3]
problem_u = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info_u)
params_u = problem_u.internal_vars['laplace']

# Clean folder
files = glob.glob(os.path.join(vtk_dir, f'{problem_name}/*'))
for f in files:
    os.remove(f)

# Save initial solution
vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{0:05d}.vtu")
save_sol(problem_T, sol_T_old, vtk_path, point_infos=[('u', np.zeros((len(sol_T_old), 3)))], cell_infos=[('f_plus', np.zeros(len(mesh.cells)))])

total_t = 0.2*Lx/vel
ts = np.arange(0., total_t, dt)
for i in range(len(ts[1:])):
    # TODO: Plasticity equation is more computational expensive to solve than thermal equation.
    # Could solve thermal equation for a couple of steps, and then solve plasticity equation for one step.
    print(f"\nStep {i + 1}, total step = {len(ts)}, laser_x = {Lx*0.2 + vel*ts[i + 1]}")

    laser_center = np.array([Lx*0.2 + vel*ts[i + 1], Ly/2., Lz])

    # Temperature solution from previous step affects current step solution
    problem_T.set_params([sol_T_old, laser_center])

    # Solve for T solution. If you're using CPU, we recommend to set use_petsc=True
    # If you're using GPU, we recommend to set use_petsc=False
    sol_T_new = solver(problem_T, use_petsc=True)

    # Sometimes T solution is smaller than T0=300K. This is a known problem with FEM for thermal problem. OK for now.
    # First update dT and let problem_u know this update
    dT = sol_T_new - sol_T_old
    print(f"max dT = {np.max(dT)}")
    params_u = problem_u.update_dT(dT, params_u)
    problem_u.set_params(params_u)

    # Solve for u solution.
    sol_u = solver(problem_u, use_petsc=True)

    # For plasticity problem, we need to update the total strain and stress for convenience of next step
    params_u, f_yield_vals = problem_u.update_stress_strain(sol_u, params_u) 

    # Update T solution
    sol_T_old = sol_T_new

    # Check if plastic deformation occurs (with f_yield_vals > 0.)
    print(f"max f_yield_vals = {np.max(f_yield_vals)}")
    
    if (i + 1) % 1 == 0:
        vtk_path = os.path.join(vtk_dir, f"{problem_name}/u_{i + 1:05d}.vtu")
        save_sol(problem_T, sol_T_old, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('f_plus', np.max(f_yield_vals, axis=1))])
