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
            Cp = 588.
            rho = 8440.
            return rho*Cp*(T - T_old)/dt
        return T_map

    def set_params(self, params):
        sol_T_old, laser_center, switch = params
        self.internal_vars['neumann'] = [[self.convert_neumann_from_dof(sol_T_old, 0)], 
                                         [self.convert_neumann_from_dof(sol_T_old, 1)]]
        self.internal_vars['mass'] = [self.convert_from_dof_to_quad(sol_T_old)]
        self.neumann_value_fns[0] = get_thermal_neumann_top(laser_center, switch)


class Plasticity(FEM):
    """We solve the following equation (weak form of FEM):
    (sigma(u_grad), v_grad) * dx = 0
    where u is the trial function, and v is the test function.
    """
    def custom_init(self):
        sigmas_old = np.zeros((len(self.cells), self.num_quads, self.vec, self.dim))
        epsilons_old = np.zeros_like(sigmas_old)
        dT = np.zeros((len(self.cells), self.num_quads, 1))
        phase = np.ones_like(dT, dtype=np.int32)*POWDER
        self.internal_vars['laplace'] = [sigmas_old, epsilons_old, dT, phase]
    
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

        def stress_return_maps(u_grad, sigma_old, epsilon_old, dT, phase):
            E0 = 70.e9
            sig0 = 250.e6 
            alpha_V0 = 1e-5
            alpha_V = np.where(phase == SOLID, alpha_V0, 0.)
            E = np.where(phase == SOLID, E0, 1e-2*E0)
            epsilon_inc_T = alpha_V*dT*np.eye(self.dim)
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc - epsilon_inc_T, E) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev, s_norm)
            return sigma, sigma[0, 0]

        stress_return_map = lambda *args: stress_return_maps(*args)[0]
        yield_val_fn = lambda *args: stress_return_maps(*args)[1]

        return strain, stress_return_map, yield_val_fn

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
        sigmas_old, epsilons_old, dT, phase = params
        sigmas_update = vmap_stress_rm(u_grads, sigmas_old, epsilons_old, dT, phase)
        epsilons_update = vmap_strain(u_grads)
        f_yield_vals = vmap_yield_val_fn(u_grads, sigmas_old, epsilons_old, dT, phase)
        return [sigmas_update, epsilons_update, dT, phase], f_yield_vals

    def update_params(self, dT, T, params):
        """Update dT
        Keep sigmas and epsilons unchanged
        """
        sigmas, epsilons, _, phase = params
        dT_update = self.convert_from_dof_to_quad(dT)
        T_quad = self.convert_from_dof_to_quad(T)
        powder_to_liquid = (phase == POWDER) & (T_quad > Tl)
        liquid_to_solid = (phase == LIQUID) & (T_quad < Tl)
        phase = phase.at[powder_to_liquid].set(LIQUID)
        phase = phase.at[liquid_to_solid].set(SOLID)
        return sigmas, epsilons, dT_update, phase

    def set_params(self, params):
        self.internal_vars['laplace'] = params


dt = 2*1e-6

POWDER = 0
LIQUID = 1
SOLID = 2 

# Inconel 625 material
# All units in SI standard
vel = 0.5 # laser scanning velocity
T0 = 300. # ambient temperature
Tl = 1623 # liquidus temperature
h = 100. # heat convection coefficient
rb = 0.05e-3 # laser beam size
eta = 0.25 # absorption rate
P = 50. # laser power
SB_constant = 5.67e-8 # Stefan-Boltzmann constant
emissivity = 0.3 # emissivity
ele_type = 'HEX8'
vtk_dir = os.path.join(data_dir, 'vtk')
Nx, Ny, Nz = 50, 20, 5
Lx, Ly, Lz = 0.5e-3, 0.2e-3, 0.05e-3 # domain size
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
    return left | right | front | back

def get_thermal_neumann_top(laser_center, switch):
    # Neumann BC values for thermal problem
    def thermal_neumann_top(point, old_T):
        # q is the heat flux into the domain
        d2 = (point[0] - laser_center[0])**2 + (point[1] - laser_center[1])**2
        q_laser = 2*eta*P/(np.pi*rb**2) * np.exp(-2*d2/rb**2) * switch
        q_conv = h*(T0 - old_T[0])
        q_rad = SB_constant*emissivity*(T0**4 - old_T[0]**4)
        q = q_conv + q_rad + q_laser
        return np.array([q])
    return thermal_neumann_top

def thermal_neumann_walls(point, old_T):
    # q is the heat flux into the domain
    q_conv = h*(T0 - old_T[0])
    q_rad = SB_constant*emissivity*(T0**4 - old_T[0]**4)
    q = q_conv + q_rad
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
sol_T_old_for_u = np.array(sol_T_old)
problem_T = Thermal(mesh, vec=1, dim=3, dirichlet_bc_info=dirichlet_bc_info_T, neumann_bc_info=neumann_bc_info_T)

# Define mechanical problem
dirichlet_bc_info_u = [[bottom]*3, [0, 1, 2], [displacement_dirichlet_bottom]*3]
problem_u = Plasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info_u)
params_u = problem_u.internal_vars['laplace']
sol_u = np.zeros((problem_u.num_total_nodes, problem_u.vec))

# Clean folder
files = glob.glob(os.path.join(vtk_dir, f'*'))
for f in files:
    os.remove(f)

# Save initial solution
vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
save_sol(problem_T, sol_T_old, vtk_path, point_infos=[('u', np.zeros((len(sol_T_old), 3)))], 
                                         cell_infos=[('plastic', np.zeros(len(mesh.cells))),
                                                     ('phase', np.mean(params_u[-1][:, :, 0], axis=1))])

laser_on_t = 0.5*Lx/vel
simulation_t = 2*laser_on_t
ts = np.arange(0., simulation_t + dt, dt)
for i in range(len(ts[1:])):
    laser_center = np.array([Lx*0.25 + vel*ts[i + 1], Ly/2., Lz])
    print(f"\nStep {i + 1}, total step = {len(ts[1:])}, laser_x = {laser_center[0]}, Lx = {Lx}")
    if ts[i + 1] >= laser_on_t:
        print(f"### LASER TURNED OFF...")

    switch = np.where(ts[i + 1] < laser_on_t, 1., 0.)

    problem_T.set_params([sol_T_old, laser_center, switch])
    sol_T_new = solver(problem_T, use_petsc=False)

    if (i + 1) % 10 == 0:
        # Sometimes T solution is smaller than T0=300K. This is a known problem with FEM for thermal problem. OK for now.

        T_new_modified = sol_T_new
        T_new_modified = np.where(T_new_modified < T0, T0, T_new_modified)
        T_new_modified = np.where(T_new_modified > Tl, Tl, T_new_modified)
        T_old_modified = sol_T_old_for_u
        T_old_modified = np.where(sol_T_old < T0, T0, T_old_modified)
        T_old_modified = np.where(sol_T_old > Tl, Tl, T_old_modified)

        dT = T_new_modified - T_old_modified
        print(f"max dT = {np.max(dT)}")
        params_u = problem_u.update_params(dT, sol_T_new, params_u)

        # Solve for u solution.
        problem_u.set_params(params_u)
        sol_u = solver(problem_u, initial_guess=sol_u, use_petsc=False)

        # For plasticity problem, we need to update the total strain and stress for convenience of next step
        params_u, plastic_info = problem_u.update_stress_strain(sol_u, params_u) 

        # Check if plastic deformation occurs (with f_yield_vals > 0.)
        print(f"max plastic_info = {np.max(plastic_info)}")
    
        # Update T solution
        sol_T_old_for_u = sol_T_new

        vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
        save_sol(problem_T, sol_T_old, vtk_path, point_infos=[('u', sol_u)], 
                                                 cell_infos=[('plastic', np.mean(plastic_info, axis=1)),
                                                             ('phase', np.max(params_u[-1][:, :, 0], axis=1))])
    # Update T solution
    sol_T_old = sol_T_new