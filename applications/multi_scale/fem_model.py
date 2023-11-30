import numpy as onp
import jax
import jax.numpy as np

from jax_fem.models import Mechanics

from applications.fem.multi_scale.arguments import args
from applications.fem.multi_scale.trainer import get_nn_batch_forward
from applications.fem.multi_scale.utils import tensor_to_flat


class HyperElasticity(Mechanics):
    """Three modes: rve, dns, nn
    """
    def custom_init(self, mode, dns_info):
        self.mode = mode
        self.dns_info = dns_info
        if self.mode == 'rve':
            self.H_bar = None
            self.physical_quad_points = self.get_physical_quad_points()
            self.E, self.nu = self.compute_moduli()
            self.internal_vars = {'laplace': [self.E, self.nu]}
        elif self.mode == 'dns':
            self.physical_quad_points = self.get_physical_quad_points()
            self.E, self.nu = self.compute_moduli()
            self.internal_vars = {'laplace': [self.E, self.nu]}
        elif self.mode == 'nn':
            # hyperparam = 'default'
            # It turns out that MLP2 has the lowest validation error.
            hyperparam = 'MLP2'
            self.nn_batch_forward = get_nn_batch_forward(hyperparam)
        else:
            raise NotImplementedError(f"mode = {self.mode} is not defined.")

    def get_tensor_map(self):
        stress_map, _ = self.get_maps()
        return stress_map

    def get_maps(self):
        if self.mode == 'rve':
            return self.maps_rve()
        elif self.mode == 'dns':
            return self.maps_dns()
        elif self.mode == 'nn':
            return self.maps_nn()
        else:
            raise NotImplementedError(f"get_maps Only support rve, dns or nn.")

    def stress_strain_fns(self):
        if self.mode == 'rve':  
            stress, psi = self.maps_rve()
            vmap_stress = lambda x: jax.vmap(jax.vmap(stress))(x, self.E, self.nu)
            vmap_energy = lambda x: jax.vmap(jax.vmap(psi))(x + self.H_bar[None, None, :, :], self.E, self.nu)
        elif self.mode == 'dns':
            stress, psi = self.maps_dns()
            vmap_stress = lambda x: jax.vmap(jax.vmap(stress))(x, self.E, self.nu)
            vmap_energy = lambda x: jax.vmap(jax.vmap(psi))(x, self.E, self.nu)
        elif self.mode == 'nn':
            stress, psi = self.maps_nn()
            vmap_stress = jax.vmap(jax.vmap(stress))
            vmap_energy = jax.vmap(jax.vmap(psi))
        else:
            raise NotImplementedError(f"get_maps Only support rve, dns or nn.")
        return vmap_stress, vmap_energy

    def maps_rve(self):
        def psi(F, E, nu):
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I + self.H_bar
            P = P_fn(F, E, nu)
            return P

        return first_PK_stress, psi

    def maps_dns(self):
        def psi(F, E, nu):
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, E, nu)
            return P

        return first_PK_stress, psi

    def maps_nn(self):
        def psi(F):
            C = F.T @ F
            C_flat = tensor_to_flat(C)
            energy = self.nn_batch_forward(C_flat[None, :])[0]
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
 
        return first_PK_stress, psi

    def compute_energy(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        F_reshape = u_grads + np.eye(self.dim)[None, None, :, :]
        _, vmap_energy  = self.stress_strain_fns()
        psi = vmap_energy(F_reshape) # (num_cells, num_quads)
        energy = np.sum(psi * self.JxW)
        return energy

    def fluc_to_disp(self, sol_fluc):
        sol_disp = (self.H_bar @ self.points.T).T + sol_fluc
        return sol_disp

    def compute_moduli(self):
        def moduli_map(point):
            inclusion = False
            for i in range(args.num_units_x):
                for j in range(args.num_units_y):
                    for k in range(args.num_units_z):
                        center = np.array([(i + 0.5)*args.L, (j + 0.5)*args.L, (k + 0.5)*args.L])
                        hit = np.max(np.absolute(point - center)) < args.L*args.ratio
                        inclusion = np.logical_or(inclusion, hit)
            E = np.where(inclusion, args.E_in, args.E_out)
            nu = np.where(inclusion, args.nu_in, args.nu_out)
            return E, nu

        E, nu = jax.vmap(jax.vmap(moduli_map))(self.physical_quad_points)

        if self.dns_info == 'in':
            E = args.E_in*np.ones_like(E)
            nu = args.nu_in*np.ones_like(nu)

        if self.dns_info == 'out':
            E = args.E_out*np.ones_like(E)
            nu = args.nu_out*np.ones_like(nu)

        return E, nu

    def compute_traction(self, location_fn, sol):
        """Not robust.
        """
        def traction_fn(u_grads):
            # u_grads: (num_selected_faces, num_face_quads, vec, dim) 
            if self.mode == 'dns':
                stress, _ = self.maps_dns()
                if self.dns_info == 'in':
                    vmap_stress = lambda x: jax.vmap(jax.vmap(stress))(x, args.E_in*np.ones(u_grads.shape[:2]), 
                                                                       args.nu_in*np.ones(u_grads.shape[:2]))
                else:
                    vmap_stress = lambda x: jax.vmap(jax.vmap(stress))(x, args.E_out*np.ones(u_grads.shape[:2]), 
                                                                       args.nu_out*np.ones(u_grads.shape[:2]))
            elif self.mode == 'nn':
                vmap_stress, _ = self.stress_strain_fns()
            else:
                raise NotImplementedError(f"traction_fn only support dns and nn.")

            sigmas = jax.jit(vmap_stress)(u_grads)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.]).reshape((self.dim, 1))
            traction = (sigmas @ normals[None, None, :, :])[:, :, :, 0]
            return traction

        traction_integral_val = self.surface_integral(location_fn, traction_fn, sol)
        return traction_integral_val
