import numpy as onp
import jax
import jax.numpy as np

from jax_am.fem.core import FEM


class Elasticity(FEM):
    def custom_init(self, case_flag):
        self.neumann_boundary_inds = self.get_boundary_conditions_inds(self.neumann_bc_info[0])[0]
        self.cell_centroids = onp.mean(onp.take(self.points, self.cells, axis=0), axis=1)
        self.flex_inds = np.arange(len(self.cells))
        if case_flag == 'freecad':
            self.get_tensor_map = self.get_tensor_map_linearelasticity
        elif case_flag == 'plate':
            self.get_tensor_map = self.get_tensor_map_hyperelasticity
        elif case_flag == 'multi_material':
            self.get_tensor_map = self.get_tensor_map_multi_material
        else:
            raise ValueError(f"Unknown case_flag = {case_flag}")

    def get_tensor_map_linearelasticity(self):
        def stress(u_grad, theta):
            Emax = 70.e3
            Emin = 70.
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*(theta[0]+0.01)**penal
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def get_tensor_map_hyperelasticity(self):
        def psi(F, theta):
            Emax = 1e3
            Emin = 1.
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*(theta[0]+0.01)**penal
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, theta):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, theta)
            return P
        return first_PK_stress

    def get_tensor_map_multi_material(self):
        def stress(u_grad, theta):
            Emax = 70.e3
            Emin = 70.
            nu = 0.3
            penal = 3.

            # E1 = Emax
            # E2 = 0.5*Emax

            E1 = Emax
            E2 = 0.5*Emax

            rho_r = 0.4
            E_r = 0.2

            theta1, theta2 = theta

            # val1 = E_r*theta**penal/rho_r**penal
            # val2 = (1 - E_r)/(1 - rho_r**penal)*(theta**penal - rho_r**penal) + E_r
            # ratio = np.where(theta < rho_r, val1, val2)
            # E = Emin + (Emax - Emin)*ratio
            # E = Emin + (Emax - Emin)*(theta)**penal
            # E = Emax*theta**penal + 0.5*Emax*(1-theta)**penal

            E = Emin + theta1**penal*(theta2**penal*E1 + (1 - theta2**penal)*E2)

            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress 


    def set_params(self, params):
        full_params = np.ones((self.num_cells, params.shape[1]))
        full_params = full_params.at[self.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars['laplace'] = [thetas]

    def compute_compliance(self, neumann_fn, sol):
        boundary_inds = self.neumann_boundary_inds
        _, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :] * self.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
        traction = jax.vmap(jax.vmap(neumann_fn))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val

    def get_von_mises_stress_fn(self):
        stress_fn = self.get_tensor_map_linearelasticity()
        def vm_stress_fn(u_grad, theta):
            sigma = stress_fn(u_grad, theta)
            s_dev = sigma - 1./self.dim*np.trace(sigma)*np.eye(self.dim)
            vm_s = np.sqrt(3./2.*np.sum(s_dev*s_dev))
            return vm_s
        return vm_stress_fn

    def compute_von_mises_stress(self, sol):
        """TODO: Move this to jax-am library?
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        vm_stress_fn = self.get_von_mises_stress_fn()
        vm_stress = jax.vmap(jax.vmap(vm_stress_fn))(u_grads, *self.internal_vars['laplace']) # (num_cells, num_quads)
        volume_avg_vm_stress = np.sum(vm_stress * self.JxW, axis=1) / np.sum(self.JxW, axis=1) # (num_cells,)
        return volume_avg_vm_stress
