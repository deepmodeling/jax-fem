import numpy as onp
import jax
import jax.numpy as np

from jax_am.fem.core import FEM


class LinearPoisson(FEM):
    def get_tensor_map(self):
        return lambda x: x

    def compute_l2_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim)
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        l2_error = np.sqrt(np.sum((u - true_u)**2 * self.JxW[:, :, None]))
        return l2_error

    def compute_h1_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = cells_sol[:, None, :, :, None] * self.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim)
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        true_u_grads = jax.vmap(jax.vmap(jax.jacrev(true_u_fn)))(physical_quad_points) # (num_cells, num_quads, vec, dim)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        val_l2_error = np.sqrt(np.sum((u - true_u)**2 * self.JxW[:, :, None]))
        # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1)
        grad_l2_error = np.sqrt(np.sum((u_grads - true_u_grads)**2 * self.JxW[:, :, None, None]))
        h1_error = val_l2_error + grad_l2_error
        return h1_error


class Mechanics(FEM):
    def surface_integral(self, location_fn, surface_fn, sol):
        """Compute surface integral specified by surface_fn: f(u_grad) * ds
        For post-processing only.
        Example usage: compute the total force on a certain surface.

        Parameters
        ----------
        location_fn: callable
            A function that inputs a point (ndarray) and returns if the point satisfies the location condition.
        surface_fn: callable
            A function that inputs a point (ndarray) and returns the value.
        sol: ndarray
            (num_total_nodes, vec)

        Returns
        -------
        int_val: ndarray
            (vec,)
        """
        boundary_inds = self.get_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
        traction = surface_fn(u_grads_face) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
        int_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))
        return int_val

    def compute_traction(self, location_fn, sol):
        """For post-processing only
        """
        stress = self.get_tensor_map()
        vmap_stress = jax.vmap(stress)
        def traction_fn(u_grads):
            """
            Returns
            -------
            traction: ndarray
                (num_selected_faces, num_face_quads, vec)
            """
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
            sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.]).reshape((self.dim, 1))
            traction = (sigmas @ normals[None, None, :, :])[:, :, :, 0]
            return traction

        traction_integral_val = self.surface_integral(location_fn, traction_fn, sol)
        return traction_integral_val

    def compute_surface_area(self, location_fn, sol):
        """For post-processing only
        """
        def unity_fn(u_grads):
            unity = np.ones_like(u_grads)[:, :, :, 0]
            return unity
        unity_integral_val = self.surface_integral(location_fn, unity_fn, sol)
        return unity_integral_val


class LinearElasticity(Mechanics):
    def get_tensor_map(self):
        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress


class HyperElasticity(Mechanics):
    def get_tensor_map(self):
        def psi(F):
            E = 1e3
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


class Plasticity(Mechanics):
    def custom_init(self):
        self.epsilons_old = onp.zeros((len(self.cells), self.num_quads, self.vec, self.dim))
        self.sigmas_old = onp.zeros_like(self.epsilons_old)
        self.internal_vars = {'laplace': [self.sigmas_old, self.epsilons_old]}

    def get_tensor_map(self):
        _, stress_return_map = self.get_maps()
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

        def stress(epsilon):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        def stress_return_map(u_grad, sigma_old, epsilon_old):
            sig0 = 250.
            epsilon_crt = strain(u_grad)
            epsilon_inc = epsilon_crt - epsilon_old
            sigma_trial = stress(epsilon_inc) + sigma_old
            s_dev = sigma_trial - 1./self.dim*np.trace(sigma_trial)*np.eye(self.dim)
            s_norm = safe_sqrt(3./2.*np.sum(s_dev*s_dev))
            f_yield = s_norm - sig0
            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)
            sigma = sigma_trial - safe_divide(f_yield_plus*s_dev, s_norm)
            return sigma

        return strain, stress_return_map

    def stress_strain_fns(self):
        strain, stress_return_map = self.get_maps()
        vmap_strain = jax.vmap(jax.vmap(strain))
        vmap_stress_return_map = jax.vmap(jax.vmap(stress_return_map))
        return vmap_strain, vmap_stress_return_map

    def update_stress_strain(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        vmap_strain, vmap_stress_rm = self.stress_strain_fns()
        self.sigmas_old = vmap_stress_rm(u_grads, self.sigmas_old, self.epsilons_old)
        self.epsilons_old = vmap_strain(u_grads)
        self.internal_vars = {'laplace': [self.sigmas_old, self.epsilons_old]}

    def compute_avg_stress(self):
        """For post-processing only
        """
        # num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1)
        sigma = np.sum(self.sigmas_old.reshape(-1, self.vec, self.dim) * self.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.JxW)
        avg_sigma = sigma/vol
        return avg_sigma


    class Elasticity(FEM):
        def custom_init(self, case_flag):
            self.cell_centroids = onp.mean(onp.take(self.points, self.cells, axis=0), axis=1)
            self.flex_inds = np.arange(len(self.cells))
            self.case_flag = case_flag
            if case_flag == 'freecad':
                self.get_tensor_map = self.get_tensor_map_freecad
            elif case_flag == 'box':
                self.get_tensor_map = self.get_tensor_map_box
            elif case_flag == 'multi_material':
                self.get_tensor_map = self.get_tensor_map_multi_material
            elif case_flag == 'plate' or case_flag == 'L_shape' or case_flag == 'eigen':
                self.get_tensor_map = self.get_tensor_map_plane_stress
                if case_flag == 'eigen':
                    self.penal = 5.
                else:
                    self.penal = 3.
            else:
                raise ValueError(f"Unknown case_flag = {case_flag}")

        def get_tensor_map_plane_stress(self):
            def stress(u_grad, theta):
                # Reference: https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/
                # constitutive-laws/linear-elastic-materials/plane-isotropic-linear-elastic-materials-constitutive-laws/
                Emax = 70.e9
                Emin = 1e-3*Emax
                nu = 0.3

                penal = self.penal

                E = Emin + (Emax - Emin)*theta[0]**penal
                epsilon = 0.5*(u_grad + u_grad.T)

                eps11 = epsilon[0, 0]
                eps22 = epsilon[1, 1]
                eps12 = epsilon[0, 1]

                sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22)
                sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
                sig12 = E/(1 + nu)*eps12

                sigma = np.array([[sig11, sig12], [sig12, sig22]])
                return sigma
            return stress

        def get_tensor_map_freecad(self):
            # Unit is not in SI, used for freecad example
            def stress(u_grad, theta):
                Emax = 70.e3
                Emin = 70.
                nu = 0.3
                penal = 3.
                E = Emin + (Emax - Emin)*theta[0]**penal
                mu = E/(2.*(1. + nu))
                lmbda = E*nu/((1+nu)*(1-2*nu))
                epsilon = 0.5*(u_grad + u_grad.T)
                sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
                return sigma
            return stress

        def get_tensor_map_box(self):
            def stress(u_grad, theta):
                Emax = 70.e9
                Emin = 70.
                nu = 0.3
                penal = 3.
                E = Emin + (Emax - Emin)*theta[0]**penal
                mu = E/(2.*(1. + nu))
                lmbda = E*nu/((1+nu)*(1-2*nu))
                epsilon = 0.5*(u_grad + u_grad.T)
                sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
                return sigma
            return stress

        def get_tensor_map_multi_material(self):
            def stress(u_grad, theta):
                Emax = 70.e3
                Emin = 70.
                nu = 0.3
                penal = 3.

                E1 = Emax
                E2 = 0.2*Emax

                theta1, theta2 = theta
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
            boundary_inds = self.neumann_boundary_inds_list[0]
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
            def stress_fn(u_grad, theta):
                Emax = 70.e9
                nu = 0.3
                penal = 0.5
                E = theta[0]**penal*Emax
                mu = E/(2.*(1. + nu))
                lmbda = E*nu/((1+nu)*(1-2*nu))
                epsilon = 0.5*(u_grad + u_grad.T)
                sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
                return sigma

            def vm_stress_fn_helper(sigma):
                dim = 3
                s_dev = sigma - 1./dim*np.trace(sigma)*np.eye(dim)
                vm_s = np.sqrt(3./2.*np.sum(s_dev*s_dev))
                return vm_s

            if self.case_flag == 'plate' or self.case_flag == 'L_shape':
                def vm_stress_fn(u_grad, theta):
                    sigma2d = stress_fn(u_grad, theta)
                    sigma3d = np.array([[sigma2d[0, 0], sigma2d[0, 1], 0.], [sigma2d[1, 0], sigma2d[1, 1], 0.], [0., 0., 0.]])
                    return vm_stress_fn_helper(sigma3d)
            else:
                def vm_stress_fn(u_grad, theta):
                    sigma = self.get_tensor_map()(u_grad, theta)
                    return vm_stress_fn_helper(sigma)

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