import numpy as onp
import jax
import jax.numpy as np

from jax_fem.core import FEM


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
        """TODO: Move this to jax-fem library?
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        vm_stress_fn = self.get_von_mises_stress_fn()
        vm_stress = jax.vmap(jax.vmap(vm_stress_fn))(u_grads, *self.internal_vars['laplace']) # (num_cells, num_quads)
        volume_avg_vm_stress = np.sum(vm_stress * self.JxW, axis=1) / np.sum(self.JxW, axis=1) # (num_cells,)
        return volume_avg_vm_stress
