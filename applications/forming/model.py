import jax
import jax.numpy as np

from jax_fem.problem import Problem


class Plasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.F_old = np.repeat(np.repeat(np.eye(self.dim)[None, None, :, :], len(self.fe.cells), axis=0), self.fe.num_quads, axis=1)
        self.Be_old = np.array(self.F_old)
        self.alpha_old = np.zeros((len(self.fe.cells), self.fe.num_quads))
        self.internal_vars = [self.F_old, self.Be_old, self.alpha_old]

    def get_tensor_map(self):
        tensor_map, _, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        K = 164.e3
        G = 80.e3
        H1 = 18.
        sig0 = 400. 

        def get_partial_tensor_map(F_old, be_bar_old, alpha_old):
            def first_PK_stress(u_grad):
                _, _, tau = return_map(u_grad)
                F = u_grad + np.eye(self.dim)
                P = tau @ np.linalg.inv(F).T 
                return P    

            def update_int_vars(u_grad):
                be_bar, alpha, _ = return_map(u_grad)
                F = u_grad + np.eye(self.dim)
                return F, be_bar, alpha

            def compute_cauchy_stress(u_grad):
                F = u_grad + np.eye(self.dim)
                J = np.linalg.det(F)
                P = first_PK_stress(u_grad)
                sigma = 1./J*P @ F.T
                return sigma

            def get_tau(F, be_bar):
                J = np.linalg.det(F)
                tau = 0.5*K*(J**2 - 1)*np.eye(self.dim) + G*deviatoric(be_bar)
                return tau

            def deviatoric(A):
                return A - 1./self.dim*np.trace(A)*np.eye(self.dim)

            def return_map(u_grad):
                F = u_grad + np.eye(self.dim)
                F_inv = np.linalg.inv(F)
                F_old_inv = np.linalg.inv(F_old)
                f = F @ F_old_inv
                f_bar =  np.linalg.det(f)**(-1./3.)*f
                # be_bar_trial = f @ be_bar_old @ f.T # Seems that there is a bug here, discovered by Jiachen; should be f_bar @ be_bar_old @ f_bar.T 
                be_bar_trial = f_bar @ be_bar_old @ f_bar.T
                s_trial = G*deviatoric(be_bar_trial)
                yield_f_trial = np.linalg.norm(s_trial) - np.sqrt(2./3.)*(sig0 + H1*alpha_old)

                def elastic_loading():
                    be_bar = be_bar_trial
                    alpha = alpha_old
                    tau = get_tau(F, be_bar)
                    return be_bar, alpha, tau

                def plastic_loading():
                    Ie_bar = 1./3.*np.trace(be_bar_trial)
                    G_bar = Ie_bar*G
                    Delta_gamma = (yield_f_trial/(2.*G_bar))/(1. + H1/(3.*G_bar))
                    direction = s_trial/np.linalg.norm(s_trial)
                    s = s_trial - 2.*G_bar*Delta_gamma * direction
                    alpha = alpha_old + np.sqrt(2./3.)*Delta_gamma
                    be_bar = s/G + Ie_bar*np.eye(self.dim)
                    tau = get_tau(F, be_bar)
                    return be_bar, alpha, tau

                return jax.lax.cond(yield_f_trial < 0., elastic_loading, plastic_loading)

            return first_PK_stress, update_int_vars, compute_cauchy_stress

        def tensor_map(u_grad, F_old, Be_old, alpha_old):
            first_PK_stress, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old)
            return first_PK_stress(u_grad)

        def update_int_vars_map(u_grad, F_old, Be_old, alpha_old):
            _, update_int_vars, _ = get_partial_tensor_map(F_old, Be_old, alpha_old)
            return update_int_vars(u_grad)

        def compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old):
            _, _, compute_cauchy_stress = get_partial_tensor_map(F_old, Be_old, alpha_old)
            return compute_cauchy_stress(u_grad)

        return tensor_map, update_int_vars_map, compute_cauchy_stress_map

    def update_int_vars_gp(self, sol, int_vars):
        _, update_int_vars_map, _ = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, self.dim) -> (num_cells, num_quads, num_nodes, vec, self.dim) 
        u_grads = np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] * self.fe.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, self.dim)
        updated_int_vars = vmap_update_int_vars_map(u_grads, *int_vars)
        return updated_int_vars

    def compute_stress(self, sol, int_vars):
        _, _, compute_cauchy_stress = self.get_maps()
        vmap_compute_cauchy_stress = jax.jit(jax.vmap(jax.vmap(compute_cauchy_stress)))
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, self.dim) -> (num_cells, num_quads, num_nodes, vec, self.dim) 
        u_grads = np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] * self.fe.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, self.dim)
        sigma = vmap_compute_cauchy_stress(u_grads, *int_vars)
        return sigma
