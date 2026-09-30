"""AT2 phase field and Miehe's spectral split for the fracture example."""
import jax.numpy as np

from jax_fem.problem import Problem
from applications.phase_field_fracture.eigen import get_eigen_f_custom


positive = lambda x: 0.5*(x + np.abs(x))
negative = lambda x: 0.5*(x - np.abs(x))
positive_strain = get_eigen_f_custom(positive)
negative_strain = get_eigen_f_custom(negative)


def get_elasticity_maps(lmbda, mu):
    # Native JAX AD may return NaN in the cases with repeated eigenvalues.
    # We define custom derivative rules to properly handle repeated eigenvalues.
    def psi_plus(u_grad):
        epsilon = 0.5*(u_grad + u_grad.T)
        eigen_vals = np.linalg.eigvalsh(epsilon)
        return lmbda/2.*positive(np.trace(epsilon))**2 + mu*np.sum(positive(eigen_vals)**2)

    def stress(u_grad, d):
        epsilon = 0.5*(u_grad + u_grad.T)
        identity = np.eye(u_grad.shape[0])
        sigma_plus = lmbda*positive(np.trace(epsilon))*identity + 2.*mu*positive_strain(epsilon)
        sigma_minus = lmbda*negative(np.trace(epsilon))*identity + 2.*mu*negative_strain(epsilon)
        # Keep damage outside the custom spectral derivative so AD can trace it.
        return (1. - d[0])**2*sigma_plus + sigma_minus

    return psi_plus, stress


class PhaseField(Problem):
    # Note how 'get_tensor_map' and 'get_mass_map' specify the corresponding terms
    # in the weak form. Since the displacement variable u affects the phase field
    # variable d through the history variable H, we need to set this using 'set_params'.
    def custom_init(self, G_c, length_scale):
        self.G_c = G_c
        self.length_scale = length_scale

    def get_tensor_map(self):
        def fn(d_grad, history):
            return self.G_c*self.length_scale*d_grad
        return fn

    def get_mass_map(self):
        def fn(d, x, history):
            return self.G_c/self.length_scale*d - 2.*(1. - d)*history
        return fn

    def set_params(self, history):
        self.internal_vars = [history]
