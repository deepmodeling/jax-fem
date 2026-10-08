import jax
import jax.numpy as np

from jax_fem.problem import Problem
from applications.phase_field_fracture.eigen import get_eigen_f_custom, get_eigen_f_jax


positive = lambda x: 0.5*(x + np.abs(x))
negative = lambda x: 0.5*(x - np.abs(x))
positive_strain_custom = get_eigen_f_custom(positive)
negative_strain_custom = get_eigen_f_custom(negative)
positive_strain_jax = get_eigen_f_jax(positive)
negative_strain_jax = get_eigen_f_jax(negative)


def get_elasticity_maps(lmbda, mu, spectral_method='custom', noise_scale=1e-8):
    # Native JAX AD may return NaN in cases with repeated eigenvalues.
    # Users can choose between custom derivative rules and adding a small noise
    # to the strain tensor.
    def strain(u_grad):
        return 0.5*(u_grad + u_grad.T)

    def psi_plus(u_grad):
        epsilon = strain(u_grad)
        eigen_vals = np.linalg.eigvalsh(epsilon)
        return lmbda/2.*positive(np.trace(epsilon))**2 + mu*np.sum(positive(eigen_vals)**2)

    def stress_from_strain(epsilon, epsilon_plus, epsilon_minus, d):
        identity = np.eye(epsilon.shape[0])
        sigma_plus = lmbda*positive(np.trace(epsilon))*identity + 2.*mu*epsilon_plus
        sigma_minus = lmbda*negative(np.trace(epsilon))*identity + 2.*mu*epsilon_minus
        # Keep damage outside the spectral maps so AD can trace it.
        return (1. - d[0])**2*sigma_plus + sigma_minus

    if spectral_method == 'custom':
        def stress(u_grad, d):
            epsilon = strain(u_grad)
            epsilon_plus = positive_strain_custom(epsilon)
            epsilon_minus = negative_strain_custom(epsilon)
            return stress_from_strain(epsilon, epsilon_plus, epsilon_minus, d)
    elif spectral_method == 'noise':
        key = jax.random.PRNGKey(0)
        def stress(u_grad, d):
            epsilon = strain(u_grad)
            noise = jax.random.uniform(key, shape=epsilon.shape,
                                       minval=-noise_scale, maxval=noise_scale)
            noise = np.diag(np.diag(noise))
            epsilon = epsilon + noise
            epsilon_plus = positive_strain_jax(epsilon)
            epsilon_minus = negative_strain_jax(epsilon)
            return stress_from_strain(epsilon, epsilon_plus, epsilon_minus, d)
    else:
        raise ValueError(f"Unknown spectral method '{spectral_method}'. "
                         "Expected 'custom' or 'noise'.")

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
