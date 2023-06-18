import jax
import jax.numpy as np
import os

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh




safe_maximum = lambda x: 0.5*(x + np.abs(x))
safe_minimum = lambda x: 0.5*(x - np.abs(x))


class LinearElasticity(FEM):
    # def get_tensor_map(self):
    #     def stress(u_grad):
    #         E = 70e3
    #         nu = 0.3
    #         mu = E/(2.*(1. + nu))
    #         lmbda = E*nu/((1+nu)*(1-2*nu))
    #         epsilon = 0.5*(u_grad + u_grad.T)
    #         sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
    #         return sigma
    #     return stress


    def get_tensor_map(self):
        E = 70e3
        nu = 0.3
        mu = E/(2.*(1. + nu))
        lmbda = E*nu/((1+nu)*(1-2*nu))

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def psi_plus_C_part(epsilon):
            tr_epsilon_plus = safe_maximum(np.trace(epsilon))
            return lmbda/2.*tr_epsilon_plus**2
            
        def psi_minus_C_part(epsilon):
            tr_epsilon_minus = safe_minimum(np.trace(epsilon))
            return lmbda/2.*tr_epsilon_minus**2
            

        def psi_pm(epsilon):
            tr_epsilon_minus = safe_maximum(np.trace(epsilon))
            tr_epsilon_plus = safe_minimum(np.trace(epsilon))
            tr_epsilon = tr_epsilon_minus + tr_epsilon_plus
            return lmbda/2.*tr_epsilon**2


        def stress_fn(u_grad):
            epsilon = strain(u_grad)

            # def f(x):
            #     return 2*mu*(safe_maximum(x) + safe_minimum(x))


            def f(x):
                return 2*mu*(x)


            grad_f = jax.grad(f)
            f_vmap = jax.vmap(f)
            grad_f_vmap = jax.vmap(grad_f)

            @jax.custom_jvp
            def eigen_f(x):
                evals, evecs = np.linalg.eigh(x)
 
                # evals = np.real(evals)
                # evecs = np.real(evecs)

                evecs = evecs.T
                M = np.einsum('bi,bj->bij', evecs, evecs)
                # [batch, dim, dim] * [batch, 1, 1] -> [dim, dim]
                result = np.sum(M * f_vmap(evals)[:, None, None], axis=0)
                return result

            @eigen_f.defjvp
            def f_jvp(primals, tangents):
                x, = primals
                v, = tangents

                evals, evecs = np.linalg.eigh(x)

                # evals = np.real(evals)
                # evecs = np.real(evecs)

                fvals = f_vmap(evals)
                grads = grad_f_vmap(evals)
                evecs = evecs.T

                M = np.einsum('bi,bj->bij', evecs, evecs)

                result = np.sum(M * fvals[:, None, None], axis=0)

                MM = np.einsum('bij,bkl->bijkl', M, M)
                # [batch, dim, dim, dim, dim] * [batch, 1, 1, 1, 1] -> [dim, dim, dim, dim]
                term1 = np.sum(MM * grads[:, None, None, None, None], axis=0)

                G = np.einsum('aik,bjl->abijkl', M, M) + np.einsum('ail,bjk->abijkl', M, M)

                diff_evals = evals[:, None] - evals[None, :]
                diff_fvals = fvals[:, None] - fvals[None, :]
                diff_grads = grads[:, None]

                theta = np.where(diff_evals == 0., diff_grads, diff_fvals/diff_evals)

                tmp = G * theta[:, :, None, None, None, None]
                tmp1 = np.sum(tmp, axis=(0, 1))
                tmp2 = np.einsum('aa...->...', tmp)
                term2 = 0.5*(tmp1 - tmp2)

                P = term1 + term2
                jvp_result = np.einsum('ijkl,kl', P, v)

                return result, jvp_result


            # epsilon += np.array([[0.0000001, 0.0000002, 0.0000003], 
            #                      [0.0000002, 0.0000004, 0.0000005], 
            #                      [0.0000003, 0.0000005, 0.0000006]])


            # sigma1 = lmbda*np.trace(epsilon)*np.eye(self.dim) 
            sigma1 = jax.grad(psi_plus_C_part)(epsilon) + jax.grad(psi_minus_C_part)(epsilon) 
            # sigma1 = jax.grad(psi_pm)(epsilon)

            sigma2 = eigen_f(epsilon)
            # sigma2 = 2*mu*epsilon

            sigma = sigma1 + sigma2


            return sigma

        return stress_fn


ele_type = 'TET10'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 10., 2., 2.
meshio_mesh = box_mesh(Nx=25, Ny=5, Nz=5, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[left]*3, 
                     [0, 1, 2], 
                     [zero_dirichlet_val]*3]

def neumann_val(point):
    return np.array([0., 0., -100.])

neumann_bc_info = [[right], [neumann_val]]

problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
    neumann_bc_info=neumann_bc_info)
sol = solver(problem, linear=True, use_petsc=True)
 
