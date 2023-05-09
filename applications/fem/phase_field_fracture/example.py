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


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')



class PF(FEM):
    G_c = 100.
    def get_tensor_map(self):
        def fn(u_grad):
            k = 15.
            return k*u_grad
        return fn
 
    def get_mass_map(self):
        def T_map(T):
            Cp = 500.
            rho = 8440.
            return rho*Cp*T/dt
        return T_map

    def get_body_map(self):
        return self.get_mass_map()

    def set_params(self, old_sol):
        self.internal_vars['body_vars'] = old_sol
        self.internal_vars['neumann_vars'] = old_sol

