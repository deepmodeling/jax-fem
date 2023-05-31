# Linear Elasticity

## Formulation

The governing equation for linear elasticity of a body $\Omega$ can be written as

$$
\begin{align*}
    -\nabla \cdot \boldsymbol{\sigma}  = \boldsymbol{b} & \quad \textrm{in}  \nobreakspace \nobreakspace \Omega, \\
    \boldsymbol{u} = \boldsymbol{u}_D &  \quad\textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \\
    \boldsymbol{\sigma}  \cdot \boldsymbol{n} = \boldsymbol{t}  & \quad \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}
$$

The weak form gives

$$
\begin{align*}
\int_{\Omega}  \boldsymbol{\sigma} : \nabla \boldsymbol{v} \nobreakspace \nobreakspace \textrm{d}x = \int_{\Omega} \boldsymbol{b}  \cdot \boldsymbol{v} \nobreakspace \textrm{d}x + \int_{\Gamma_N} \boldsymbol{t} \cdot \boldsymbol{v} \nobreakspace\nobreakspace \textrm{d}s.
\end{align*}
$$

In this example, we consider a vertical bending load applied to the right side of the beam ($\boldsymbol{t}=[0, 0, -100]$) while fixing the left side ($\boldsymbol{u}_D=[0,0,0]$), and ignore body force ($\boldsymbol{b}=[0,0,0]$).

The constitutive relationship is given by


$$
\begin{align*}
     \boldsymbol{\sigma} &=  \lambda \nobreakspace \textrm{tr}(\boldsymbol{\varepsilon}) \boldsymbol{I} + 2\mu \nobreakspace \boldsymbol{\varepsilon}, \\
    \boldsymbol{\varepsilon} &= \frac{1}{2}\left[\nabla\boldsymbol{u} + (\nabla\boldsymbol{u})^{\top}\right].
\end{align*}
$$

## Implementation

Import some useful modules
```python
import jax
import jax.numpy as np
import os

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
```

Define constitutive relationship. The `get_tensor_map` function overrides base class method. Generally, *JAX-FEM* solves $-\nabla \cdot \boldsymbol{f}(\nabla \boldsymbol{u}) = \boldsymbol{b}$. Here, we have $\boldsymbol{f}(\nabla \boldsymbol{u})=\boldsymbol{\sigma}$.
```python
class LinearElasticity(FEM):
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
```

Specify mesh-related information. We use second-order tetrahedron element.
```python
ele_type = 'TET10'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 10., 2., 2.
meshio_mesh = box_mesh(Nx=25, Ny=5, Nz=5, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
```

Define boundary locations:
```python
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)
```

Define Dirichlet boundary values. This means on the `left` side, we apply the function `zero_dirichlet_val` to all components of the displacement variable $\boldsymbol{u}$. 
```python
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[left]*3, 
                     [0, 1, 2], 
                     [zero_dirichlet_val]*3]
```

Define Neumann boundary value $\boldsymbol{t}$.
```python
def neumann_val(point):
    return np.array([0., 0., -100.])

neumann_bc_info = [[right], [neumann_val]]
```

Create an instance of the problem, solve it, and store the solution to local file.
```python
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
    neumann_bc_info=neumann_bc_info)
sol = solver(problem, linear=True, use_petsc=True)
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem, sol, vtk_path)
```

## Execution
Run
```bash
python -m demos.fem.linear_elasticity.example
```
from the `jax-am/` directory.


## Results

Visualized with *ParaWiew*:

<p align="middle">
  <img src="materials/sol.png" width="500" />
</p>
<p align="middle">
    <em >Solution</em>
</p>