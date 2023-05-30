# Poisson's equation

### Formulation

The Poisson's equation is the canonical elliptic partial differential equation. Consider a domain $\Omega \subset \mathbb{R}^\textrm{d}$ with boundary $\partial \Omega = \Gamma_D \cup \Gamma_N$, the strong form gives

$$
\begin{align}
    -\nabla^2 u = b & \quad \textrm{in}  \, \, \Omega, \\
    u = 0 &  \quad\textrm{on} \, \, \Gamma_D,  \\
    \nabla u  \cdot \boldsymbol{n} = t  & \quad \textrm{on} \, \, \Gamma_N.
\end{align}
$$

The weak form gives

$$
\begin{aligned}
\int_{\Omega} \nabla u \cdot \nabla v \, \, \textrm{d}x = \int_{\Omega} b \, v \, \textrm{d}x + \int_{\Gamma_N} t\, v \,\, \textrm{d}s.
\end{aligned}
$$

We have the following definitions:
* $\Omega=[0,1]\times[0,1]$ (a unit square)
* $\Gamma_D=\{(0, x_2)\cup (1, x_2)\subset\partial\Omega\}$ (Dirichlet boundary)
* $\Gamma_N=\{(x_1, 0)\cup (x_1, 1)\subset\partial\Omega\}$ (Neumann boundary)
* $b=10\,\textrm{exp}\big(-((x_1-0.5)^2+(x_2-0.5)^2)/0.02 \big)$
* $t=\textrm{sin}(5x_1)$

### Implementation

Import some generally useful packages:
```python
import jax
import jax.numpy as np
import os
```

Import *JAX-FEM* specific modules:
```python
from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh
```

Define constitutive relationship. The `get_tensor_map` function overrides base class method. *JAX-FEM* generally solves $-\nabla \cdot f(\nabla u) = b$. Here, we define $f$ to be the identity function. We will see how $f$ is defined as more complicated to solve non-linear problems in later examples.
```python
class Poisson(FEM):
    def get_tensor_map(self):
        return lambda x: x
```

Specify mesh-related information. We make use of the external package `meshio` and create a mesh named `meshio_mesh`, then converting it into a *JAX-FEM* compatible one.
```python
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
```

Define boundary locations:
```python
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)
```


Define Dirichlet boundary values. This means on the `left` side, we apply the function `dirichlet_val_left` to the `0` component of the solution variable $u$; on the `right` side, we apply `dirichlet_val_right` to the `0` component.
```python
def dirichlet_val_left(point):
    return 0.

def dirichlet_val_right(point):
    return 0.

location_fns = [left, right]
value_fns = [dirichlet_val_left, dirichlet_val_right]
vecs = [0, 0]
dirichlet_bc_info = [location_fns, vecs, value_fns]
```

Define Neumann boundary value $t$. Note that Neumann values are not imposed component-wisely as what we did in Dirichlet values. Rather, Neumann values are imposed with a vector value. This is why `neumann_val` returns a shape `(1,)` array, rather than a scalar value.
```python
def neumann_val(point):
    return np.array([np.sin(5.*point[0])])

neumann_bc_info = [[bottom, top], [neumann_val, neumann_val]]
```

Define the source term $b$:
```python
def body_force(point):
    return np.array([10*np.exp(-(np.power(point[0] - 0.5, 2) + np.power(point[1] - 0.5, 2)) / 0.02)])
```

Create an instance of the `Poisson` class. Here, `vec` is the number of components for the solution $u$. 
```python
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
    neumann_bc_info=neumann_bc_info, source_info=body_force)
```

Solve the problem. Setting the flag `linear` is optional, but gives a slightly better performance. If the program runs on CPU, we suggest setting `use_petsc` to be `True` to use *PETSc* solver; if GPU is available, we suggest setting `use_petsc` to be `False` to call the *JAX* built-in solver that can often times be faster.
```python
sol = solver(problem, linear=True, use_petsc=True)
```

Save the solution to a local folder that can be visualized with *ParaWiew*. 
```python
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem, sol, vtk_path)
```

### Execution
Run
```bash
python -m demos.fem.poisson.example
```
from the `jax-am/` directory.


### Results

<p align="middle">
  <img src="materials/sol.png" width="500" />
</p>
<p align="middle">
    <em >Solution</em>
</p>


### References

[1] https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/poisson/python/documentation.html

[2] Xue, Tianju, et al. "JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science." arXiv preprint arXiv:2212.00964 (2022).