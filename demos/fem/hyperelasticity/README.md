# Hyperelasticity

## Formulation

The governing equation for linear elasticity of a body $\Omega$ can be written as

$$
\begin{align*}
    -\nabla \cdot \boldsymbol{P}  = \boldsymbol{b} & \quad \textrm{in}  \nobreakspace \nobreakspace \Omega, \\
    \boldsymbol{u} = \boldsymbol{u}_D &  \quad\textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \\
    \boldsymbol{P}  \cdot \boldsymbol{n} = \boldsymbol{t}  & \quad \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}
$$

The weak form gives

$$
\begin{align*}
\int_{\Omega}  \boldsymbol{P} : \nabla \boldsymbol{v} \nobreakspace \nobreakspace \textrm{d}x = \int_{\Omega} \boldsymbol{b}  \cdot \boldsymbol{v} \nobreakspace \textrm{d}x + \int_{\Gamma_N} \boldsymbol{t} \cdot \boldsymbol{v} \nobreakspace\nobreakspace \textrm{d}s.
\end{align*}
$$

Here, $\boldsymbol{P}$ is the first Piola-Kirchhoff stress and is given by

$$
\begin{align*} 
    \boldsymbol{P} &= \frac{\partial W}{\partial \boldsymbol{F}},  \\
    \boldsymbol{F} &= \nabla \boldsymbol{u} + \boldsymbol{I},  \\
    W (\boldsymbol{F}) &= \frac{G}{2}(J^{-2/3} I_1 - 3) + \frac{\kappa}{2}(J - 1)^2,
\end{align*}
$$

where $\boldsymbol{F}$ is the deformation gradient and $W$ is the strain energy density function. This constitutive relationship comes from a neo-Hookean solid model [2].


We have the following definitions:
* $\Omega=(0,1)\times(0,1)\times(0,1)$ (a unit cube)
* $\Gamma_{D_1}=0\times(0,1)\times(0,1)$ (first part of Dirichlet boundary)
* $ \boldsymbol{u}_{D_1}= [0,(0.5+(x_2−0.5)\textrm{cos}(\pi/3)−(x_3−0.5)\textrm{sin}(\pi/3)−x_2)/2, (0.5+(x_2−0.5)\textrm{sin}(\pi/3)+(x_3−0.5)\textrm{cos}(\pi/3)−x_3))/2)] $
* $\Gamma_{D_2}=1\times(0,1)\times(0,1)$ (second part of Dirichlet boundary)
* $\boldsymbol{u}_{D_2}=[0,0,0]$ 
* $b=[0, 0, 0]$
* $t=[0, 0, 0]$


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

Define constitutive relationship. The `get_tensor_map` function overrides base class method. Generally, *JAX-FEM* solves $-\nabla \cdot \boldsymbol{f}(\nabla \boldsymbol{u}) = \boldsymbol{b}$. Here, we define $\boldsymbol{f}(\nabla \boldsymbol{u})=\boldsymbol{P}$. Notice how we first define `psi` (representing $W$), and then use automatic differentiation (`jax.grad`) to obtain the `P_fn` function.
```python
class HyperElasticity(FEM):
    def get_tensor_map(self):
        def psi(F):
            E = 10.
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
```

Specify mesh-related information. We use first-order hexahedron element.
```python
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh(Nx=20, Ny=20, Nz=20, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
```

Define boundary locations:
```python
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)
```

Define Dirichlet boundary values:
```python
def zero_dirichlet_val(point):
    return 0.

def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5)*np.cos(np.pi/3.) - (point[2] - 0.5)*np.sin(np.pi/3.) - point[1])/2.

def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5)*np.sin(np.pi/3.) + (point[2] - 0.5)*np.cos(np.pi/3.) - point[2])/2.

dirichlet_bc_info = [[left]*3 + [right]*3, 
                     [0, 1, 2]*2, 
                     [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] + [zero_dirichlet_val]*3]
```

Create an instance of the problem, solve it, and store the solution to local file.
```python
problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
sol = solver(problem, use_petsc=True)
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem, sol, vtk_path)
```

## Execution
Run
```bash
python -m demos.fem.hyperelasticity.example
```
from the `jax-am/` directory.


## Results

Visualized with *ParaWiew* "Warp By Vector" function:

<p align="middle">
  <img src="materials/sol.png" width="500" />
</p>
<p align="middle">
    <em >Solution</em>
</p>

## References

[1] https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/hyperelasticity/python/documentation.html

[2] https://en.wikipedia.org/wiki/Neo-Hookean_solid