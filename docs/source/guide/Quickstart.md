## Quickstart

You can create an `example.py` to solve the classical Poisson's equation:

```python
import jax
import jax.numpy as np
import os

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x: x

    def get_mass_map(self):
        def mass_map(u, x):
            val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def dirichlet_val(point):
    return 0.

location_fns = [left, right, bottom, top]
value_fns = [dirichlet_val]*4
vecs = [0]*4
dirichlet_bc_info = [location_fns, vecs, value_fns]

problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
sol = solver(problem)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol[0], vtk_path)
```
and run it:

```bash
python example.py
```

The generated result file `u.vtu` can be visualized with [ParaView](https://www.paraview.org/).

<p align="middle">
  <img src="../_static/images/poisson.png" width="400" />
</p>
<p align="middle">
    <em >Solution to the Poisson's equation due to a source term.</em>
</p>

<!-- You can also navigate to `jax_fem/` and run

```bash
python -m tests.benchmarks
```
to execute a set of test cases. -->


<!-- You can also check [`demos/`](https://github.com/deepmodeling/jax-fem/tree/main/demos) for a variety of FEM cases. 

| Example                                                      | Highlight                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [poisson](https://github.com/deepmodeling/jax-fem/tree/main/demos/poisson) | ${\color{green}Basics:}$  Poisson's equation in a unit square domain with Dirichlet and Neumann boundary conditions, as well as a source term. |
| [linear_elasticity](https://github.com/deepmodeling/jax-fem/tree/main/demos/linear_elasticity) | ${\color{green}Basics:}$  Bending of a linear elastic beam due to Dirichlet and Neumann boundary conditions. Second order tetrahedral element (TET10) is used. |
| [hyperelasticity](https://github.com/deepmodeling/jax-fem/tree/main/demos/hyperelasticity) | ${\color{blue}Nonlinear \space Constitutive \space Law:}$ Deformation of a hyperelastic cube due to Dirichlet boundary conditions. |
| [plasticity](https://github.com/deepmodeling/jax-fem/tree/main/demos/plasticity) | ${\color{blue}Nonlinear \space Constitutive \space Law:}$ Perfect J2-plasticity model is implemented for small deformation theory. |
| [phase_field_fracture](https://github.com/deepmodeling/jax-fem/tree/main/demos/phase_field_fracture) | ${\color{orange}Multi-physics \space Coupling:}$ Phase field fracture model is implemented. Staggered scheme is used for two-way coupling of displacement field and damage field. Miehe's model of spectral decomposition is implemented for a 3D case. |
| [thermal_mechanical](https://github.com/deepmodeling/jax-fem/tree/main/demos/thermal_mechanical) | ${\color{orange}Multi-physics \space Coupling:}$ Thermal-mechanical modeling of metal additive manufacturing process. One-way coupling is implemented (temperature affects displacement). |
| [thermal_mechanical_full](https://github.com/deepmodeling/jax-fem/tree/main/demos/thermal_mechanical_full) | ${\color{orange}Multi-physics \space Coupling:}$ Thermal-mechanical modeling of 2D plate. Two-way coupling (temperature and displacement) is implemented with a monolithic scheme. |
| [wave](https://github.com/deepmodeling/jax-fem/tree/main/demos/wave) | ${\color{lightblue}Time \space Dependent \space Problem:}$ The scalar wave equation is solved with backward difference scheme. |
| [topology_optimization](https://github.com/deepmodeling/jax-fem/tree/main/demos/topology_optimization) | ${\color{red}Inverse \space Problem:}$ SIMP topology optimization for a 2D beam. Note that sensitivity analysis is done by the program, rather than manual derivation. |
| [inverse](https://github.com/deepmodeling/jax-fem/tree/main/demos/inverse) | ${\color{red}Inverse \space Problem:}$ Sanity check of how automatic differentiation works. |

For example, run

```bash
python -m demos.hyperelasticity.example
```
for hyperelasticity.  -->

