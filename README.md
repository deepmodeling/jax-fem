A GPU-accelerated differentiable finite element analysis package based on [JAX](https://github.com/google/jax). Used to be part of the suite of open-source python packages for Additive Manufacturing (AM) research, [JAX-AM](https://github.com/tianjuxue/jax-am).

## Finite Element Method (FEM)
![Github Star](https://img.shields.io/github/stars/deepmodeling/jax-fem)
![Github Fork](https://img.shields.io/github/forks/deepmodeling/jax-fem)
![License](https://img.shields.io/github/license/deepmodeling/jax-fem)

FEM is a powerful tool, where we support the following features

- 2D quadrilateral/triangle elements
- 3D hexahedron/tetrahedron elements
- First and second order elements
- Dirichlet/Neumann/Robin boundary conditions
- Linear and nonlinear analysis including
  - Heat equation
  - Linear elasticity
  - Hyperelasticity
  - Plasticity (macro and crystal plasticity)
- Differentiable simulation for solving inverse/design problems __without__ human deriving sensitivities, e.g.,
  - Topology optimization
  - Optimal thermal control
- Integration with PETSc for solver choices

**Updates** (Dec 11, 2023):

- We now support multi-physics problems in the sense that multiple variables can be solved monolithically. For example, consider running  `python -m applications.stokes.example`
- Weak form is now defined through  volume integral and surface integral. We can now treat body force, "mass kernel" and "Laplace kernel" in a unified way through volume integral, and treat "Neumann B.C." and "Robin B.C." in a unified way through surface integral. 

<p align="middle">
  <img src="images/ded.gif" width="600" />
</p>
<p align="middle">
    <em >Thermal profile in direct energy deposition.</em>
</p>

<p align="middle">
  <img src="images/von_mises.png" width="400" />
</p>
<p align="middle">
    <em >Linear static analysis of a bracket.</em>
</p>

<p align="middle">
  <img src="images/polycrystal_grain.gif" width="360" />
  <img src="images/polycrystal_stress.gif" width="360" />
</p>
<p align="middle">
    <em >Crystal plasticity: grain structure (left) and stress-xx (right).</em>
</p>

<p align="middle">
  <img src="images/stokes_u.png" width="360" />
  <img src="images/stokes_p.png" width="360" />
</p>
<p align="middle">
    <em >Stokes flow: velocity (left) and pressure(right).</em>
</p>

<p align="middle">
  <img src="images/to.gif" width="600" />
</p>
<p align="middle">
    <em >Topology optimization with differentiable simulation.</em>
</p>

## Installation

Create a conda environment from the given [`environment.yml`](https://github.com/deepmodeling/jax-fem/blob/main/environment.yml) file and activate it:

```bash
conda env create -f environment.yml
conda activate jax-fem-env
```

Install JAX
- See jax installation [instructions](https://jax.readtheimages.io/en/latest/installation.html#). Depending on your hardware, you may install the CPU or GPU version of JAX. Both will work, while GPU version usually gives better performance.


Then there are two options to continue:

### Option 1

Clone the repository:

```bash
git clone https://github.com/deepmodeling/jax-fem.git
cd jax-fem
```

and install the package locally:

```bash

pip install -e .
```

**Quick tests**: You can check `demos/` for a variety of FEM cases. For example, run

```bash
python -m demos.hyperelasticity.example
```

for hyperelasticity. 

Also, 

```bash
python -m tests.benchmarks
```

will execute a set of test cases.


### Option 2

Install the package from the [PyPI release](https://pypi.org/project/jax-fem/) directly:

```bash
pip install jax-fem
```

**Quick tests**: You can create an `example.py` file and run it:

```bash
python example.py
```

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
sol = solver(problem, linear=True, use_petsc=True)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol[0], vtk_path)
```

By running the code above and use [Paraview](https://www.paraview.org/) for visualization, you should see the following solution

<p align="middle">
  <img src="images/poisson.png" width="400" />
</p>
<p align="middle">
    <em >Solution to the Poisson's equation due to a source term.</em>
</p>


## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citations

If you found this library useful in academic or industry work, we appreciate your support if you consider 1) starring the project on Github, and 2) citing relevant papers:

```bibtex
@article{xue2023jax,
  title={JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science},
  author={Xue, Tianju and Liao, Shuheng and Gan, Zhengtao and Park, Chanwook and Xie, Xiaoyu and Liu, Wing Kam and Cao, Jian},
  journal={Computer Physics Communications},
  pages={108802},
  year={2023},
  publisher={Elsevier}
}
```

