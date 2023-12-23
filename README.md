A GPU-accelerated differentiable finite element analysis package based on [JAX](https://github.com/google/jax). Used to be part of the suite of open-source python packages for Additive Manufacturing (AM) research, [JAX-AM](https://github.com/tianjuxue/jax-am).

## Finite Element Method (FEM)
![Github Star](https://img.shields.io/github/stars/tianjuxue/jax-fem)
![Github Fork](https://img.shields.io/github/forks/tianjuxue/jax-fem)
![License](https://img.shields.io/github/license/tianjuxue/jax-fem)

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
  <img src="docs/ded.gif" width="600" />
</p>
<p align="middle">
    <em >Thermal profile in direct energy deposition.</em>
</p>

<p align="middle">
  <img src="docs/von_mises.png" width="400" />
</p>
<p align="middle">
    <em >Linear static analysis of a bracket.</em>
</p>

<p align="middle">
  <img src="docs/polycrystal_grain.gif" width="360" />
  <img src="docs/polycrystal_stress.gif" width="360" />
</p>
<p align="middle">
    <em >Crystal plasticity: grain structure (left) and stress-xx (right).</em>
</p>

<p align="middle">
  <img src="docs/stokes_u.png" width="360" />
  <img src="docs/stokes_p.png" width="360" />
</p>
<p align="middle">
    <em >Stokes flow: velocity (left) and pressure(right).</em>
</p>

<p align="middle">
  <img src="docs/to.gif" width="600" />
</p>
<p align="middle">
    <em >Topology optimization with differentiable simulation.</em>
</p>

## Installation

Clone the repository:

```bash
git clone git@github.com:tianjuxue/jax-fem.git
cd jax-fem
```

Create a conda environment from the given `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment and install the package:

```bash
conda activate jax-fem-env
pip install -e .
```

### Dependencies
Install JAX
- See jax installation [instructions](https://jax.readthedocs.io/en/latest/installation.html#)


## Quick start

Check `demos/` for a variety of FEM cases. For example, run

```bash
python -m demos.hyperelasticity.example
```

for hyperelasticity. 

Also, 

```bash
python -m tests.benchmarks
```

will execute a set of test cases.


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
