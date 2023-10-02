<p align="middle">
  <img src="docs/materials/logo.png" width="200" />
</p>

A GPU-accelerated differentiable simulation toolbox for additive manufacturing (AM) based on [JAX](https://github.com/google/jax). 

# JAX-AM

[![Doc](https://img.shields.io/readthedocs/jax-am)](https://jax-am.readthedocs.io/en/latest/)
![PyPI](https://img.shields.io/pypi/v/jax-am)
![Github Star](https://img.shields.io/github/stars/tianjuxue/jax-am)
![Github Fork](https://img.shields.io/github/forks/tianjuxue/jax-am)
![License](https://img.shields.io/github/license/tianjuxue/jax-am)

JAX-AM is a collection of several numerical tools, currently including __Discrete Element Method (DEM)__, __Lattice Boltzmann Methods (LBM)__, __Computational Fluid Dynamics (CFD)__, __Phase Field Method (PFM)__ and __Finite Element Method (FEM)__, that cover the analysis of the __Process-Structure-Property__ relationship in AM. 

Our vision is to share with the AM community a __free, open-source__ (under the GPL-3.0 License) software that facilitates the relevant computational research. In the JAX ecosystem, we hope to emphasize the potential of JAX for scientific computing. At the same time, AI-enabled research in AM can be made easy with JAX-AM. 

<!-- Authors:
* [Tianju Xue](https://tianjuxue.github.io/)
* [Shuheng Liao](https://scholar.google.com/citations?user=uxIh-nYAAAAJ&hl=en)
* [Zhengtao Gan](https://zgan.my.canva.site/) -->

:fire: <u>***Join us for the development of JAX-AM!***</u> :rocket:

## Discrete Element Method (DEM)

DEM simulation can be used for simulating powder dynamics in metal AM.

<p align="middle">
  <img src="docs/materials/dem.gif" width="400" />
</p>
<p align="middle">
    <em >Free falling of 64,000 spherical particles.</em>
</p>

## Lattice Boltzmann Methods (LBM)

LBM can simulate melt pool dynamics with a free-surface model.


<p align="middle">
  <img src="docs/materials/lbm_pbf.gif" width="700" />
</p>
<p align="middle">
    <em >A powder bed fusion process considering surface tension, Marangoni effect and recoil pressure.</em>
</p>


## Computational Fluid Dynamics (CFD)

CFD helps to understand the AM process by solving the (incompressible) Navier-Stokes equations for velocity, pressure and temperature.

<p align="middle">
  <img src="docs/materials/melt_pool_dynamics.png" width="400" />
</p>
<p align="middle">
    <em >Melt pool dynamics.</em>
</p>


## Phase Field Method (PFM)

PFM models the grain development that is critical to form the structure of the as-built sample.

<p align="middle">
  <img src="docs/materials/single_track_eta_DNS.gif" width="600" />
</p>
<p align="middle">
    <em >Microstructure evolution.</em>
</p>

<p align="middle">
  <img src="docs/materials/solidification_isotropic.gif" width="300" />
  <img src="docs/materials/solidification_anisotropic.gif" width="300" /> 
</p>
<p align="middle">
    <em >Directional solidification with isotropic (left) and anisotropic (right) grain growth.</em>
</p>


## Finite Element Method (FEM)

FEM is a powerful tool for thermal-mechanical analysis in AM. We support the following features

- 2D quadrilateral/triangle elements
- 3D hexahedron/tetrahedron elements
- First and second order elements
- Dirichlet/Neumann/Cauchy/periodic boundary conditions
- Linear and nonlinear analysis including
  - Heat equation
  - Linear elasticity
  - Hyperelasticity
  - Plasticity (macro and crystal plasticity)
- Differentiable simulation for solving inverse/design problems __without__ human deriving sensitivities, e.g.,
  - Toplogy optimization
  - Optimal thermal control


<p align="middle">
  <img src="docs/materials/ded.gif" width="600" />
</p>
<p align="middle">
    <em >Thermal profile in direct energy deposition.</em>
</p>

<p align="middle">
  <img src="docs/materials/von_mises.png" width="400" />
</p>
<p align="middle">
    <em >Linear static analysis of a bracket.</em>
</p>

<p align="middle">
  <img src="docs/materials/polycrystal_grain.gif" width="350" />
  <img src="docs/materials/polycrystal_stress.gif" width="350" /> 
</p>
<p align="middle">
    <em >Crystal plasticity: grain structure (left) and stress-xx (right).</em>
</p>

<p align="middle">
  <img src="docs/materials/to.gif" width="600" />
</p>
<p align="middle">
    <em >Topology optimization with differentiable simulation.</em>
</p>

## Documentation

Please see the [web documentation](https://jax-am.readthedocs.io/en/latest/) for the installation and use of this project.


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

```bibtex
@article{xue2022physics,
  title={Physics-embedded graph network for accelerating phase-field simulation of microstructure evolution in additive manufacturing},
  author={Xue, Tianju and Gan, Zhengtao and Liao, Shuheng and Cao, Jian},
  journal={npj Computational Materials},
  volume={8},
  number={1},
  pages={201},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```