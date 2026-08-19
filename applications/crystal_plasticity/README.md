# Crystal Plasticity

## Overview

This directory contains a finite-strain FCC crystal-plasticity model and three
small applications built around it: two single-element sensitivity tests and a
100-grain polycrystal tensile simulation.

The CPFEM formulation is not repeated here. The multiplicative decomposition,
slip kinetics, hardening law, stress update, and implicit differentiation are
introduced in
[A note on crystal plasticity](https://deepmodeling.github.io/jax-fem/more/useful/main.html#crystal-plasticity-finite-element-method-fem).
This README instead describes what is implemented in these files, the question
answered by each application, and the supplied inputs and outputs.

| File | Task |
| --- | --- |
| `models.py` | Shared constitutive update, local nonlinear solve, internal-variable update, and stress post-processing. |
| `stress_sensitivity.py` | Differentiate the final axial stress with respect to the initial slip resistance. |
| `volume_sensitivity.py` | Differentiate a deformed-volume target with respect to the prescribed displacement. |
| `polycrystal_neper.py` | Run a forward tensile simulation on a supplied Neper polycrystal mesh. |

## Implemented model

`models.py` supplies the material response used by all three applications. The
model uses the 12 `{111}<110>` slip systems of an FCC crystal and a
Kalidindi-type self/latent hardening update [1]. Each cell is assigned a
quaternion, which is converted to a rotation matrix and used to rotate the
elastic tensor and Schmid tensors into the cell orientation.

At every quadrature point, the state contains:

- the inverse plastic deformation gradient, $\boldsymbol{F}_p^{-1}$;
- the resistance of every slip system, $g^\alpha$;
- the slip on every slip system, $\gamma^\alpha$; and
- the crystal rotation matrix.

For a given displacement gradient and previous state, the code solves an
implicit local residual for the second Piola--Kirchhoff stress. The local
Newton iteration uses a backtracking line search. A custom JAX JVP applies
implicit differentiation to this local solve, while `ad_wrapper` differentiates
through the global equilibrium problem. This combination is what makes the two
sensitivity applications possible without deriving their gradients by hand.

The default parameters in `models.py` are:

| Parameter | Value | Role |
| --- | ---: | --- |
| $C_{11}$ | $1.684\times10^5$ MPa | Cubic elastic constant |
| $C_{12}$ | $1.214\times10^5$ MPa | Cubic elastic constant |
| $C_{44}$ | $0.754\times10^5$ MPa | Cubic elastic constant |
| $g_\mathrm{ini}$ | $60.8$ MPa | Initial slip resistance |
| $g_\mathrm{sat}$ | $109.8$ MPa | Saturation slip resistance |
| $h_0$ | $541.5$ MPa | Hardening coefficient |
| $a$ | $2.5$ | Hardening exponent |
| $\dot{\gamma}_0$ | $0.001$ | Reference slip rate |
| $m$ | $0.1$ | Rate-sensitivity exponent |

The current interaction matrix uses a self/latent hardening ratio of one.
These values define the included demonstration and should be replaced or
calibrated for a different material.

## Application 1: stress sensitivity

`stress_sensitivity.py` uses one `HEX8` element on a unit cube with the
identity crystal orientation. The bottom face is fixed in $z$, one corner
removes the remaining rigid translations, and the top face is displaced to
$u_z=0.005$ in ten increments.

The task is to ask how the final axial stress changes when the initial slip
resistance is scaled by a scalar $\alpha$:

```math
g^s_\mathrm{ini}(\alpha)=\alpha g_\mathrm{ini},
\qquad
G(\alpha)=\sigma_{zz}^{\mathrm{final}}(\alpha).
```

The script evaluates
$\mathrm{d}G/\mathrm{d}\alpha$ at $\alpha=1$ with `jax.grad`. It is therefore
a sensitivity kernel that could be used inside parameter calibration, rather
than a calibration optimizer by itself. The expected derivative is
approximately `163.4479802`. The supplied `moose_reference.csv` stores the
MOOSE stress history used by the optional comparison helper after a
corresponding JAX-FEM history has been generated.

A normal invocation first runs the forward problem at $\alpha=1$, saves the
ten solution fields and the stress history, and creates a JAX-FEM/MOOSE
stress--strain comparison. It then reruns the same calculation without file
I/O under `jax.grad`. Keeping output operations outside the differentiated
path avoids introducing host callbacks into the AD calculation.

## Application 2: volume sensitivity

`volume_sensitivity.py` uses the same unit-cube single crystal, but treats the
scale $\beta$ of the prescribed top displacement as the differentiable input.
It defines the deformed volume and squared target mismatch as

```math
V(\beta)=\int_{\Omega}\det(\boldsymbol{I}+\nabla
\boldsymbol{u}(\beta))\,\mathrm{d}V,
\qquad
J(\beta)=\left(V(\beta)-1.01\right)^2.
```

Two loading increments are advanced before the script evaluates
$\mathrm{d}J/\mathrm{d}\beta$ at $\beta=1$. The expected objective after the
second increment is approximately `9.6807590e-05`, and the expected gradient
is `-3.1402652e-06`. This is a compact demonstration of differentiating through
history-dependent constitutive updates and displacement boundary conditions;
it does not run a design optimization loop.

## Application 3: Neper polycrystal

`polycrystal_neper.py` solves a forward tensile problem on a
$0.1\times0.1\times0.1$ cube. The supplied Neper mesh contains 100 grains,
12,167 nodes, and 10,648 `HEX8` elements. Grain IDs are read from the Gmsh
physical tags.

At the beginning of a run, each grain is randomly assigned one of the first
ten orientations in `quaternions.txt`. The left face is fixed in $x$, the
right face is displaced by $0.002L_x$ over ten increments, and point
constraints suppress rigid-body motion. Because the orientation assignment is
random, stress fields can differ between runs unless a NumPy random seed is
set by the user.

For every increment, the application:

1. solves global equilibrium with the PETSc linear solver;
2. computes the cell-averaged Cauchy stress $\sigma_{xx}$;
3. advances $\boldsymbol{F}_p^{-1}$, slip resistance, and slip; and
4. writes a VTU file containing displacement, orientation index, and
   $\sigma_{xx}$.

The checked-in `domain.msh` is sufficient for a normal run. If it is missing,
the script attempts to regenerate the tessellation and mesh with the `neper`
executable under `output/polycrystal_neper/neper/`.

## Inputs and outputs

All required or reference inputs are version-controlled:

```text
input/
├── slip_systems_fcc.txt
├── stress_sensitivity/
│   └── moose_reference.csv
└── polycrystal_neper/
    ├── domain.msh
    └── quaternions.txt
```

| Input | Used by | Contents |
| --- | --- | --- |
| `slip_systems_fcc.txt` | All applications | Slip-plane normals followed by slip directions for the 12 FCC systems. |
| `stress_sensitivity/moose_reference.csv` | Optional comparison helper | Reference single-element stress history from MOOSE. |
| `polycrystal_neper/domain.msh` | Neper polycrystal | Mesh connectivity and grain IDs stored as Gmsh physical tags. |
| `polycrystal_neper/quaternions.txt` | Neper polycrystal | Orientation ID followed by the four quaternion components. |

Generated files are kept below `output/`, which is ignored by Git. The two
sensitivity applications place their Gmsh-generated unit-cube meshes in
`output/<case>/msh/` and their displacement fields in `output/<case>/vtk/`.
The current scripts generate the following files:

```text
output/
├── stress_sensitivity/
│   ├── figures/stress_strain_comparison.png
│   ├── msh/box.msh
│   ├── numpy/jax_fem_out.npy
│   └── vtk/u_000.vtu ... u_009.vtu
├── volume_sensitivity/
│   ├── msh/box.msh
│   └── vtk/u_000.vtu ... u_001.vtu
└── polycrystal_neper/
    └── vtk/u_000.vtu ... u_009.vtu
```

The polycrystal application writes
`output/polycrystal_neper/vtk/u_000.vtu` through `u_009.vtu`. None of the
version-controlled inputs are overwritten.

## Execution

Run all commands from the `jax-fem/` directory. The sensitivity examples need
a working Gmsh installation:

```bash
python -m applications.crystal_plasticity.stress_sensitivity
python -m applications.crystal_plasticity.volume_sensitivity
```

The first command performs the forward simulation, writes its data and
comparison plot, and finally reports the AD sensitivity. The second command
writes the displacement fields for its two loading steps, then reports the
volume objective and its sensitivity.

The polycrystal example reads the supplied mesh directly, so Neper is optional
for a normal run. It is configured to use PETSc:

```bash
python -m applications.crystal_plasticity.polycrystal_neper
```

| Application | Primary result | Expected behavior |
| --- | --- | --- |
| Stress sensitivity | VTU/NumPy history, comparison PNG, and console gradient | $\max\lvert\sigma_{zz}^{\mathrm{JAX-FEM}}-\sigma_{zz}^{\mathrm{MOOSE}}\rvert\approx3.19\times10^{-5}$ MPa and $\mathrm{d}\sigma_{zz}^{\mathrm{final}}/\mathrm{d}\alpha\approx163.4479802$ |
| Volume sensitivity | Console objective and gradient | $J(1)\approx9.6807590\times10^{-5}$ and $\mathrm{d}J/\mathrm{d}\beta\approx-3.1402652\times10^{-6}$ |
| Neper polycrystal | Ten VTU files | Heterogeneous grain-scale $\sigma_{xx}$ field under $x$-direction tension |

## Results

<p align="middle">
  <img src="assets/polycrystal_grain.gif" width="360" />
  <img src="assets/polycrystal_stress.gif" width="360" />
</p>
<p align="middle">
  <em>Grain structure (left) and cell-averaged axial stress (right).</em>
</p>

The VTU sequence can be inspected or animated with
[ParaView](https://www.paraview.org/).

## More examples

For larger and more varied differentiable CPFEM applications, including BCC
and FCC materials, single-crystal and polycrystal benchmarks, parameter
calibration, AD-based sensitivity analysis, and inverse design of grain
orientations, see
[SuperkakaSCU/JAX-CPFEM](https://github.com/SuperkakaSCU/JAX-CPFEM) and the
associated npj Computational Materials article [2].

## References

[1] Kalidindi, S. R., C. A. Bronkhorst, and L. Anand. “Crystallographic texture
evolution in bulk deformation processing of FCC metals.” *Journal of the
Mechanics and Physics of Solids* 40.3 (1992): 537–569.
[doi:10.1016/0022-5096(92)80003-9](https://doi.org/10.1016/0022-5096(92)80003-9)

[2] Hu, F., S. Niezgoda, T. Xue, and J. Cao. “Efficient GPU-computing
simulation platform JAX-CPFEM for differentiable crystal plasticity finite
element method.” *npj Computational Materials* 11 (2025): 46.
[doi:10.1038/s41524-025-01528-2](https://doi.org/10.1038/s41524-025-01528-2)
