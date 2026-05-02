# Differentiable mesh (nodal-coordinate derivatives)

This example shows how **JAX-FEM** can differentiate a scalar **objective** with respect to **nodal coordinates** $\mathbf{X}\in\mathbb{R}^{N\times d}$ of the mesh. The implementation uses **`ad_wrapper`** (implicit adjoint through the nonlinear solve). A small **gold finite-difference** check on two independent meshes is provided for one scalar component of $\partial J/\partial \mathbf{X}$.

---

## Problem (nonlinear Poisson on the unit square)

On $\Omega=(0,1)^2$, consider governing equation

$$
-\nabla^2 u + u^3 = g \quad \text{in }\Omega,
$$

with weak form: find $u$ such that for all test functions $v$,

$$
\int_\Omega \nabla u\cdot\nabla v \ \mathrm{d}\Omega + \int_\Omega u^3 v \ \mathrm{d}\Omega = \int_\Omega g v \ \mathrm{d}\Omega.
$$

We use a **manufactured** smooth solution

$$
u^{\mathrm{ex}}(x,y)=\sin(\pi x)(1+y),
$$

so that the source $g$ is chosen consistently (see `NonlinearPoisson.get_mass_map` in `example.py`). On the bottom edge $y=0$,

$$
u^{\mathrm{ex}}(x,0)=\sin(\pi x)\neq 0 \quad\text{(except at corners)},
$$

which makes the bottom nodes important (we will select one node at the bottom side to compare JAX-FEM with finite difference approach).

### Boundary conditions

- **Dirichlet:** $u=0$ on $x=0$ and $x=1$; $u=2\sin(\pi x)$ on $y=1$ (matching $u^{\mathrm{ex}}$ on that edge).
- **Neumann** on $y=0$: outward normal $\mathbf{n}=(0,-1)$, flux data consistent with $u^{\mathrm{ex}}$ (implemented as the surface map in `get_surface_maps`).

---

## Objective and mesh sensitivity

Let $\mathbf{U}(\mathbf{X})\in\mathbb{R}^N$ be the vector of nodal finite-element values after solving the nonlinear system for fixed $\mathbf{X}$. Write $U^{(i)}$ for the scalar at node $i$ (each entry depends on $\mathbf{X}$). This matches the scalar field stored in the code’s solution vector. The objective is

$$
J(\mathbf{X}) = \sum_{i=1}^{N} \bigl(U^{(i)}\bigr)^2,
$$

i.e. the sum of squared nodal values. The quantity of interest for **shape / mesh sensitivity** is the **gradient** of $J$ with respect to $\mathbf{X}$ (same $(N,d)$ layout as the nodal coordinates). Each entry is the partial derivative of $J$ with respect to one nodal coordinate:

$$
\frac{\partial J}{\partial X_{i,\alpha}}
$$

for node index $i$ and spatial component index $\alpha$ (in 2D, $\alpha\in\{1,2\}$ match the two coordinate directions stored in the VTK field `dJ_dxy`).

- **Automatic differentiation (AD):** `jax.value_and_grad` on a forward map that takes nodal positions $\mathbf{X}$, wrapped with `ad_wrapper(problem)`, returns the full gradient tensor together with $J$ and the primal solution.
- **Gold finite difference (FD):** `finite_difference.py` builds two **new** `Mesh` / `Problem` instances, perturbs one chosen node by $\pm\varepsilon$ along one axis, solves each side with the standard `solver`, and forms a **central difference** for the **same** scalar entry (same $(i,\alpha)$ as above; subject to nonlinear solver noise when $\varepsilon$ is small).

The script checks that the **number of Neumann faces** is unchanged under the FD perturbation; otherwise AD and FD would not be comparable (topology of the Neumann set can jump if the perturbation crosses the boundary-classification tolerance).

---

## Files in this folder

| File | Role |
|------|------|
| `example.py` | Full driver: mesh, BCs, `ad_wrapper`, `value_and_grad`, gold FD, VTK export. |
| `finite_difference.py` | `gold_fd_two_independent_problems`: two independent solves, central difference. |
| `images/X.png`, `images/Y.png` | Screenshots of the two coordinate components of the mesh gradient of $J$ (same information as `dJ_dxy`) in ParaView (side by side below). |

---

## Running the example

From the **repository root** (so that `applications` and `jax_fem` resolve):

```bash
python -m applications.differentiable_mesh.example
```

Outputs:

- Console: maximum nodal error against the exact solution at nodes, and a one-line comparison of AD vs gold FD for the chosen perturbation.
- `applications/differentiable_mesh/output/vtk/u.vtu`: solution, exact field, absolute error, and **`dJ_dxy`** (two numbers per node: derivatives of $J$ w.r.t. that node’s first and second coordinate, matching the display formula above with $\alpha=1,2$).

Open the VTU in **ParaView**, color by the first or second component of `dJ_dxy` (or split vectors). Results similar to:

<table>
  <tr>
    <td width="50%" valign="top" align="center"><strong>∂J/∂X<sub>·,1</sub></strong> (1st component of <code>dJ_dxy</code>)<br/><img src="images/X.png" alt="dJ/dX column 1 in ParaView"/></td>
    <td width="50%" valign="top" align="center"><strong>∂J/∂X<sub>·,2</sub></strong> (2nd component of <code>dJ_dxy</code>)<br/><img src="images/Y.png" alt="dJ/dX column 2 in ParaView"/></td>
  </tr>
</table>

The patterns reflect how moving each boundary or interior node affects the discrete energy (the same sum of squared nodal values as in the definition of $J$ above) through the finite element procedures.
