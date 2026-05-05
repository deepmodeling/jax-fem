# Updated Lagrangian Neo-Hookean

Updated Lagrangian (UL) hyperelasticity with SIMP-scaled stress, implicit AD, and AD-vs-FD directional checks in density space.

## 1) Problem Setup

At each load step, in the current configuration:

$$
\nabla_x \cdot \sigma + b = 0
\quad \text{in } \omega,
$$

with boundary conditions

$$
u = \bar{u} \ \text{on } \Gamma_u, \qquad
\sigma n = \bar{t} \ \text{on } \Gamma_t.
$$

In this example, displacement is prescribed on left/right faces. Right-face displacement is load-factor driven (axial pull + rigid rotation), built by `nodal_displacement_total_target`.

## 2) Constitutive Model

Kirchhoff stress:

$$
\tau(F) = \mu (b - I) + \lambda \ln J I,
\qquad
b = F F^{\mathsf T},
\qquad
J = \det(F).
$$

Cauchy stress:

$$
\sigma = \frac{1}{J}\tau.
$$

SIMP scaling in code is applied at quadrature points as $\tau \leftarrow \rho^p \tau$.

## 3) UL Weak Form: Transforming from Step n to Step n-1

At step $n$, freeze geometry at step-start placement $x^{n-1}$, and solve for increment $\Delta u$:

$$
x^n = x^{n-1} + \Delta u(x^{n-1}),
\qquad
F_{\text{inc}} = I + \nabla_{x^{n-1}} \Delta u.
$$

Start from the weak form on the Eulerian configuration:

$$
\int_{\omega^n} \sigma : \nabla_x v dv^n = 0.
$$

Use the map $x = \varphi(x^{n-1})$, with

$$
dv^n = J_{\text{inc}} dv^{n-1},
\qquad
\nabla_x v = \nabla_{x^{n-1}} v F_{\text{inc}}^{-1}.
$$

Then

$$
\int_{\omega^n} \sigma : \nabla_x v dv^n = \int_{\omega^{n-1}} J_{\text{inc}} \sigma F_{\text{inc}}^{-\mathsf T} : \nabla_{x^{n-1}} v dv^{n-1} = 0.
$$

Define

$$
T_{\text{pull}} = J_{\text{inc}} \sigma F_{\text{inc}}^{-\mathsf T},
$$

so the implemented weak form is

$$
\int_{\omega^{n-1}} T_{\text{pull}} : \nabla_{x^{n-1}} v dv^{n-1} = 0.
$$

This matches code variables in `get_universal_kernel`: `F_inc`, `F`, `tau`, `cauchy`, `J_inc`, `F_inc_inv_T`, `T_pull`, and contraction with `cell_v_grads_JxW`.

## 4) Kinematics and State Update

Multiplicative update is chain rule:

$$
F^n
= \frac{\partial x^n}{\partial X_0}
= \frac{\partial x^n}{\partial x^{n-1}} \frac{\partial x^{n-1}}{\partial X_0}
= F_{\text{inc}}^n F^{n-1}.
$$

In code:

- `F_prev` stores $F^{n-1}$ at quadrature points.
- `set_params(...)` updates frozen geometry and state.
- `push_F_prev(...)` advances $F^{n-1} \to F^n$ after each converged increment.

## 5) Load Schedule and Incremental Dirichlet Data

The loading process is divided into several steps. At each step $n$, the load factor $\lambda$ is calculated as: $n$ divided by the total number of load steps, where $n$ runs from 1 up to the total number of steps.

The incremental Dirichlet boundary data for each step is determined by taking the difference between the total displacement at the current load factor and the total displacement at the previous load factor.

$$
\Delta u_\Gamma
= u_{\text{tot}}(\lambda^n) - u_{\text{tot}}(\lambda^{n-1}).
$$

Implemented by `nodal_displacement_total_target` and `set_params`.

## 6) Design Variable, Objective, AD/FD Check

- Design variable: `rho_design` (cellwise), clipped to `[RHO_MIN, 1]`.
- Quadrature density: `rho_q` (broadcast from cell values).
- Objective:

$$
J = \sum_i \|u_{\text{cum},i}\|^2
$$

(`objective_ul` is `jnp.sum(u_cum ** 2)`).

- AD: `jax.value_and_grad(..., has_aux=True)` through full load history **considering mesh update**.
- FD directional check:

$$
\frac{J(\rho + \varepsilon v) - J(\rho)}{\varepsilon}
$$

with random unit direction $v$.

## 7) Run and Output

Run from repo root:

```bash
python -m applications.updated_lagrangian.example
```

The script clears `applications/updated_lagrangian/output/` and writes `delta_u_*.vtu`.

For each step file:

- mesh points: step-start configuration $x^{n-1}$,
- `sol`: increment $\Delta u$.

In ParaView, `Warp By Vector` with `sol` gives the step-end placement.
