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

## Execution
Run
```bash
python -m demos.linear_elasticity.example
```
from the `jax-fem/` directory.


## Results

Visualized with *ParaWiew*:

<p align="middle">
  <img src="materials/sol.png" width="500" />
</p>
<p align="middle">
    <em >Solution</em>
</p>