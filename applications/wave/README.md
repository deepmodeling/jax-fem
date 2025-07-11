# Wave equation

## Formulation

### Governing Equations

We consider the scalar wave equation in a domain $\Omega\subset\mathbb{R}^d$ with boundary $\partial\Omega =\Gamma_D\cup\Gamma_N$, the strong form gives:

$$\begin{align*}\frac{1}{c^2}\frac{\partial^2u}{\partial t^2}&=\nabla^2u+q& &\textrm{in}  \nobreakspace \nobreakspace \Omega \times(0, t_f],\\
u  &= u_0 & &\textrm{at} \nobreakspace \nobreakspace t=0, \\
u&=u_D & &\textrm{on} \nobreakspace \nobreakspace \Gamma_{D} \times (0,t_f], \\
\nabla u \cdot \boldsymbol{n} &= t &&  \textrm{on} \nobreakspace \nobreakspace \Gamma_N \times (0,t_f].\end{align*}$$

where $u$ is the unknown pressure field, $c$ the speed of wave, and $q$ the source term.

We have the following definitions:

* $\Omega=(0,1)\times(0,1)$ (a unit square)
* $\Gamma_D=\partial\Omega$ 
* $q = 0$
* $t = 0$

### Discretization in Time

We first approximate the second-order time derivative with the backward difference scheme:

$$\displaystyle\frac{\partial^2 u}{\partial t^2}\approx\displaystyle\frac{u^{n}-2u^{n-1}+u^{n-2}}{\Delta t^2}$$

The governing equation at time step $n$ for the pressure field can be stated as:

$$\begin{align*}\frac{1}{c^2}\displaystyle\frac{u^{n}-2u^{n-1}+u^{n-2}}{\Delta t^2}&=\nabla^2u^n& &\textrm{in}  \nobreakspace \nobreakspace \Omega,\\
u&=u_D & &\textrm{on} \nobreakspace \nobreakspace \partial\Omega. \\
\end{align*}$$

### Weak form
The weak form for $u^n$ is the following:

$$\int_{\Omega}(c^2\Delta t^2\nabla u^n\cdot\nabla \delta u+u^n\delta u)dx-\int_{\Omega}(2u^{n-1}-u^{n-2})\delta udx=0$$


## Execution
Run
```bash
python -m demos.wave.example
```
from the `jax-fem/` directory.


## Results

Results can be visualized with *ParaWiew*.
<p align="middle">
  <img src="material/pressure.gif" width="600" />
</p>
<p align="middle">
    <em >Wave: pressure</em>
</p>

