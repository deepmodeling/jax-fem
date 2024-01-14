# Thermal-mechanical Coupling

## Formulation

### Governing Equations

We solve a time-dependent problem that involves one-way coupling (temperature field $T$ affects displacement field $\boldsymbol{u}$). The scenario considered here is a Laser Powder Bed Fusion (LPBF) process. We ignore the powder geometry and only consider a box domain $\Omega=[0, L_x]\times[0,L_y]\times[0,L_z]$ for this problem. The governing PDE of heat equation states that

$$
\begin{align*}
\rho C_p \frac{\partial T}{\partial t} &= \nabla \cdot (k \nabla T) &   &\textrm{in}  \nobreakspace \nobreakspace \Omega \times(0, t_f],  \\
T  &= T_0 &  &\textrm{at} \nobreakspace \nobreakspace t=0, \\
T&=T_D & &\textrm{on} \nobreakspace \nobreakspace \Gamma_{D} \times (0,t_f], \\
k\nabla T \cdot \boldsymbol{n} &= q &&  \textrm{on} \nobreakspace \nobreakspace \Gamma_N \times (0,t_f],
\end{align*}
$$

where $\rho$ is material density, $C_p$ is heat capacity, $k$ is thermal conductivity and $q$ is heat flux. The governing PDE of momentum balance states that

$$
\begin{align*}
    -\nabla \cdot \boldsymbol{\sigma} &= \boldsymbol{0} && \textrm{in}  \nobreakspace \nobreakspace \Omega \times(0, t_f], \nonumber \\
    \boldsymbol{u} &= \boldsymbol{u}_D    && \textrm{on} \nobreakspace \nobreakspace \Gamma_D\times(0, t_f],   \\
    \boldsymbol{\sigma} \cdot \boldsymbol{n} &= \boldsymbol{0}   && \textrm{on} \nobreakspace \nobreakspace \Gamma_N\times(0, t_f].
\end{align*}
$$

### Discretization in Time

Let us first discretize in time and obtain the governing equation at time step $n$ for the temperature field:

$$
\begin{align*}
\rho C_p \frac{T^n - T^{n-1}}{\Delta t} &= \nabla \cdot k \nabla T^n &   &\textrm{in}  \nobreakspace \nobreakspace \Omega,  \\
T^n &= T_{\textrm{ambient}} && \textrm{on} \nobreakspace \nobreakspace \Gamma_{\textrm{bottom}}, \\
k\nabla T^n \cdot \boldsymbol{n} &= q_{\textrm{rad}} + q_{\textrm{conv}} && \textrm{on} \nobreakspace \nobreakspace \Gamma_{\textrm{walls}}, \\
k\nabla T^n \cdot \boldsymbol{n} &= q_{\textrm{laser}} + q_{\textrm{rad}} + q_{\textrm{conv}} && \textrm{on} \nobreakspace \nobreakspace \Gamma_{\textrm{top}}, \\
\end{align*}
$$

where $T^n$ is the unknown variable field to be solved, $T^{n-1}$ is known from previous time step. We have imposed Dirichlet boundary condition on the bottom side $\Gamma_{\textrm{bottom}}$ with $T_{\textrm{ambient}}$ being the ambient temperature. On the four side walls $\Gamma_{\textrm{walls}}$,  convective and radiative heat loss  ($q_{\textrm{rad}}$ and $q_{\textrm{conv}}$) are considered. While on the top side $\Gamma_{\textrm{top}}$, laser heat flux $q_{\textrm{laser}}$ should be added. These heat flux terms are defined as the following:


$$
\begin{align*}
q_{\textrm{rad}} &= \sigma \epsilon ( T_{\textrm{ambient}}^4 - (T^{n-1})^4), \\
q_{\textrm{conv}} &= h(T_{\textrm{ambient}} - T^{n-1}), \\
q_{\textrm{laser}} &= \frac{2\eta P}{\pi r_b^2} \textrm{exp}\Big( \frac{-2 \big( (x-x_l)^2 + (y-y_l)^2 \big)}{r_b^2} \Big),
\end{align*}
$$

where $\sigma$ is Stefan-Boltzmann constant, $\epsilon$ is emissivity of the material, $h$ is convection heat-transfer coefficient, $P$ is laser power, $\eta$ is absorption coefficient, $r_b$ is laser beam radius and $(x_l, y_l)$ is laser position. Note that $T^{n-1}$ is used in $q_{\textrm{rad}}$ and $q_{\textrm{conv}}$ so that this recovers Neumann boundary conditions. If we use $T^n$, we will get Robin boundary conditions that require different treatment, which is possible in _JAX-FEM_ but not used in this particular tutorial.

Next, we consider solving $\boldsymbol{u}^n$ with a perfect J2-plasticity model also considering thermal strain. We assume that the total strain $\boldsymbol{\varepsilon}^{n-1}$ and stress $\boldsymbol{\sigma}^{n-1}$ from the previous loading step are known and the temperature solution at current step $T^n$ is also known. The problem states that find the displacement field $\boldsymbol{u}^n$ at the current loading step such that

$$
\begin{align*} 
    -\nabla \cdot \big(\boldsymbol{\sigma}^n (\nabla \boldsymbol{u}^n, \boldsymbol{\varepsilon}^{n-1}, \boldsymbol{\sigma}^{n-1}, \Delta T^n) \big) = \boldsymbol{0} & \quad \textrm{in}  \nobreakspace \nobreakspace \Omega, \nonumber \\
    \boldsymbol{u}^n = \boldsymbol{u}_D &  \quad\textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \nonumber \\
    \boldsymbol{\sigma}^n \cdot \boldsymbol{n} = \boldsymbol{0}  & \quad \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}
$$

The stress $\boldsymbol{\sigma}^n$ is defined through the following relationships:

```math
\begin{align*} 
    \boldsymbol{\sigma}_\textrm{trial} &= \boldsymbol{\sigma}^{n-1} + \Delta \boldsymbol{\sigma}, \nonumber\\
    \Delta \boldsymbol{\sigma} &= \lambda \nobreakspace \textrm{tr}(\Delta \boldsymbol{\varepsilon}) \boldsymbol{I} + 2\mu \nobreakspace \Delta \boldsymbol{\varepsilon}, \nonumber \\
    \Delta \boldsymbol{\varepsilon} &= \boldsymbol{\varepsilon}^n  - \boldsymbol{\varepsilon}^{n-1} -  \boldsymbol{\varepsilon}_{\textrm{th}} = \frac{1}{2}\left[\nabla\boldsymbol{u}^n + (\nabla\boldsymbol{u}^n)^{\top}\right] - \boldsymbol{\varepsilon}^{n-1} - \boldsymbol{\varepsilon}_{\textrm{th}}, \nonumber\\
    \boldsymbol{\varepsilon}_{\textrm{th}}  &= \alpha_V \Delta T^n \boldsymbol{I} = \alpha_V (T^n - T^{n-1}) \boldsymbol{I},\\
    \boldsymbol{s} &= \boldsymbol{\sigma}_\textrm{trial} - \frac{1}{3}\textrm{tr}(\boldsymbol{\sigma}_\textrm{trial})\boldsymbol{I},\nonumber\\
    s &= \sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}, \nonumber\\
    f_{\textrm{yield}} &= s - \sigma_{\textrm{yield}}, \nonumber\\
    \boldsymbol{\sigma}^n &= \boldsymbol{\sigma}_\textrm{trial} -  \frac{\boldsymbol{s}}{s} \langle f_{\textrm{yield}} \rangle_{+}, \nonumber
\end{align*}
```

where $`\boldsymbol{\varepsilon}_{\textrm{th}}`$ is the thermal strain, and the other parameters are explained in our [plasticity example](https://github.com/tianjuxue/jax-fem/tree/main/demos/plasticity). Note that we define material phase to be either in POWDER, LIQUID or SOLID state. A material point is initially in POWDER state, and transforms into LIQUID state if the temperature goes beyond the melting point, and transforms from LIQUID to SOLID state if the temperature drops below the melting point thereafter. The Young's modulus $E$  is set to be a normal value for the SOLID state, while 1% of that for the LIQUID and POWDER state. The thermal expansion coefficient $\alpha_V$ is set to be a normal value for the SOLID state, while 0 for the LIQUID and POWDER state. 


### Weak Form

The weak form for $T^n$ is the following:

$$
\begin{align*}
\int_{\Omega}  \rho C_p \frac{T^n - T^{n-1}}{\Delta t} \delta T \nobreakspace \nobreakspace \textrm{d}x + \int_{\Omega} k \nabla T^n : \nabla \delta T \nobreakspace \nobreakspace \textrm{d}x = \int_{\Gamma_{\textrm{walls}}} (q_{\textrm{rad}} + q_{\textrm{conv}} )\nobreakspace \delta T \nobreakspace \nobreakspace \textrm{d}s   + \int_{\Gamma_{\textrm{top}}} (q_{\textrm{laser}} + q_{\textrm{rad}} + q_{\textrm{conv}} )\nobreakspace \delta T \nobreakspace \nobreakspace \textrm{d}s.
\end{align*}
$$

The weak form for $\boldsymbol{u}^n$ is the following: 

$$
\begin{align*}
\int_{\Omega}  \boldsymbol{\sigma}^n : \nabla \delta \boldsymbol{u} \nobreakspace \nobreakspace \textrm{d}x = 0.
\end{align*}
$$

For each time step, we first solve for $T^n$ and then solve for $\boldsymbol{u}^n$ that depends on $T^n$, hence forming the one-way coupling mechanism. Finally, a good reference on this subject is this paper [1] from Northwestern AMPL, where a Direct Energy Deposition (DED) process is considered.


## Execution
Run
```bash
python -m demos.thermal_mechanical.example
```
from the `jax-fem/` directory.


## Results

Results can be visualized with *ParaWiew*.

<p align="middle">
  <img src="materials/T.gif" width="600" />
</p>
<p align="middle">
    <em >Temperature</em>
</p>


<p align="middle">
  <img src="materials/phase.gif" width="600" />
</p>
<p align="middle">
    <em >Deformation (x10) with legend [Blue: POWDER; White: LIQUID; RED: SOLID]</em>
</p>

<p align="middle">
  <img src="materials/line.gif" width="600" />
  <img src="materials/value.gif" width="600" /> 
</p>
<p align="middle">
    <em >f_plus and stress_xx along the center line on the top surface</em>
</p>

## References

[1] Liao, Shuheng, et al. "Efficient GPU-accelerated thermomechanical solver for residual stress prediction in additive manufacturing." *Computational Mechanics* 71.5 (2023): 879-893.