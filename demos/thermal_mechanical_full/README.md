# Thermal-mechanical Coupling (Full coupling)

## Formulation

### Governing Equations

We solve a time-dependent problem that involves full coupling (temperature field $T$ and displacement field $\boldsymbol{u}$ interact with each other). Here, we refer to the example shown in [1], where the domain consists of a quarter of a square plate $\Omega_s=[0, L_x]\times[0,L_y]$ perforated by a circular hole $\Omega_c= \lbrace(x,y)|x^2+y^2\lt R\rbrace$, i.e., $\Omega=\Omega_s\setminus\Omega_c$.

The linearized thermoelastic constitutive equations are given by:

$$\boldsymbol{\varepsilon}=\frac{1}{2}[\nabla \boldsymbol{u}+(\nabla\boldsymbol{u})^T]$$

$$\boldsymbol{\sigma} = \boldsymbol{C}:(\boldsymbol{\varepsilon}-\alpha(T-T_0)\boldsymbol{I}) = \lambda tr(\boldsymbol{\varepsilon})\boldsymbol{I}+2\mu\boldsymbol{\varepsilon}-\alpha(3\lambda+2\mu)(T-T_0)\boldsymbol{I}$$

$$\rho s = \rho s_0+\displaystyle\frac{\rho C_{\varepsilon}}{T_0}(T-T_0) + \kappa tr(\boldsymbol{\varepsilon})$$

where $\lambda$ and $\mu$ are the Lamé coefficients. $\rho$ is the material density. $\alpha$ is the thermal expansion coefficient. $\kappa = \alpha(3\lambda+2\mu)$ is the thermal conductiivty. $C_{\varepsilon}$ is the specific heat per unit of mass at constant strain. $s$ is the entropy per unit of mass in the current.

The governing PDE of temperature field without source terms states that:

$$
\begin{align*}
\rho T_0 \dot{s}&=\nabla\cdot({k\nabla T})& &\textrm{in}  \nobreakspace \nobreakspace \Omega \times(0, t_f],\\
T  &= T_0 & &\textrm{at} \nobreakspace \nobreakspace t=0, \\
T&=T_D & &\textrm{on} \nobreakspace \nobreakspace \Gamma_{D} \times (0,t_f], \\
k\nabla T \cdot \boldsymbol{n} &= q &&  \textrm{on} \nobreakspace \nobreakspace \Gamma_N \times (0,t_f].
\end{align*}
$$

where $k$ and $q$ are the thermal conductivity and the heat flux, respectively. $T_0$ is the the ambient temperature. The governing PDE of momentum balance states that:

$$
\begin{align*}
    -\nabla \cdot \boldsymbol{\sigma} &= \boldsymbol{0} && \textrm{in}  \nobreakspace \nobreakspace \Omega \times(0, t_f], \nonumber \\
    \boldsymbol{u} &= \boldsymbol{u}_D    && \textrm{on} \nobreakspace \nobreakspace \Gamma_D\times(0, t_f],   \\
    \boldsymbol{\sigma} \cdot \boldsymbol{n} &= \boldsymbol{0}   && \textrm{on} \nobreakspace \nobreakspace \Gamma_N\times(0, t_f].
\end{align*}
$$

### Discretization in Time

Let us first discretize in time and obtain the governing equation at time step $n$ for the temperature field. From the above definition, we have:

$$\rho T_0 \dot{s} = \rho C_\varepsilon\dot{T}+\kappa T_0tr(\dot{\boldsymbol{\varepsilon}})$$

We use the implicit Euler scheme to calculate the time derivative, and the governing equation of the temperature field can be written as:

$$\begin{align*}
    \rho C_\varepsilon\displaystyle\frac{T^n-T^{n-1}}{\Delta t}+\kappa T_0tr(\displaystyle\frac{\boldsymbol{\varepsilon}^n-\boldsymbol{\varepsilon}^{n-1}}{\Delta t})&=\nabla\cdot({k\nabla T^n}) && \textrm{in}  \nobreakspace \nobreakspace \Omega , \nonumber \\
    T^n&=T_{inc} & &\textrm{on} \nobreakspace \nobreakspace \Gamma_{hole}, \\
    k\nabla T^n \cdot \boldsymbol{n} &= q_0 &&  \textrm{on} \nobreakspace \nobreakspace \Gamma_{square}.
\end{align*}$$

where $T^n$ and $\boldsymbol{\varepsilon}^n$ the unknown variable field to be solved, $T^{n-1}$ and $\boldsymbol{\varepsilon}^{n-1}$ are known from previous time step. We have imposed Dirichlet boundary condition on the circular hole boudary with $T_{inc}$ being the assigned temperature. On the other boundaries, the zero Neumann boundary conditions $q_0 = 0$ is considered. It should be noted that in this example, the temperature variation $\Theta = T-T_0$ is treated as the unknown variable field, which also appears in the stress constitutive relation.

With the stress constitutive relation defined above, the displacemnet field $\boldsymbol{u}^n$ at the current time step satisfies that:

$$\begin{align*}
    -\nabla \cdot (\boldsymbol{\sigma}^n(\nabla\boldsymbol{u}^n,T^n)) &= \boldsymbol{0} && \textrm{in}  \nobreakspace \nobreakspace \Omega, \nonumber \\
    \boldsymbol{u}^n &= \boldsymbol{u}_D    && \textrm{on} \nobreakspace \nobreakspace \Gamma_D,   \\
    \boldsymbol{\sigma}^n \cdot \boldsymbol{n} &= \boldsymbol{0}   && \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}$$


### Weak form
The weak form for $T^n$ is the following:

$$R_T = \int_{\Omega}(\rho C_\varepsilon\displaystyle\frac{T^n-T^{n-1}}{\Delta t}+\kappa T_0tr(\displaystyle\frac{\boldsymbol{\varepsilon}^n-\boldsymbol{\varepsilon}^{n-1}}{\Delta t}))\delta Td\Omega+\int_{\Omega}k\nabla T\cdot\nabla\delta Td\Omega=0$$

The weak form for $\boldsymbol{u}^n$ is the following:

$$R_{\boldsymbol{u}}=\int_{\Omega}(\lambda tr(\boldsymbol{\varepsilon}^n)\boldsymbol{I}+2\mu\boldsymbol{\varepsilon}^n-\kappa(T^n-T_0)\boldsymbol{I}):\nabla \delta \boldsymbol{u}d\Omega = 0$$

So,the weak form of the coupling system can be stated as 

$$R=R_T+R_{\boldsymbol{u}}=0$$

For each time step, we simultaneously solving $T^n$ and $\boldsymbol{u}^n$, hence forming the full coupling mechanism.


## Execution
Run
```bash
python -m demos.thermal_mechanical_full.example
```
from the `jax-fem/` directory.


## Results

Results can be visualized with *ParaWiew*.
<p align="middle">
  <img src="material/theta.gif" width="400" />
  <img src="material/uy.gif" width="400" />
</p>
<p align="middle">
    <em >Thermal mechanical coupling: temperature change (left) and displacement-y (right).</em>
</p>

## References

[1] https://comet-fenics.readthedocs.io/en/latest/demo/thermoelasticity/thermoelasticity_transient.html
