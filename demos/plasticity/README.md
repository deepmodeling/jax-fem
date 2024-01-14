# Plasticity

## Formulation

For perfect J2-plasticity model [1], we assume that the total strain $\boldsymbol{\varepsilon}^{n-1}$ and stress $\boldsymbol{\sigma}^{n-1}$ from the previous loading step are known, and the problem states that find the displacement field $\boldsymbol{u}^n$ at the current loading step such that

$$
\begin{align*} 
    -\nabla \cdot \big(\boldsymbol{\sigma}^n (\nabla \boldsymbol{u}^n, \boldsymbol{\varepsilon}^{n-1}, \boldsymbol{\sigma}^{n-1}) \big) = \boldsymbol{b} & \quad \textrm{in}  \nobreakspace \nobreakspace \Omega, \nonumber \\
    \boldsymbol{u}^n = \boldsymbol{u}_D &  \quad\textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \nonumber \\
    \boldsymbol{\sigma}^n \cdot \boldsymbol{n} = \boldsymbol{t}  & \quad \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}
$$

The stress $\boldsymbol{\sigma}^n$ is defined with the following relationships:

```math
\begin{align*}
    \boldsymbol{\sigma}_\textrm{trial} &= \boldsymbol{\sigma}^{n-1} + \Delta \boldsymbol{\sigma}, \nonumber\\
    \Delta \boldsymbol{\sigma} &= \lambda \nobreakspace \textrm{tr}(\Delta \boldsymbol{\varepsilon}) \boldsymbol{I} + 2\mu \nobreakspace \Delta \boldsymbol{\varepsilon}, \nonumber \\
    \Delta \boldsymbol{\varepsilon} &= \boldsymbol{\varepsilon}^n  - \boldsymbol{\varepsilon}^{n-1} = \frac{1}{2}\left[\nabla\boldsymbol{u}^n + (\nabla\boldsymbol{u}^n)^{\top}\right] - \boldsymbol{\varepsilon}^{n-1}, \nonumber\\
    \boldsymbol{s} &= \boldsymbol{\sigma}_\textrm{trial} - \frac{1}{3}\textrm{tr}(\boldsymbol{\sigma}_\textrm{trial})\boldsymbol{I},\nonumber\\
    s &= \sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}, \nonumber\\
    f_{\textrm{yield}} &= s - \sigma_{\textrm{yield}}, \nonumber\\
    \boldsymbol{\sigma}^n &= \boldsymbol{\sigma}_\textrm{trial} -  \frac{\boldsymbol{s}}{s} \langle f_{\textrm{yield}} \rangle_{+}, \nonumber
\end{align*}
```

where $`\boldsymbol{\sigma}_\textrm{trial}`$ is the elastic trial stress, $`\boldsymbol{s}`$ is the devitoric part of $`\boldsymbol{\sigma}_\textrm{trial}`$, $`f_{\textrm{yield}}`$ is the yield function, $`\sigma_{\textrm{yield}}`$ is the yield strength, $`{\langle x \rangle_{+}}:=\frac{1}{2}(x+|x|)`$ is the ramp function, and $`\boldsymbol{\sigma}^n`$ is the stress at the currently loading step.


The weak form gives

$$
\begin{align*}
\int_{\Omega}  \boldsymbol{\sigma}^n : \nabla \boldsymbol{v} \nobreakspace \nobreakspace \textrm{d}x = \int_{\Omega} \boldsymbol{b}  \cdot \boldsymbol{v} \nobreakspace \textrm{d}x + \int_{\Gamma_N} \boldsymbol{t} \cdot \boldsymbol{v} \nobreakspace\nobreakspace \textrm{d}s.
\end{align*}
$$

In this example, we consider a displacement-controlled uniaxial tensile loading condition. We assume free traction ($\boldsymbol{t}=[0, 0, 0]$) and ignore body force ($\boldsymbol{b}=[0,0,0]$). We assume quasi-static loadings from 0 to 0.1 mm and then unload from 0.1 mm to 0.


> :ghost: A remarkable feature of *JAX-FEM* is that automatic differentiation is used to enhance the development efficiency. In this example, deriving the fourth-order elastoplastic tangent moduli tensor $\mathbb{C}=\frac{\partial \boldsymbol{\sigma}^n}{\partial \boldsymbol{\varepsilon}^n}$ is usually required by traditional FEM implementation, but is **NOT** needed in our program due to automatic differentiation.


## Execution
Run
```bash
python -m demos.plasticity.example
```
from the `jax-fem/` directory.


## Results

Results can be visualized with *ParaWiew*.

<p align="middle">
  <img src="materials/sol.gif" width="400" />
</p>
<p align="middle">
    <em >Deformation (x50)</em>
</p>

Plot of the $`z-z`$ component of volume-averaged stress versus displacement of the top surface:


<p align="middle">
  <img src="materials/stress_strain.png" width="500" />
</p>
<p align="middle">
    <em >Stress-strain curve</em>
</p>

## References

[1] Simo, Juan C., and Thomas JR Hughes. *Computational inelasticity*. Vol. 7. Springer Science & Business Media, 2006.