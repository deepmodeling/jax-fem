# Hyperelasticity

## Formulation

The governing equation for hyperelasticity of a body $\Omega$ can be written as

$$
\begin{align*}
    -\nabla \cdot \boldsymbol{P}  = \boldsymbol{b} & \quad \textrm{in}  \nobreakspace \nobreakspace \Omega, \\
    \boldsymbol{u} = \boldsymbol{u}_D &  \quad\textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \\
    \boldsymbol{P}  \cdot \boldsymbol{n} = \boldsymbol{t}  & \quad \textrm{on} \nobreakspace \nobreakspace \Gamma_N.
\end{align*}
$$

The weak form gives

$$
\begin{align*}
\int_{\Omega}  \boldsymbol{P} : \nabla \boldsymbol{v} \nobreakspace \nobreakspace \textrm{d}x = \int_{\Omega} \boldsymbol{b}  \cdot \boldsymbol{v} \nobreakspace \textrm{d}x + \int_{\Gamma_N} \boldsymbol{t} \cdot \boldsymbol{v} \nobreakspace\nobreakspace \textrm{d}s.
\end{align*}
$$

Here, $\boldsymbol{P}$ is the first Piola-Kirchhoff stress and is given by

$$
\begin{align*} 
    \boldsymbol{P} &= \frac{\partial W}{\partial \boldsymbol{F}},  \\
    \boldsymbol{F} &= \nabla \boldsymbol{u} + \boldsymbol{I},  \\
    W (\boldsymbol{F}) &= \frac{G}{2}(J^{-2/3} I_1 - 3) + \frac{\kappa}{2}(J - 1)^2,
\end{align*}
$$

where $\boldsymbol{F}$ is the deformation gradient and $W$ is the strain energy density function. This constitutive relationship comes from a neo-Hookean solid model [2].


We have the following definitions:
* $\Omega=(0,1)\times(0,1)\times(0,1)$ (a unit cube)
* $\Gamma_{D_1}=0\times(0,1)\times(0,1)$ (first part of Dirichlet boundary)
* $\boldsymbol{u}_{D_1}= [0,(0.5+(x_2−0.5)\textrm{cos}(\pi/3)−(x_3−0.5)\textrm{sin}(\pi/3)−x_2)/2, (0.5+(x_2−0.5)\textrm{sin}(\pi/3)+(x_3−0.5)\textrm{cos}(\pi/3)−x_3)/2]$
* $\Gamma_{D_2}=1\times(0,1)\times(0,1)$ (second part of Dirichlet boundary)
* $\boldsymbol{u}_{D_2}=[0,0,0]$ 
* $b=[0, 0, 0]$
* $t=[0, 0, 0]$

## Execution
Run
```bash
python -m demos.hyperelasticity.example
```
from the `jax-fem/` directory.


## Results

Visualized with *ParaWiew* "Warp By Vector" function:

<p align="middle">
  <img src="materials/sol.png" width="500" />
</p>
<p align="middle">
    <em >Solution</em>
</p>

## References

[1] https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/hyperelasticity/python/documentation.html

[2] https://en.wikipedia.org/wiki/Neo-Hookean_solid