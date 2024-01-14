# Automatic differentiation

## Formulation

In this tutorial, we demostrate the process to calculate the derivative by automatic differentiation and validate the results by the finite difference method. The same hyperelastic body as in our [hyperelasticity example](https://github.com/tianjuxue/jax-fem/tree/main/demos/hyperelasticity) is considered here, i.e., a unit cube with a neo-Hookean solid model. In addition, we have the following definitions:
* $\Omega=(0,1)\times(0,1)\times(0,1)$ (a unit cube)
*  $b=[0, 0, 0]$
* $\Gamma_{D}=(0,1)\times(0,1)\times0$
* $\boldsymbol{u}_{D}=[0,0,\beta]$ 
* $\Gamma_{N_1}=(0,1)\times(0,1)\times1$
* $\boldsymbol{t}_{N_1}=[0, 0, -1000]$
* $\Gamma_{N_2}=\partial\Omega\backslash(\Gamma_{D}\cup\Gamma_{N_1})$
* $\boldsymbol{t}_{N_2}=[0, 0, 0]$

The objective function is defined as:

$$J= \sum_{i=1}^{N_d}(\boldsymbol{u}[i])^2$$
where $N_d$ is the total number of degrees of freedom. $\boldsymbol{u}[i]$ is the $i\text{th}$ component of the dispalcement vector $\boldsymbol{u}$, which is obtained by solving the following discretized governing PDE:

$$
\boldsymbol{C}(\boldsymbol{u},\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...\boldsymbol{\alpha}_N)=\boldsymbol{0}
$$

where $\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...\boldsymbol{\alpha}_N$ are the parameter vectors. Here, we set up three parameters, $\boldsymbol{\alpha}_1 = \boldsymbol{E}$ the elasticity modulus, $\boldsymbol{\alpha}_2 =\boldsymbol{\rho}$ the material density, and $\boldsymbol{\alpha}_3 =\boldsymbol{\beta}$ the scale factor of the Dirichlet boundary conditions.

We can see that $\boldsymbol{u}(\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...\boldsymbol{\alpha}_N)$ is the implicit function of the parameter vectors. In JAX-FEM, users can easily compute the derivative of the objective function with respect to these parameters through automatic differentiation. We first wrap the forward problem with the function `jax_fem.solver.ad_wrapper`, which defines the implicit differentiation through `@jax.custom_vjp`. Next, we can use the `jax.grad` to calculate the derivative. 


We then use the forward differnce scheme to validate the results. The derivative of the objective with respect to the $k\text{th}$ component of the parameter vector $\boldsymbol{\alpha}_i$ is defined as:
$$\frac{\partial J}{\partial \boldsymbol{\alpha}_i[k]} = \frac{J(\boldsymbol{\alpha}_i+h\boldsymbol{\alpha}_i[k])-J(\boldsymbol{\alpha}_i)}{h\boldsymbol{\alpha}_i[k]}$$

where $h$ is a small perturbation.



## Execution
Run
```bash
python -m demos.inverse.example
```
from the `jax-fem/` directory.


## Results

```bash
Derivative comparison between automatic differentiation (AD) and finite difference (FD)
dE = 4.0641751938577116e-07, dE_fd = 0.0, WRONG results! Please avoid gradients w.r.t self.E
drho[0, 0] = 0.002266954599447443, drho_fd_00 = 0.0022666187078357325
dscale_d = 431.59223609853564, dscale_d_fd = 431.80823609844765
```
