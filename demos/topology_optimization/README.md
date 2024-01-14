# Topology Optimization with the SIMP Method

## Formulation

We study compliance minimization of a 2D cantilever beam made of a linear elastic material. Following the classic Solid Isotropic Material with Penalization (SIMP) [1] method, the governing PDE is 

$$
\begin{align*} 
    -\nabla \cdot (\boldsymbol{\sigma}(\nabla \boldsymbol{u}, \theta)) = \boldsymbol{0} & \quad \textrm{in}  \, \, \Omega, \nonumber \\
    \boldsymbol{u} = \boldsymbol{0} &  \quad\textrm{on} \, \, \Gamma_D,  \nonumber \\
    \boldsymbol{\sigma} \cdot \boldsymbol{n} =  \boldsymbol{t} & \quad \textrm{on} \, \, \Gamma_N,
\end{align*}
$$

where $\boldsymbol{\sigma}$ is parametrized with $\theta(\boldsymbol{x}) \in [0, 1]$, which is the spatially varying design density field. Specifically, we set the Young's modulus $E=E_{\textrm{min}} + \theta^p (E_{\textrm{max}} - E_{\textrm{min}})$ with $p$ being the penalty exponent. 

The weak form corresponding to the governing PDE states that for any test function $\boldsymbol{v}$, the following equation must hold:

$$
\begin{align*} 
\int_{\Omega}  \boldsymbol{\sigma} :  \nabla \boldsymbol{v} \textrm{ d} \Omega - \int_{\Gamma_N} \boldsymbol{t} \cdot  \boldsymbol{v} \textrm{ d} \Gamma = 0.
\end{align*}
$$

The compliance minimization problem states that

$$
\begin{align*} 
    \min_{\boldsymbol{U}\in\mathbb{R}^{N}, \boldsymbol{\Theta}\in\mathbb{R}^{M}} J(\boldsymbol{U},\boldsymbol{\Theta}) =  \int_{\Gamma_N} \boldsymbol{u}^h \cdot \boldsymbol{t}  \\
    \textrm{s.t.} \quad \boldsymbol{C}(\boldsymbol{U}, \boldsymbol{\Theta}) = \textbf{0}, 
\end{align*}
$$

where $\boldsymbol{u}^h(\boldsymbol{x}) = \sum_k \boldsymbol{U}[k] \boldsymbol{\phi}_k(\boldsymbol{x})$  is the finite element solution field constructed with the solution vector $\boldsymbol{U}$. The design vector $\boldsymbol{\Theta}$ is the discretized version of $\theta$, and the constraint equation $\boldsymbol{C}(\boldsymbol{U}, \boldsymbol{\Theta}) = \textbf{0}$ corresponds to the discretized weak form. The topology optimization problem is therefore a typical **PDE-constrained optimization** problem. 

As one of its salient features, *JAX-FEM* allows users to solve such problems in a handy way. In this example, the external MMA optimizer [2] is adopted. The original optimization problem is reformulated in the following reduced form:

$$
\nonumber \min_{\boldsymbol{\Theta}\in\mathbb{R}^{M}} \widehat{J}(\boldsymbol{\Theta}) = J(\boldsymbol{U}(\boldsymbol{\Theta}),\boldsymbol{\Theta}).
$$

Note that $\boldsymbol{U}$ is implicitly a function of $\boldsymbol{\Theta}$. To call the MMA optimizer, we need to provide the total derivative $\frac{\textrm{d}\widehat{J}}{\textrm{d}\boldsymbol{\Theta}}$, which is computed automatically with *JAX-FEM*. The adjoint method is used under the hood. 

The MMA optimizer accepts constraints. For example, we may want to pose the volume constraint such that the material used for topology optimization cannot exceed a threshold value. Then the previous optimization problem is modified as the following

$$
\begin{align*} 
\min_{\boldsymbol{\Theta}\in\mathbb{R}^{M}} \widehat{J}(\boldsymbol{\Theta}) = J(\boldsymbol{U}(\boldsymbol{\Theta}),\boldsymbol{\Theta}) \\
\textrm{s.t.} \quad g(\boldsymbol{\Theta}) = \frac{\int_{\Omega} \theta \textrm{d}\Omega}{\int_{\Omega} \textrm{d}\Omega }- \bar{v}  \leq 0,
\end{align*}
$$

where $\bar{v}$ is the upper bound of volume ratio.  In this case, we need to pass $\frac{\textrm{d}g}{\textrm{d}\boldsymbol{\Theta}}$ to the MMA solver as the necessary information to handle such constraint. 

> In certain scenario, constraint function may depend not only on the design variable, but also on the state variable, i.e., $g(\boldsymbol{U},\boldsymbol{\Theta})$. For example, limiting the maximum von Mises stress globally over the domain could be such a constraint. This will be handled just fine with *JAX-FEM*. You may check our paper [3] for more details or the more advanced application examples in our repo.

Finally, we want to point to an excellent educational paper on using *JAX* for topology optimization for your further information [4].


## Execution
Run
```bash
python -m demos.topology_optimization.example
```
from the `jax-fem/` directory.


## Results

Results can be visualized with *ParaWiew*.

<p align="middle">
  <img src="materials/to.gif" width="600" />
</p>
<p align="middle">
    <em >TO iterations</em>
</p>

Plot of compliance versus design iterations:


<p align="middle">
  <img src="materials/obj_val.png" width="500" />
</p>
<p align="middle">
    <em >Optimization result</em>
</p>

## References

[1] Bendsoe, Martin Philip, and Ole Sigmund. *Topology optimization: theory, methods, and applications*. Springer Science & Business Media, 2003.

[2] Svanberg, Krister. "The method of moving asymptotes—a new method for structural optimization." *International journal for numerical methods in engineering* 24.2 (1987): 359-373.

[3] Xue, Tianju, et al. "JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science." *Computer Physics Communications* (2023): 108802.

[4] Chandrasekhar, Aaditya, Saketh Sridhara, and Krishnan Suresh. "Auto: a framework for automatic differentiation in topology optimization." *Structural and Multidisciplinary Optimization* 64.6 (2021): 4355-4365.

