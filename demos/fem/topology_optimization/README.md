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

## Implementation

Import some useful modules
```python
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, ad_wrapper
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.fem.mma import optimize
from jax_am.common import rectangle_mesh
```

Define the problem and constitutive relationship. Generally, *JAX-FEM* solves $`-\nabla \cdot \boldsymbol{f}(\nabla \boldsymbol{u}, \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...,\boldsymbol{\alpha}_N) = \boldsymbol{b}`$. Here, we have $`\boldsymbol{f}(\nabla \boldsymbol{u}, \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,...,\boldsymbol{\alpha}_N)=\boldsymbol{\sigma} (\nabla \boldsymbol{u}, \theta)`$, reflected by the function `stress`. The first three functions `custom_init`, `get_tensor_map` and `set_params` override base class methods. In particular, `set_params` sets the design variable $\boldsymbol{\Theta}$. 

```python
class Elasticity(FEM):
    def custom_init(self):
        """Override base class method.
        Modify self.flex_inds so that location-specific TO can be realized. Not important in this example.
        """
        self.flex_inds = np.arange(len(self.cells))

    def get_tensor_map(self):
        """Override base class method.
        """
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            Emax = 70.e3
            Emin = 1e-3*Emax
            nu = 0.3
            penal = 3.
            E = Emin + (Emax - Emin)*theta[0]**penal
            epsilon = 0.5*(u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            return sigma
        return stress

    def set_params(self, params):
        """Override base class method.
        """
        full_params = np.ones((self.num_cells, params.shape[1]))
        full_params = full_params.at[self.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars['laplace'] = [thetas] # 'laplace' is a reserved keyword in JAX-FEM

    def compute_compliance(self, neumann_fn, sol):
        """Surface integral
        """
        boundary_inds = self.neumann_boundary_inds_list[0]
        _, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :] * self.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
        traction = jax.vmap(jax.vmap(neumann_fn))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val
```

Do some cleaning work. Remove old solution files.
```python
data_path = os.path.join(os.path.dirname(__file__), 'data') 
files = glob.glob(os.path.join(data_path, f'vtk/*'))
for f in files:
    os.remove(f)
```

Specify mesh-related information. We use first-order quadrilateral element.
```python
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.
meshio_mesh = rectangle_mesh(Nx=60, Ny=30, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
```

Define boundary values
```python
def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)
    
def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

def dirichlet_val(point):
    return 0.

def neumann_val(point):
    return np.array([0., -100.])

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
neumann_bc_info = [[load_location], [neumann_val]]
```

Define forward problem:
```python
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)
```

Apply the automatic differentiation wrapper. The flag `linear` and `use_petsc` specifies how the forward problem (could be linear or nonlinear) should be solved. The backward adjoint problem is always linear. This is a critical step that makes the problem solver differentiable.
```python
fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)
```

Define the objective function $\widehat{J}(\boldsymbol{\Theta})$. In the following, `sol = fwd_pred(params)` basically says $\boldsymbol{U}=\boldsymbol{U}(\boldsymbol{\Theta})$.
```python
def J_total(params):
    """J(u(theta), theta)
    """     
    sol = fwd_pred(params)
    compliance = problem.compute_compliance(neumann_val, sol)
    return compliance
```

Output solution files to local disk
```python
outputs = []
def output_sol(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol = fwd_pred(params)
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"compliance = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1
output_sol.counter = 0
```

Prepare $\widehat{J}$ and $\frac{\textrm{d}\widehat{J}}{\textrm{d}\boldsymbol{\Theta}}$ that are required by the MMA optimizer:
```python
def objectiveHandle(rho):
    """MMA solver requires (J, dJ) as inputs
    J has shape ()
    dJ has shape (...) = rho.shape
    """
    J, dJ = jax.value_and_grad(J_total)(rho)
    output_sol(rho, J)
    return J, dJ
```

Prepare $g$ and $\frac{\textrm{d}g}{\textrm{d}\boldsymbol{\Theta}}$ that are required by the MMA optimizer:
```python
def consHandle(rho, epoch):
    """MMA solver requires (c, dc) as inputs
    c should have shape (numConstraints,)
    gradc should have shape (numConstraints, ...)
    """
    def computeGlobalVolumeConstraint(rho):
        g = np.mean(rho)/vf - 1.
        return g
    c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    c, gradc = c.reshape((1,)), gradc[None, ...]
    return c, gradc
```


Finalize the details of the MMA optimizer, and solve the TO problem.
```python
vf = 0.5
optimizationParams = {'maxIters':51, 'movelimit':0.1}
rho_ini = vf*np.ones((len(problem.flex_inds), 1))
numConstraints = 1
optimize(problem, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints)
print(f"As a reminder, compliance = {J_total(np.ones((len(problem.flex_inds), 1)))} for full material")
```

Plot the optimization results:
```python
obj = onp.array(outputs)
plt.figure(figsize=(10, 8))
plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
plt.show()
```


## Execution
Run
```bash
python -m demos.fem.topology_optimization.example
```
from the `jax-am/` directory.


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

[1] Bendsoe, Martin Philip, and Ole Sigmund. Topology optimization: theory, methods, and applications. Springer Science & Business Media, 2003.

[2] Svanberg, Krister. "The method of moving asymptotes—a new method for structural optimization." *International journal for numerical methods in engineering* 24.2 (1987): 359-373.

[3] Xue, Tianju, et al. "JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science." Computer Physics Communications (2023): 108802.

[4] Chandrasekhar, Aaditya, Saketh Sridhara, and Krishnan Suresh. "Auto: a framework for automatic differentiation in topology optimization." Structural and Multidisciplinary Optimization 64.6 (2021): 4355-4365.

