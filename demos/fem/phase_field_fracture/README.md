# Phase Field Fracture

## Formulation

### Minimization of Energy Functional

In the phase field framework, fracture processes can be determined via the minimization of the following energy functional over displacement field $\boldsymbol{u}$ and phase field variable $d$:

$$
\begin{align*} 
    \Pi(\boldsymbol{u}, d) = \int_{\Omega} \psi(\boldsymbol{\varepsilon}(\boldsymbol{u}), d)\textrm{ d}\Omega + \int_{\Omega}g_c\gamma(d, \nabla d)\textrm{ d}\Omega  - \int_{\Omega}\boldsymbol{b}\cdot\boldsymbol{u} \textrm{ d}\Omega - \int_{\Gamma_N}\boldsymbol{t}\cdot\boldsymbol{u} \textrm{ d}\Gamma, 
\end{align*}
$$

where $g_c$ is the Griffith-type critical energy release rate and $\gamma$ is the regularized crack surface density function (per volume):

$$
\begin{align*} 
    \gamma(d, \nabla d) = \frac{1}{2l}d^2 + \frac{l}{2} |\nabla d|^2,
\end{align*}
$$

with $l$ being the length-scale parameter. The bulk elastic energy $\psi$ is assumed to take the following form:

$$
\begin{align*}
    \psi(\boldsymbol{\varepsilon}, d) = g(d) \psi_{+}(\boldsymbol{\varepsilon}) + \psi_{-}(\boldsymbol{\varepsilon}),
\end{align*}
$$

with $g(d) = (1-d)^2$ being the degradation function that models partial loss of stiffness due to the presence of cracks. 


Miehe's model [1] of spectral decomposition of the strain tensor is adopted:

```math
\begin{align*} 
    \psi_{\pm}(\boldsymbol{\varepsilon}) &= \frac{\lambda}{2}\langle\textrm{tr}(\boldsymbol{\varepsilon})\rangle^2_{\pm} + \mu\boldsymbol{\varepsilon}_{\pm}:\boldsymbol{\varepsilon}_{\pm},
\end{align*}
```


where $`{\boldsymbol{\varepsilon}_{\pm}:=\Sigma_{a=1}^n \langle \epsilon_a \rangle_{\pm} \boldsymbol{n}_a \otimes \boldsymbol{n}_a}`$, with $`\{ \epsilon_a \}_{a=1}^n`$ and $`\{ \boldsymbol{n}_a \}_{a=1}^n`$ being the principal strains and principal directions of $\boldsymbol{\varepsilon}$, respectively. We have ${\langle x \rangle_{\pm}}:=\frac{1}{2}(x\pm|x|)$ being the bracket operator. Miehe's model prevents crack generation in compression.

### Strong Form

The governing equations and boundary conditions obtained by minimizing the total energy functional are: 

$$
\begin{align*}
    \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{b} &= \boldsymbol{0} && \textrm{in}\nobreakspace \nobreakspace \Omega,  \\
    \frac{g_c}{l}\Big( d - l^2 \Delta d \Big) &= 2(1-d)  \mathcal{H} && \textrm{in} \nobreakspace \nobreakspace \Omega, \\
    \boldsymbol{u} &= \boldsymbol{u}_D && \textrm{on} \nobreakspace \nobreakspace \Gamma_D,  \\
    \boldsymbol{\sigma} \cdot \boldsymbol{n} &= \boldsymbol{t}   && \textrm{on} \nobreakspace\nobreakspace\Gamma_N, \\
    \nabla d \cdot \boldsymbol{n} &= 0    && \textrm{on} \nobreakspace\nobreakspace \Gamma,
\end{align*}
$$

where $\boldsymbol{\sigma} = \frac{\partial \psi}{\partial \boldsymbol{\varepsilon}}=g(d)\boldsymbol{\sigma}^+ + \boldsymbol{\sigma}^-$, and we have

$$
\begin{align*}
\boldsymbol{\sigma}^{\pm} = \frac{\partial \psi^{\pm}}{\partial \boldsymbol{\varepsilon}} = \lambda \langle\textrm{tr}(\boldsymbol{\varepsilon})\rangle_{\pm} \boldsymbol{I} + 2\mu \boldsymbol{\varepsilon}_{\pm}.
\end{align*}
$$

The history variable prevents crack from healing itself during unloading:

$$
\begin{align*}
 \mathcal{H}(\boldsymbol{x}, t)= \max_{s\in[0, t]}\psi_+(\boldsymbol{\varepsilon}(\boldsymbol{x}, s)).
\end{align*}
$$

### Weak Form

For arbitrary test function $(\delta\boldsymbol{u}, \delta d)$, we pose the following variational problem

$$
\begin{align*}
    a\big( (\boldsymbol{u}, d), (\delta\boldsymbol{u}, \delta d) \big) &= F\big( (\delta \boldsymbol{u}, \delta d) \big),
\end{align*}
$$

where we have

$$
\begin{align*}
    a\big( (\boldsymbol{u}, d), (\delta\boldsymbol{u}, \delta d) \big) &= \int_{\Omega} \boldsymbol{\sigma}(\boldsymbol{\varepsilon}, d):\nabla \delta \boldsymbol{u} \textrm{ d}\Omega + g_c\int_{\Omega} \Big(\frac{d}{l}\delta d + l\nabla d \cdot \nabla \delta d \Big)  \textrm{ d}\Omega + \int_{\Omega} 2\mathcal{H}d \nobreakspace \delta d \textrm{ d}\Omega, \\
    F\big( (\delta \boldsymbol{u}, \delta d) \big) &= \int_{\Gamma_N} \boldsymbol{t} \cdot \delta \boldsymbol{u} \textrm{ d}\Gamma + \int_{\Omega}\boldsymbol{b}\cdot\delta\boldsymbol{u} \textrm{ d}\Omega + \int_{\Omega} 2\mathcal{H}\delta d \textrm{ d}\Omega.
\end{align*}
$$

There are two common schemes to solve this coupled nonlinear problem: monolithic and staggered schemes. In this example, we use the staggered scheme. More details can be found in our paper [2]. 



### Eigenvalue/Eigenvector Derivative with Degeneracy

Before we move to the implementation section, caveats on computing the derivative of eigenvalues and eigenvectors (especially with **degenerate** eigenvalues) are briefly discussed here. One may tend to fully rely on _JAX_ automatic differentiation to compute the derivative of eigenvalues and eigenvectors. However, when repeated eigenvalues occur, _JAX_ native `jax.grad` may fail and return `np.nan`, as discussed in this [post](https://github.com/google/jax/issues/669). The issue has its own complexity, and is not resolved yet. 

One workaround is to add a small random noise to the matrix so that it always has distinct eigenvalues. This approach proves to be effective in our implementation of the phase field method. 

The second approach is to define [custom derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) with knowledge to handle repeated eigenvalues. In our example, _JAX-FEM_ needs to computes $\frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}$, which further requires to compute $\frac{\partial \boldsymbol{\varepsilon}^+}{\partial \boldsymbol{\varepsilon}}$ and $\frac{\partial \boldsymbol{\varepsilon}^-}{\partial \boldsymbol{\varepsilon}}$. More generally, if a second order tensor $\boldsymbol{A}$ decompose as $`\boldsymbol{A} =\Sigma_{a=1}^n \lambda_a  \boldsymbol{n}_a \otimes \boldsymbol{n}_a`$ and we define tensor map  $`\boldsymbol{F}(\boldsymbol{A}):=\Sigma_{a=1}^n f(\lambda_a)  \boldsymbol{n}_a \otimes \boldsymbol{n}_a`$, then we are interested in computing $\frac{\partial \boldsymbol{F}}{\partial \boldsymbol{A}}$. The procedures are well presented in Miehe's paper [3], in particular, Eq. (19) is what we are concerned about. We implemented the algorithms in the file  [`eigen.py`](https://github.com/tianjuxue/jax-am/blob/main/demos/fem/phase_field_fracture/eigen.py). In this file, you will see how native AD of _JAX_ fails on repeated eigenvalues, but once custom derivative rules are specified, the issues is resolved.

Finally, make sure your _JAX_ version is up-to-date, since we have observed some possible unexpected behavior of the function `np.linalg.eigh` in older versions of _JAX_, e.g., 0.3.x version.


## Implementation 

Import some useful modules
```python
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import meshio
import matplotlib.pyplot as plt
import time

from jax_am.fem.generate_mesh import box_mesh, Mesh
from jax_am.fem.solver import solver
from jax_am.fem.core import FEM
from jax_am.fem.utils import save_sol

from demos.fem.phase_field_fracture.eigen import get_eigen_f_custom
```


If you have multiple GPUs, set the one to use. 
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

Define some useful directory paths
```python
crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')
vtk_dir = os.path.join(data_dir, 'vtk')
numpy_dir = os.path.join(data_dir, 'numpy')
os.makedirs(numpy_dir, exist_ok=True)
```

The bracket operator. One may define something like `lambda x: np.maximum(x, 0.)`  and `lambda x: np.minimum(x, 0.)`, but it turns out that they may lead to unexpected behaviors. See more discussions and tests in the file [`eigen.py`](https://github.com/tianjuxue/jax-am/blob/main/demos/fem/phase_field_fracture/eigen.py). 
```python
safe_plus = lambda x: 0.5*(x + np.abs(x))
safe_minus = lambda x: 0.5*(x - np.abs(x))
```

Define the phase field variable class. Note how `get_tensor_map` and `get_mass_map` specify the corresponding terms in the weak form. Since the displacement variable $\boldsymbol{u}$ affects the phase field variable $d$ through the history variable $\mathcal{H}$, we need to set this using `set_params`.
```python
class PhaseField(FEM):
    def get_tensor_map(self):
        """Override base class method.
        """
        def fn(d_grad):
            return G_c*l*d_grad
        return fn

    def get_mass_map(self):
        """Override base class method.
        """
        def fn(d, history):
            return G_c/l*d - 2.*(1 - d)*history
        return fn
    
    def set_params(self, history):
        """Override base class method.
        Note that 'mass' is a reserved keyword.
        """
        self.internal_vars['mass'] = [history]
```

Define the displacement variable class. As we previously discussed, native _JAX_ AD may return NaN in the cases with repeated eigenvalues. We provide two workarounds and users can choose either one to use. The first option adds a small noise to the strain tensor, while the second option defines custom derivative rules to properly handle repeated eigenvalues.
```python
class Elasticity(FEM):
    def get_tensor_map(self):
        """Override base class method.
        """
        _, stress_fn = self.get_maps()
        return stress_fn

    def get_maps(self):
        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon
    
        def psi_plus(epsilon):
            eigen_vals, eigen_evecs = np.linalg.eigh(epsilon)
            tr_epsilon_plus = safe_plus(np.trace(epsilon))
            return lmbda/2.*tr_epsilon_plus**2 + mu*np.sum(safe_plus(eigen_vals)**2)
    
        def psi_minus(epsilon):
            eigen_vals, eigen_evecs = np.linalg.eigh(epsilon)
            tr_epsilon_minus = safe_minus(np.trace(epsilon))
            return lmbda/2.*tr_epsilon_minus**2 + mu*np.sum(safe_minus(eigen_vals)**2) 
    
        def g(d):
            return (1 - d[0])**2 + 1e-3
    
        key = jax.random.PRNGKey(0)
        noise = jax.random.uniform(key, shape=(self.dim, self.dim), minval=-1e-8, maxval=1e-8)
        noise = np.diag(np.diag(noise))
    
        def stress_fn_opt1(u_grad, d):
            epsilon = strain(u_grad)
            epsilon += noise
            sigma = g(d)*jax.grad(psi_plus)(epsilon) + jax.grad(psi_minus)(epsilon) 
            return sigma
    
        def stress_fn_opt2(u_grad, d):
            epsilon = strain(u_grad)
    
            def fn(x):
                return 2*mu*(g(d) * safe_plus(x) + safe_minus(x))
            eigen_f = get_eigen_f_custom(fn)
    
            tr_epsilon_plus = safe_plus(np.trace(epsilon))
            tr_epsilon_minus = safe_minus(np.trace(epsilon))
            sigma1 = lmbda*(g(d)*tr_epsilon_plus + tr_epsilon_minus)*np.eye(self.dim) 
    
            sigma2 = eigen_f(epsilon)
            sigma = sigma1 + sigma2
    
            return sigma  

        # Replace stress_fn_opt1 with stress_fn_opt2 will use the second option 
        stress_fn = stress_fn_opt1
    
        def psi_plus_fn(u_grad):
            epsilon = strain(u_grad)
            return psi_plus(epsilon)
    
        return psi_plus_fn, stress_fn
    
    def compute_history(self, sol_u, history_old):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol_u, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        psi_plus_fn, _ = self.get_maps()
        vmap_psi_plus_fn = jax.vmap(jax.vmap(psi_plus_fn))
        psi_plus = vmap_psi_plus_fn(u_grads)
        history = np.maximum(psi_plus, history_old)
        return history
    
    def set_params(self, params):
        """Override base class method.
        """
        sol_d, disp = params
        d = self.convert_from_dof_to_quad(sol_d)
        self.internal_vars['laplace'] = [d]
        dirichlet_bc_info[-1][-2] = get_dirichlet_load(disp)
        self.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    
    def compute_traction(self, location_fn, sol_u, sol_d):
        """For post-processing only
        """
        stress = self.get_tensor_map()
        vmap_stress = jax.vmap(stress)
        def traction_fn(u_grads, d_face):
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
            # (num_selected_faces, num_face_quads, vec) -> (num_selected_faces*num_face_quads, vec)
            d_face_reshape = d_face.reshape(-1, d_face.shape[-1])
            sigmas = vmap_stress(u_grads_reshape, d_face_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.])
            traction = (sigmas @ normals[None, None, :, None])[:, :, :, 0]
            return traction
    
        boundary_inds = self.get_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = sol_u[self.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
        selected_cell_sols_d = sol_d[self.cells][boundary_inds[:, 0]] # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) -> (num_selected_faces, num_face_quads, vec) 
        d_face = np.sum(selected_cell_sols_d[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        traction = traction_fn(u_grads_face, d_face) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
        traction_integral_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))
    
        return traction_integral_val
```

Define some material parameters
```python
# Units are in [kN], [mm] and [s]
G_c = 2.7e-3 # Critical energy release rate [kN/mm] 
E = 210 # Young's modulus [kN/mm^2]
nu = 0.3 # Poisson's ratio
l = 0.02 # Length-scale parameter [mm]
mu = E/(2.*(1. + nu)) # First Lamé parameter
lmbda = E*nu/((1+nu)*(1-2*nu)) # Second Lamé parameter
```

Create the mesh. HEX8 is used in this example.
```python
Nx, Ny, Nz = 50, 50, 1 
Lx, Ly, Lz = 1., 1., 0.02
meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
```

Define boundary locations:
```python
def y_max(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def y_min(point):
    return np.isclose(point[1], 0., atol=1e-5)
```

Create an instance of the phase field problem. 
```python
problem_d = PhaseField(mesh, vec=1, dim=3)
sol_d = onp.zeros((len(mesh.points), 1))
flag = (mesh.points[:, 1] > 0.5*Ly - 0.01*Ly) & (mesh.points[:, 1] < 0.5*Ly + 0.01*Ly) & (mesh.points[:, 0] > 0.5*Lx) 
sol_d[flag] = 1. # Specify initial crack
sol_d_old = onp.array(sol_d)
```

Create an instance of the displacement problem. 
```python
def dirichlet_val(point):
    return 0.

def get_dirichlet_load(disp):
    def val_fn(point):
        return disp
    return val_fn

# disps = np.linspace(0., 0.01*Ly, 101)
disps = 0.01*Ly*np.hstack((np.linspace(0, 0.6, 21),
                           np.linspace(0.6, 0.8, 121),
                           np.linspace(0.8, -0.4, 61),
                           np.linspace(-0.4, 0.8, 61),
                           np.linspace(0.8, 1., 121)))

location_fns = [y_min, y_min, y_min, y_max, y_max, y_max]
vecs = [0, 1, 2, 0, 1, 2]
value_fns = [dirichlet_val, dirichlet_val, dirichlet_val, 
             dirichlet_val, get_dirichlet_load(disps[0]), dirichlet_val]
dirichlet_bc_info = [location_fns, vecs, value_fns]

problem_u = Elasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
sol_u = onp.zeros((len(mesh.points), 3))
sol_u_old = onp.array(sol_u)
history = onp.zeros((problem_u.num_cells, problem_u.num_quads))
history_old = onp.array(history)
```

Start the major loop for loading steps
```python
simulation_flag = False
if simulation_flag:
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)
    
    vtk_path = os.path.join(vtk_dir, f"u_{0:05d}.vtu")
    save_sol(problem_d, sol_d, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('history', np.mean(history, axis=1))])

    tractions = [0.]
    for i, disp in enumerate(disps[1:]):
        print(f"\nStep {i} in {len(disps)}, disp = {disp}")
    
        err = 1.
        tol = 1e-5
        while err > tol:
            print(f"####### max history = {np.max(history)}")
            problem_u.set_params([sol_d, disp])
            sol_u = solver(problem_u, use_petsc=False)
    
            problem_d.set_params(history)
            sol_d = solver(problem_d, use_petsc=False)
    
            history = problem_u.compute_history(sol_u, history_old)
            sol_d = onp.maximum(sol_d, sol_d_old)
    
            err_u = onp.linalg.norm(sol_u - sol_u_old)
            err_d = onp.linalg.norm(sol_d - sol_d_old)
            err = onp.maximum(err_u, err_d)
            sol_u_old = onp.array(sol_u)
            sol_d_old = onp.array(sol_d)
            print(f"####### err = {err}, tol = {tol}")
            
            # Technically, we are not doing the real 'staggered' scheme. This is an early stop strategy.
            # Comment the following two lines out to get the real staggered scheme, which is more  computationally demanding.
            if True:
                break
    
        history_old = onp.array(history)
     
        traction = problem_u.compute_traction(y_max, sol_u, sol_d)/Lz
        tractions.append(traction[-1])
        print(f"Traction force = {traction}")
        vtk_path = os.path.join(vtk_dir, f"u_{i + 1:05d}.vtu")
        save_sol(problem_d, sol_d, vtk_path, point_infos=[('u', sol_u)], cell_infos=[('history', np.mean(history, axis=1))])
    
    tractions = np.array(tractions)
    
    results = np.stack((disps, tractions))
    np.save(os.path.join(numpy_dir, 'results.npy'), results)
    
else:
    results = np.load(os.path.join(numpy_dir, 'results.npy'))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(results[0], results[1], color='red', marker='o', markersize=4, linestyle='-') 
    plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
    plt.ylabel(r'Force on top surface [kN]', fontsize=20)
    plt.tick_params(labelsize=18)
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(1e3*onp.hstack((0., onp.cumsum(np.abs(np.diff(results[0]))))), results[0], color='blue', marker='o', markersize=4, linestyle='-') 
    plt.xlabel(r'Time [s]', fontsize=20)
    plt.ylabel(r'Displacement of top surface [mm]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()
```

## Execution
Run
```bash
python -m demos.fem.phase_field_fracture.example
```
from the `jax-am/` directory.


## Results

Results can be visualized with *ParaWiew*.

<p align="middle">
  <img src="materials/fracture.gif" width="700" />
</p>
<p align="middle">
    <em >Deformation (x10)</em>
</p>

<p align="middle">
  <img src="materials/time_disp.png" width="500" /> 
  <img src="materials/disp_force.png" width="500" />
</p>
<p align="middle">
    <em >Loading history and tensile force</em>
</p>


## References

[1] Miehe, Christian, Martina Hofacker, and Fabian Welschinger. "A phase field model for rate-independent crack propagation: Robust algorithmic implementation based on operator splits." *Computer Methods in Applied Mechanics and Engineering* 199.45-48 (2010): 2765-2778.

[2] Xue, Tianju, Sigrid Adriaenssens, and Sheng Mao. "Mapped phase field method for brittle fracture." *Computer Methods in Applied Mechanics and Engineering* 385 (2021): 114046.

[3] Miehe, Christian, and Matthias Lambrecht. "Algorithms for computation of stresses and elasticity moduli in terms of Seth–Hill's family of generalized strain tensors." *Communications in numerical methods in engineering* 17.5 (2001): 337-353.
