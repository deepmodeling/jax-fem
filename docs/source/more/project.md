# About the project

*By Tianju Xue*

_JAX-FEM_ is a light-weight Finite Element Method library in pure Python, accelerated with [_JAX_](https://github.com/google/jax). This folder contains tutorial examples with explanatory comments. The software is still at an experimental stage. 

In the following paragraphs, I want to share some of my motivations on developing the software: How is _JAX-FEM_ different from other FEM codes? What's new? Who cares? What should users expect and NOT expect from this software?

## Life Is Short, Use Python

My first exposure to open-source FEM library was [_Dealii_](https://www.dealii.org/), a powerful C++ software library that allows users to build FEM codes to solve a broad variety of PDEs. While I enjoyed very much the flexibility of Dealii, a significant amount of my time was indeed spent on writing lengthy C++ code that easily became challenging for debugging and maintaining. 

My second choice was [_FEniCS_](https://fenicsproject.org/) (now _FEniCSx_), an amazing FEM library with high-level Python interfaces. The beauty of _FEniCS_ is that users write near-math code in Python, and immediately solve their (possibly nonlinear) problems, with highly competitive performance due to the C++ backend. Yet, the use of automatic (symbolic) differentiation by _FEniCS_ comes with a price: it becomes cumbersome for complicated constitutive relationships. When solving problems of solid mechanics, typically, a mapping from strain to stress needs to be specified. If this mapping can be explicitly expressed with an analytical form, _FEniCS_ works just fine. However, this is not always the case. There are two examples in my field. One is crystal plasticity, where strain is often times related to stress through an implicit function. The other example is the phase field fracture problem, where eigenvalue decomposition for the strain is necessary. After weeks of unsuccessful trials with _FEniCS_, I started the idea of implementing an FEM code myself that handles complicated constitutive relationships, and that became the start of _JAX-FEM_. 

Staying in the Python ecosystem, _JAX_ becomes a natural choice, due to its [outstanding performance for scientific computing workloads](https://github.com/dionhaefner/pyhpc-benchmarks/tree/master). 

## The Magic of Automatic Differentiation

The design of _JAX-FEM_ fundamentally exploits automatic differentiation. The rule of thumb is that whenever there is a derivative to take, let the machine do it. Some typical examples include

1. In a hyperelasticity problem, given strain energy density function $\psi(\boldsymbol F)$, compute first PK stress $\boldsymbol{P}=\frac{\partial \psi}{\partial \boldsymbol{F}}$. 

2. In a plasticity problem, given stress $\boldsymbol{\sigma} (\boldsymbol{\varepsilon}, \boldsymbol{\alpha})$ as a function of strain and some internal variables , compute fourth-order elasto-plastic tangent moduli tensor $\mathbb{C}=\frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}$.
3. In a topology optimization problem, the computation of sensitivity can be fully automatic.

As developers, we are actively using _JAX-FEM_ to solve inverse problems (or PDE-constrained optimizaiton problems) involving complicated constitutive relationships, with thanks to AD that makes this effort easy.

## Native in Machine Learning 

Since _JAX_ itself is a framework for machine learning, _JAX-FEM_ trivially has access to the ecosystem of _JAX_. If you have a material model represented by a neural network, and you want to deploy that model into the computation of FEM, _JAX-FEM_ will be a perfect tool. No need to hard code the neural network coefficients into a Fortran file and run _Abaqus_!

## Heads Up! 

1. **Kernels**. _JAX-FEM_ uses kernels to handle different terms in the FEM weak form, a concept similar as in [_MOOSE_](https://mooseframework.inl.gov/syntax/Kernels/). Currently, we can handle the "Laplace kernel" $\int_{\Omega} f(\nabla u)\cdot \nabla v$ and the "mass kernel" $\int_{\Omega}h(u)v$ in the weak form. This covers solving typical second-order elliptic equations like those occurring in quasi-static solid mechanics, or time-dependent parabolic problems like a heat equation. We also provide a "universal kernel" that lets users define their own weak form. This is a new feature introduced on Dec 11, 2023.

2. **Performance**. In most cases, the majority of computational time is spent on solving the linear system from the Newton's method. If CPU is available, the linear system will be solved by [_PETSc_](https://petsc.org/release/); if GPU is available, solving the linear system with _JAX_ built-in sparse linear solvers will usually be faster and scalable to larger problems. Exploiting multiple CPUs and/or even multiple GPUs is our future work. Please see our _JAX-FEM_ journal paper for performance report.

3. **Memory**. The largest problem that is solved without causing memory insufficiency issue on a 48G memory RTX8000 Nvidia GPU contains around 9 million DOFs. 

4. **Nonlinearity**. _JAX-FEM_ handles material nonlinearity well, but currently does not handle other types of nonlinearities such as contact. Secondary development is needed.

5. **Boundary conditions**. As of now, we cannot handle periodic boundary conditions. We need some help on this.