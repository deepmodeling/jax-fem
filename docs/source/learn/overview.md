# Overview

| Example                                                      | Highlight                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Poisson equation](poisson) | ${\color{green}Basics:}$  Poisson's equation in a unit square domain with Dirichlet and Neumann boundary conditions, as well as a source term. |
| [Linear elasticity](linear_elasticity) | ${\color{green}Basics:}$  Bending of a linear elastic beam due to Dirichlet and Neumann boundary conditions. Second order tetrahedral element (TET10) is used. |
| [Hyperelasticity](hyperelasticity) | ${\color{blue}Nonlinear \space Constitutive \space Law:}$ Deformation of a hyperelastic cube due to Dirichlet boundary conditions. |
| [Plasticity](plasticity) | ${\color{blue}Nonlinear \space Constitutive \space Law:}$ Perfect J2-plasticity model is implemented for small deformation theory. |
| [Compute gradients](compute_gradients) | ${\color{red}Inverse \space Problem:}$ Sanity check of how automatic differentiation works. |
| [Topology optimization](topology_optimization) | ${\color{red}Inverse \space Problem:}$ SIMP topology optimization for a 2D beam. Note that sensitivity analysis is done by the program, rather than manual derivation. |
| [Source field identification](source_field_identification/source_field_identification) | ${\color{red}Inverse \space Problem:}$ Gradient of the objective function with respect to the source field term. |
| [Traction force identification](traction_force_identification/traction_force_identification) | ${\color{red}Inverse \space Problem:}$ Gradient of the objective function with respect to the Neumann boundary condition. |
| [Thermal mechanical control](thermal_mechanical_control/thermal_mechanical_control) | ${\color{red}Inverse \space Problem:}$ Gradient of the objective function with respect to the Dirichlet boundary condition. |
| [Shape optimization](shape_optimization/shape_optimization) | ${\color{red}Inverse \space Problem:}$ Gradient of the objective function with respect to a stiffness related term. |