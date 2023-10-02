import numpy as onp
import os
from dolfin import *

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) \
                    and on_boundary)


# Create mesh and finite element
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Create Dirichlet boundary condition
u0 = Constant(1.0)
dbc = DirichletBoundary()
bc0 = DirichletBC(V, u0, dbc)

# Collect boundary conditions
bcs = [bc0]

# Define variational problem
u = Function(V)
v = TestFunction(V)
F = dot(grad(u), grad(v))*dx + 5*u**2 *v*ds
solve(F == 0, u, bcs)
print(f"solution max = {onp.max(u.vector().get_local())}, min = {onp.min(u.vector().get_local())}")

u.rename("u", "u")
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_file = os.path.join(data_dir, f"vtk/u_fenics.pvd")
File(vtk_file) << u 
