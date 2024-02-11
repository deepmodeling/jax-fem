import os

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as onp

input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')

# Define domain and mesh
Lx , Ly = 1., 1.
Nx, Ny = 100, 100
mesh = RectangleMesh(Point(0, 0), Point(1,1), Nx, Ny, 'left')
plot(mesh)
# Define parameters
dt = 1/250000 # temporal sampling interval
c = 5000 # speed of sound
steps = 200

# Build function space
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary conditions 
bcs = DirichletBC(V, Constant(1.), "on_boundary") # Pure Dirichlet boundary conditions

# Define variational problem
u0 = interpolate(Constant(0.0), V) # u_old_2dt
u1 = interpolate(Constant(0.0), V) # u_old_dt

u = TrialFunction(V)
v = TestFunction(V)

a = inner(u, v) * dx + Constant(dt**2 * c**2) * inner(grad(u), grad(v)) * dx
L = (2*u1 - u0) * v * dx

# Compute solution
u = Function(V)
for n in range(steps):
    solve(a==L, u, bcs)
    u0.assign(u1)
    u1.assign(u)

print(f"Max u = {onp.max(u.vector()[:])}, Min u = {onp.min(u.vector()[:])}")

# Save points and cells for the use of JAX-FEM
# Build function space
V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V_fs = FunctionSpace(mesh, V)

points = V_fs.tabulate_dof_coordinates()
print(f"points.shape = {points.shape}")

cells_v = []
dofmap_v = V_fs.dofmap()
for cell in cells(mesh):
    dof_index = dofmap_v.cell_dofs(cell.index())
    # print(cell.index(), dof_index)
    cells_v.append(dof_index)
cells = onp.stack(cells_v)
print(f"cells.shape = {cells.shape}")

numpy_dir = os.path.join(input_dir, f'numpy/')
if not os.path.exists(numpy_dir): os.makedirs(numpy_dir)
onp.save(os.path.join(input_dir, f'numpy/points.npy'), points)
onp.save(os.path.join(input_dir, f'numpy/cells.npy'), cells)