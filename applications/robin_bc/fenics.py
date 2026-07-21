import numpy as onp
import os
from dolfin import *

input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')

class LeftorRight(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (near(x[0], 0) or near(x[0], 1))

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Create mesh and finite element
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
boundaries.set_all(0)

# Now mark your Neumann boundary
rightbound = LeftorRight()
rightbound.mark(boundaries, 1)
ds=Measure('ds')[boundaries]


# Collect boundary conditions
bc = DirichletBC(V, Constant(1.0), boundary)
bcs = [bc]

f = Expression("x[0]*sin(5.0*pi*x[1]) + 1.0*exp(-((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5))/0.02)", degree=3)

# Define variational problem
u = Function(V)
v = TestFunction(V)

# F = dot(grad(u), grad(v))*dx + 5*u**2 *v*ds(1)
F = dot(grad(u), grad(v))*dx -f*v*dx + 5*u**2 *v*ds(1)
solve(F == 0, u, bcs)
print(f"solution max = {onp.max(u.vector().get_local())}, min = {onp.min(u.vector().get_local())}")

u.rename("u", "u")

vtk_file = os.path.join(output_dir, f"vtk/u_fenics.pvd")
File(vtk_file) << u 

# Save points and cells for the use of JAX-FEM
# Build function space
V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V_fs = FunctionSpace(mesh, V)

points_u = V_fs.tabulate_dof_coordinates()
print(f"points_u.shape = {points_u.shape}")

cells_u = []
dofmap = V_fs.dofmap()
for cell in cells(mesh):
    dof_index = dofmap.cell_dofs(cell.index())
    # print(cell.index(), dof_index)
    cells_u.append(dof_index)
cells_u = onp.stack(cells_u)
print(f"cells_u.shape = {cells_u.shape}")

onp.save(os.path.join(input_dir, f'numpy/points_u.npy'), points_u)
onp.save(os.path.join(input_dir, f'numpy/cells_u.npy'), cells_u)

