from __future__ import print_function
from dolfin import *
import os
import numpy as onp
import matplotlib.pyplot as plt

input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')

# Load mesh and subdomains
mesh = Mesh(os.path.join(input_dir, "xml/dolfin_fine.xml.gz"))
sub_domains = MeshFunction("size_t", mesh, os.path.join(input_dir, "xml/dolfin_fine_subdomains.xml.gz"))


# Build function space
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V * Q)


# No-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
# x0 = 0
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
a = (inner(grad(u), grad(v)) + div(v)*p + q*div(u))*dx
L = inner(f, v)*dx

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

print(f"Max u = {onp.max(u.vector()[:])}, Min u = {onp.min(u.vector()[:])}")
print(f"Max p = {onp.max(p.vector()[:])}, Min p = {onp.min(p.vector()[:])}")

# # Split the mixed solution using a shallow copy
(u, p) = w.split()

# Save solution in VTK format
ufile_pvd = File(os.path.join(output_dir, "vtk/fenics_velocity.pvd"))
u.rename("u", "u")
ufile_pvd << u
pfile_pvd = File(os.path.join(output_dir, "vtk/fenics_pressure.pvd"))
p.rename("p", "p")
pfile_pvd << p

# Save points and cells for the use of JAX-FEM

# Build function space
V = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V_fs = FunctionSpace(mesh, V)
Q_fs = FunctionSpace(mesh, Q)

points_v = V_fs.tabulate_dof_coordinates()
points_p = Q_fs.tabulate_dof_coordinates()
print(f"points_v.shape = {points_v.shape}")
print(f"points_p.shape = {points_p.shape}")

cells_v = []
dofmap_v = V_fs.dofmap()
for cell in cells(mesh):
    dof_index = dofmap_v.cell_dofs(cell.index())
    # print(cell.index(), dof_index)
    cells_v.append(dof_index)
cells_v = onp.stack(cells_v)
print(f"cells_v.shape = {cells_v.shape}")

cells_p = []
dofmap_p = Q_fs.dofmap()
for cell in cells(mesh):
    dof_index = dofmap_p.cell_dofs(cell.index())
    # print(cell.index(), dof_index)
    cells_p.append(dof_index)
cells_p = onp.stack(cells_p)
print(f"cells_p.shape = {cells_p.shape}")

re_order = [0, 1, 2, 5, 3, 4]
cells_v = cells_v[:, re_order]

onp.save(os.path.join(input_dir, f'numpy/points_u.npy'), points_v)
onp.save(os.path.join(input_dir, f'numpy/cells_u.npy'), cells_v)
onp.save(os.path.join(input_dir, f'numpy/points_p.npy'), points_p)
onp.save(os.path.join(input_dir, f'numpy/cells_p.npy'), cells_p)

# The dof order now should follow JAX-FEM (same as Abaqus)
# https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/stm/default.htm?startat=ch03s02ath64.html
# But the dof can be clockwise, which needs further modification
selected_p = points_v[cells_v[6]]

plt.plot(selected_p[0, 0], selected_p[0, 1], marker='o', color='red')
plt.plot(selected_p[1, 0], selected_p[1, 1], marker='o', color='blue')
plt.plot(selected_p[2, 0], selected_p[2, 1], marker='o', color='orange')
plt.plot(selected_p[3, 0], selected_p[3, 1], marker='s', color='red')
plt.plot(selected_p[4, 0], selected_p[4, 1], marker='s', color='blue')
plt.plot(selected_p[5, 0], selected_p[5, 1], marker='s', color='orange')

plt.show()
