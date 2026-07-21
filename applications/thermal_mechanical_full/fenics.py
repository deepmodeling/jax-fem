import os

from dolfin import *
from mshr import *
import numpy as onp
import matplotlib.pyplot as plt

input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')

# Define domain and mesh
L = 1.
R = 0.1
N = 50  # mesh density

domain = Rectangle(Point(0., 0.), Point(L, L)) - Circle(Point(0., 0.), R, 100)
mesh = generate_mesh(domain, N)

# Define parameters
T0 = Constant(293.) # ambient temperature
DThole = Constant(10.) # temperature change at hole boundary
E = 70e3
nu = 0.3
lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu))
rho = Constant(2700.)     # density
alpha = 2.31e-5  # thermal expansion coefficient
kappa = Constant(alpha*(2*mu + 3*lmbda))
cV = Constant(910e-6)*rho  # specific heat per unit volume at constant strain
k = Constant(237e-6)  # thermal conductivity

# Build function space
Vue = VectorElement('CG', mesh.ufl_cell(), 1) # displacement finite element
Vte = FiniteElement('CG', mesh.ufl_cell(), 1) # temperature finite element
V = FunctionSpace(mesh, MixedElement([Vue, Vte]))

# Boundary condition
def inner_boundary(x, on_boundary):
    return near(x[0]**2+x[1]**2, R**2, 1e-3) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def left(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom) 
bc2 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc3 = DirichletBC(V.sub(1), DThole, inner_boundary) 
bcs = [bc1, bc2, bc3]

# Define variational problem
U_ = TestFunction(V)
(u_, Theta_) = split(U_)
dU = TrialFunction(V)
(du, dTheta) = split(dU)
Uold = Function(V)
(uold, Thetaold) = split(Uold)

def eps(v):
    return sym(grad(v))

def sigma(v, Theta):
    return (lmbda*tr(eps(v)) - kappa*Theta)*Identity(2) + 2*mu*eps(v)

dt = Constant(0.)
mech_form = inner(sigma(du, dTheta), eps(u_))*dx
therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +
              kappa*T0*tr(eps(du-uold))/dt*Theta_ +
              dot(k*grad(dTheta), grad(Theta_)))*dx
form = mech_form + therm_form

# Compute solution
Nincr = 200
t = onp.logspace(1, 4, Nincr+1)
U = Function(V)
for (i, dti) in enumerate(onp.diff(t)):
    print("Increment " + str(i+1))
    dt.assign(dti)
    solve(lhs(form) == rhs(form), U, bcs)
    Uold.assign(U)

(u, theta) = U.split(True)

# Save solution in VTK format
# ufile_pvd = File(os.path.join(output_dir, "vtk/fenics_u.pvd"))
# u.rename("u", "u")
# ufile_pvd << u
# tfile_pvd = File(os.path.join(output_dir, "vtk/fenics_theta.pvd"))
# theta.rename("theta", "theta")
# tfile_pvd << theta

print(f"Max u = {onp.max(u.vector()[:])}, Min u = {onp.min(u.vector()[:])}")
print(f"Max theta = {onp.max(theta.vector()[:])}, Min p = {onp.min(theta.vector()[:])}")

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