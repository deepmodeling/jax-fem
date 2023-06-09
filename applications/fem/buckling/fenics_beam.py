# https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/hyperelasticity/python/documentation.html

from dolfin import *
from ufl import nabla_div
import os
from applications.fem.buckling.dr import DynamicRelaxSolve

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = BoxMesh(Point(0., 0., 0.), Point(20., 1., 1.), 100, 5, 5) 

V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 20.0)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"), degree=1)
r = Expression(("-4", "2.", "0."), degree=1)

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
 
# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)


# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)


def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lmbda*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)
    
F_linear = inner(sigma(u), epsilon(v))*dx

solve(F_linear == 0, u, bcs)


DynamicRelaxSolve(F, u, bcs, J, tol=1e-6)

# Solve variational problem
# solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)

data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Save solution in VTK format
file = File(os.path.join(data_dir, "pvd/displacement.pvd"))
file << u


# Plot and hold solution
# plot(u, mode = "displacement", interactive = True)