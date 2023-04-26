import fenics as fe
import os
from dolfin import *

data_path = os.path.join(os.path.dirname(__file__), 'data') 


# --------------------
# Functions and classes
# --------------------
# Strain function
def epsilon(u):
    return 0.5*(fe.nabla_grad(u) + fe.nabla_grad(u).T)

# Stress function
def sigma(u):
    return lmbda*fe.div(u)*fe.Identity(3) + 2*mu*epsilon(u)

# --------------------
# Parameters
# --------------------
# Young modulus, poisson number and density
E, nu = 70.0E9, 0.23
rho = 2500.0

# Lame's constants
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

l_x, l_y, l_z = 1.0, 1.0, 0.01  # Domain dimensions
n_x, n_y, n_z = 20, 20, 2  # Number of elements


# l_x, l_y, l_z = 1.0, 1.0, 1.  # Domain dimensions
# n_x, n_y, n_z = 2, 2, 2  # Number of elements




# --------------------
# Geometry
# --------------------
mesh = fe.BoxMesh(fe.Point(0.0, 0.0, 0.0), fe.Point(l_x, l_y, l_z), n_x, n_y, n_z)


# mesh = Mesh()
# filename = os.path.join(data_path, f'xdmf/jax_mesh.xdmf')
# f = XDMFFile(MPI.comm_world, filename)
# f.read(mesh)



# --------------------
# Function spaces
# --------------------
V = fe.VectorFunctionSpace(mesh, "CG", 1)
u_tr = fe.TrialFunction(V)
u_test = fe.TestFunction(V)

# --------------------
# Forms & matrices
# --------------------
a_form = fe.inner(sigma(u_tr), epsilon(u_test))*fe.dx
m_form = rho*fe.inner(u_tr, u_test)*fe.dx

A = fe.PETScMatrix()
M = fe.PETScMatrix()
A = fe.assemble(a_form, tensor=A)
M = fe.assemble(m_form, tensor=M)

# --------------------
# Eigen-solver
# --------------------
eigensolver = fe.SLEPcEigenSolver(A, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
# eigensolver.parameters["spectrum"] = "smallest real"
eigensolver.parameters["spectrum"] = "target magnitude"
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 1.

N_eig = 12   # number of eigenvalues
eigensolver.solve(N_eig)

# --------------------
# Post-process
# --------------------
# Set up file for exporting results
file_results = fe.XDMFFile(os.path.join(data_path, "xdmf/MA.xdmf"))
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Eigenfrequencies
for i in range(0, N_eig):
    # Get i-th eigenvalue and eigenvector
    # r - real part of eigenvalue
    # c - imaginary part of eigenvalue
    # rx - real part of eigenvector
    # cx - imaginary part of eigenvector
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # Calculation of eigenfrequency from real part of eigenvalue
    # freq_3D = fe.sqrt(r)/2/fe.pi
    freq_3D = r
    print("Eigenfrequency: {0:8.5f} [Hz]".format(freq_3D))

    # Initialize function and assign eigenvector
    eigenmode = fe.Function(V, name="Eigenvector" + str(i))
    eigenmode.vector()[:] = rx

    # Write i-th eigenfunction to xdmf file
    file_results.write(eigenmode, i)


vtkfile = File(os.path.join(data_path, f'xdmf/mesh.pvd'))
vtkfile << eigenmode
