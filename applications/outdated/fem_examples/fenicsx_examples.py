import numpy as np
import dolfinx
from dolfinx import fem, io, mesh, plot, log
import ufl
from ufl import ds, dx, grad, inner
import basix
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import time
import meshio
import sys
import os

from jax_fem.generate_mesh import cylinder_mesh_gmsh 

comm = MPI.COMM_WORLD

data_dir = os.path.join(os.path.dirname(__file__), 'data')


def mpi_print(msg):
    if comm.rank == 0:
        print(f"Rank {comm.rank} print: {msg}")
        sys.stdout.flush()


def get_dog_bone_file_path(file_type, index):
    abaqus_root = os.path.join(data_dir, f'abaqus')
    abaqus_files = ['DogBone_mesh6_disp10.inp',
                    'DogBone_mesh2_disp10.inp',
                    'DogBone_mesh1_disp10.inp',
                    'DogBone_mesh05_disp10.inp',
                    'DogBone_mesh03_disp10.inp']

    xdmf_root = os.path.join(data_dir, f'xdmf')
    xdmf_files = ['DogBone_mesh6_disp10.xdmf',
                  'DogBone_mesh2_disp10.xdmf',
                  'DogBone_mesh1_disp10.xdmf',
                  'DogBone_mesh05_disp10.xdmf',
                  'DogBone_mesh03_disp10.xdmf']

    if file_type == 'abaqus':
        return os.path.join(abaqus_root, abaqus_files[index])
    else:
        return os.path.join(xdmf_root, xdmf_files[index])


def get_cylinder_file_path():
    return os.path.join(data_dir, f"msh/cylinder.xdmf")


def generate_xdmf():
    cell_type = 'hexahedron'
    for i in range(5):
        abaqus_file = get_dog_bone_file_path('abaqus', i)
        meshio_mesh = meshio.read(abaqus_file)
        cells = meshio_mesh.get_cells_type(cell_type)
        out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
        xdmf_file = get_dog_bone_file_path('xdmf', i) 
        out_mesh.write(xdmf_file)
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = get_cylinder_file_path()
    out_mesh.write(xdmf_file)


def linear_elasticity(disp, case, dog_bone_index=None):
    if case == 'cylinder':
        xdmf_file = get_cylinder_file_path()
    else:
        xdmf_file = get_dog_bone_file_path('xdmf', dog_bone_index) 

    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")
    E = 70e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    V = fem.VectorFunctionSpace(msh, ("CG", 1))
    normal = ufl.FacetNormal(msh)

    if case == 'cylinder':
        def max_loc(x):
            H = 10.
            return np.isclose(x[2], H)

        def min_loc(x):
            return np.isclose(x[2], 0.)
    else:
        abaqus_file = get_dog_bone_file_path('abaqus', dog_bone_index)
        meshio_mesh = meshio.read(abaqus_file)
        min_x = np.min(meshio_mesh.points[:, 0])
        max_x = np.max(meshio_mesh.points[:, 0])
        mpi_print(f"max_x = {max_x}, min_x = {min_x}")
     
        def min_loc(point):
            return np.isclose(point[0], min_x, atol=1e-5)

        def max_loc(point):
            return np.isclose(point[0], max_x, atol=1e-5)

    fdim = msh.topology.dim - 1
    boundary_facets_min = mesh.locate_entities_boundary(msh, fdim, min_loc)
    boundary_facets_max = mesh.locate_entities_boundary(msh, fdim, max_loc)

    marked_facets = boundary_facets_max
    marked_values = np.full(len(boundary_facets_max), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_max[sorted_facets], marked_values[sorted_facets])

    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)

    if case == 'cylinder':
        u_max_loc = np.array([0, 0, disp], dtype=ScalarType)
    else:
        u_max_loc = np.array([disp, 0, 0.], dtype=ScalarType)
    
    bc_max_loc = fem.dirichletbc(u_max_loc, fem.locate_dofs_topological(V, fdim, boundary_facets_max), V)
 
    u_min_loc = np.array([0, 0, 0], dtype=ScalarType)
    bc_min_loc = fem.dirichletbc(u_min_loc, fem.locate_dofs_topological(V, fdim, boundary_facets_min), V)
 
    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    f = fem.Function(V)
    L = ufl.dot(f, v)*dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_min_loc, bc_max_loc], 
                                      petsc_options={"ksp_type": "bicg", "pc_type": "jacobi", 
                                      "ksp_rtol": 1e-10, "ksp_atol": 1e-10, "ksp_max_it": 10000})
    start_time = time.time()
    uh = problem.solve()
    end_time = time.time()
    solve_time = end_time - start_time
    mpi_print(f"Time elapsed {solve_time}")

    mpi_print(f"max of sol = {np.max(uh.x.array)}")
    mpi_print(f"min of sol = {np.min(uh.x.array)}") 

    if case == 'cylinder':
        traction = fem.assemble_scalar(fem.form(ufl.dot(sigma(uh), normal)[2]*ds(2)))
        mpi_print(f"traction force is {traction}")
        return traction
    else:
        return solve_time


def hyperelasticity(disp):
    xdmf_file = get_cylinder_file_path()
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")

    E = 1e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    kappa = E/(3.*(1. - 2.*nu))

    V = fem.VectorFunctionSpace(msh, ("CG", 1))

    def boundary_top(x):
        H = 10.
        return np.isclose(x[2], H)

    def boundary_bottom(x):
        return np.isclose(x[2], 0.)

    fdim = msh.topology.dim - 1
    boundary_facets_top = mesh.locate_entities_boundary(msh, fdim, boundary_top)
    boundary_facets_bottom = mesh.locate_entities_boundary(msh, fdim, boundary_bottom)

    marked_facets = boundary_facets_top
    marked_values = np.full(len(boundary_facets_top), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_top[sorted_facets], marked_values[sorted_facets])

    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"} 
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=msh, metadata=metadata)
    normal = ufl.FacetNormal(msh)

    u_top = np.array([0, 0, disp], dtype=ScalarType)
    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, boundary_facets_top), V)

    u_bottom = np.array([0, 0, 0], dtype=ScalarType)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets_bottom), V)
 
    uh = fem.Function(V)
    v = ufl.TestFunction(V)

    d = len(uh)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(uh))
    C = ufl.variable(F.T * F)
    J = ufl.det(F)
    Jinv = J**(-2 / 3)
    I1 = ufl.tr(C)
    energy = energy = ((mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.) 
    P = ufl.diff(energy, F)

    F_res = ufl.inner(ufl.grad(v), P)*dxm

    problem = dolfinx.fem.petsc.NonlinearProblem(F_res, uh, [bc_bottom, bc_top])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    ksp = solver.krylov_solver
    opts = petsc4py.PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "bicg"
    opts[f"{option_prefix}pc_type"] = "none"
    opts[f"{option_prefix}ksp_rtol"] = 1e-10
    opts[f"{option_prefix}ksp_atol"] = 1e-10
    opts[f"{option_prefix}ksp_max_it"] = 10000
    ksp.setFromOptions()
    log.set_log_level(log.LogLevel.INFO)

    start_time = time.time()
    n, converged = solver.solve(uh)
    end_time = time.time()
    solve_time = end_time - start_time
    mpi_print(f"Time elapsed {solve_time}")

    mpi_print(f"max of sol = {np.max(uh.x.array)}")
    mpi_print(f"min of sol = {np.min(uh.x.array)}") 
 
    traction = fem.assemble_scalar(fem.form(ufl.dot(P, normal)[2]*ds(2)))
    mpi_print(f"traction force is {traction}")

    return traction


def plasticity(disps, path):
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    os.makedirs(os.path.join(data_dir, 'msh'), exist_ok=True)
    xdmf_file = os.path.join(data_dir, f"msh/cylinder.xdmf")
    out_mesh.write(xdmf_file)
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")
    E = 70e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    H = 10.

    deg_stress = 2
    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
 
    W_ele = ufl.TensorElement("Quadrature", msh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W = fem.FunctionSpace(msh, W_ele)
    V = fem.VectorFunctionSpace(msh, ("CG", 1))

    def boundary_top(x):
        return np.isclose(x[2], H)

    def boundary_bottom(x):
        return np.isclose(x[2], 0.)

    fdim = msh.topology.dim - 1
    boundary_facets_top = mesh.locate_entities_boundary(msh, fdim, boundary_top)
    boundary_facets_bottom = mesh.locate_entities_boundary(msh, fdim, boundary_bottom)

    marked_facets = boundary_facets_top
    marked_values = np.full(len(boundary_facets_top), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_top[sorted_facets], marked_values[sorted_facets])

    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=msh, metadata=metadata)

    u_bottom = np.array([0, 0, 0], dtype=ScalarType)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets_bottom), V)
 
    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def elastic_stress(eps):
        return lmbda * ufl.tr(eps) * ufl.Identity(u_crt.geometric_dimension()) + 2*mu*eps

    def quad_interpolation(v, V):
        '''
        See https://github.com/FEniCS/dolfinx/issues/2243
        '''
        quadrature_points, wts = basix.make_quadrature(basix.CellType.hexahedron, deg_stress)
        u = fem.Function(V)
        e_expr = fem.Expression(v, quadrature_points)
        map_c = msh.topology.index_map(msh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        e_eval = e_expr.eval(cells)

        with u.vector.localForm() as u_local:
            u_local.setBlockSize(u.function_space.dofmap.bs)
            u_local.setValuesBlocked(V.dofmap.list.array, e_eval, addv=PETSc.InsertMode.INSERT)
        return u

    sig = fem.Function(W)
    eps = fem.Function(W)

    EPS = 1e-10
    ppos = lambda x: ufl.conditional(ufl.gt(x, 0.1), x, 0.)
    heaviside = lambda x: ufl.conditional(ufl.gt(x, 0.), x, EPS)

    def stress_fn(u_crt):
        sig0 = 250.
        sig_elas = elastic_stress(epsilon(u_crt) - eps) + sig
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(heaviside(3/2.*ufl.inner(s, s)))
        f_elas = sig_eq - sig0
        # Prevent divided by zero error
        # The original example (https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html)
        # didn't consider this, and can cause nan error in the solver.
        new_sig = sig_elas - ppos(f_elas)*s/(sig_eq + EPS)
        return new_sig

    x = ufl.SpatialCoordinate(msh)
    u_crt = fem.Function(V)
    v = ufl.TestFunction(V)
    F_res = ufl.inner(stress_fn(u_crt), epsilon(v)) * dxm

    u_vol = fem.Function(V)
    u_vol.x.array[:] = 1.
    vol = fem.assemble_scalar(fem.form(u_vol[0]*dxm))

    start_time = time.time()
    avg_stresses = []

    for i, disp in enumerate(disps):
        # Remark(Tianju): "problem" should be better defined outside of the for loop, 
        # but I wasn't able to find a way to assign Dirichlet values on the top boundary.
        mpi_print(f"\nStep {i} in {len(disps)}")

        u_top = np.array([0, 0, disp], dtype=ScalarType)
        bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, boundary_facets_top), V)

        # No initial guess from previous step
        u_crt.x.array[:] = 0.

        problem = dolfinx.fem.petsc.NonlinearProblem(F_res, u_crt, [bc_bottom, bc_top])
        solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

        # solver.max_it = 0
        # solver.atol = 1.
        ksp = solver.krylov_solver
        opts = petsc4py.PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "bicg"
        opts[f"{option_prefix}pc_type"] = "none"
        opts[f"{option_prefix}ksp_rtol"] = 1e-10
        opts[f"{option_prefix}ksp_atol"] = 1e-10
        opts[f"{option_prefix}ksp_max_it"] = 10000
        ksp.setFromOptions()
        log.set_log_level(log.LogLevel.INFO)

        n, converged = solver.solve(u_crt)
        new_sig = stress_fn(u_crt)
        sig.x.array[:] = quad_interpolation(new_sig, W).x.array

        eps.x.array[:] = quad_interpolation(epsilon(u_crt), W).x.array
        avg_stress_sig = fem.assemble_scalar(fem.form(sig[2, 2]*dxm))/vol
        mpi_print(f"avg_stress_sig = {avg_stress_sig}")
        avg_stresses.append(avg_stress_sig)

    end_time = time.time()

    solve_time = end_time - start_time
    mpi_print(f"Time elapsed {solve_time}")
    mpi_print(f"max of sol = {np.max(u_crt.x.array)}")
    mpi_print(f"min of sol = {np.min(u_crt.x.array)}") 

    avg_stresses = np.array(avg_stresses)

    mpi_print(f"Volume averaged stress = {avg_stresses}")

    sig.x.array[:] = quad_interpolation(new_sig, W).x.array
    avg_stress_quad = fem.assemble_scalar(fem.form(sig[2, 2]*dxm))/vol
    avg_stress_node = fem.assemble_scalar(fem.form(stress_fn(u_crt)[2, 2]*dxm))/vol
    mpi_print(avg_stress_quad)
    mpi_print(avg_stress_node)

    u_crt.name = 'sol'

    np.save(os.path.join(path, 'numpy/plasticity/fenicsx/forces.npy'), avg_stresses)
    np.save(os.path.join(path, 'numpy/plasticity/fenicsx/disps.npy'), disps)
    file = io.VTKFile(msh.comm, os.path.join(path, 'vtk/plasticity/fenicsx/sol.pvd'), "w")

    file.write_function(u_crt, 0) 
    return solve_time


def performance_test():
    solve_time = []
    for i in range(4):
        wall_time = linear_elasticity(10., 'dog_bone', i)
        solve_time.append(wall_time)
    solve_time = np.array(solve_time)
    np.savetxt(os.path.join(data_dir, f"txt/fenicsx_fem_time.txt"), solve_time, fmt='%.3f')


def generate_fem_examples():
    plasticity_disps = np.hstack((np.linspace(0., 0.1, 11), np.linspace(0.09, 0., 10)))
    plasticity_path =  f"applications/fem/fem_examples/data/"
    plasticity(plasticity_disps, plasticity_path)

    linear_elasticity_disps = np.linspace(0., 0.1, 11)
    tractions = []
    for disp in linear_elasticity_disps:
        traction = linear_elasticity(disp, 'cylinder')
        tractions.append(traction)
    tractions = np.array(tractions)
    np.save(os.path.join(data_dir, f'numpy/linear_elasticity/fenicsx/disps.npy'), linear_elasticity_disps)
    np.save(os.path.join(data_dir, f'numpy/linear_elasticity/fenicsx/forces.npy'), tractions)

    hyperelasticity_disps = np.linspace(0., 2., 11)
    tractions = []
    for disp in hyperelasticity_disps:
        traction = hyperelasticity(disp)
        tractions.append(traction)
    tractions = np.array(tractions)
    np.save(os.path.join(data_dir, f'numpy/hyperelasticity/fenicsx/disps.npy'), hyperelasticity_disps)
    np.save(os.path.join(data_dir, f'numpy/hyperelasticity/fenicsx/forces.npy'), tractions)


def exp():
    # linear_elasticity(1., 'cylinder')
    # linear_elasticity(10., 'dog_bone', 1)
    hyperelasticity(1.)

if __name__ == "__main__":
    # generate_xdmf()
    # generate_fem_examples()
    performance_test()
    # exp()

