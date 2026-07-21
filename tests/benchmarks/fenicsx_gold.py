"""
This is FEniCSx code to generate gold standard results used as ground truth to compare with JAX-FEM results.
"""
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

np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)

crt_dir = os.path.dirname(__file__)
data_dir = os.path.join(crt_dir, 'data')


def linear_poisson(N):
    L = 1.
    msh = mesh.create_box(comm=MPI.COMM_WORLD, 
                          points=((0., 0., 0.), (L, L, L)), 
                          n=(N, N, N), 
                          cell_type=mesh.CellType.hexahedron)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))

    def boundary_left(x):
        return np.isclose(x[0], 0)

    def boundary_right(x):
        return np.isclose(x[0], L)

    fdim = msh.topology.dim - 1
    boundary_facets_left = mesh.locate_entities_boundary(msh, fdim, boundary_left)
    boundary_facets_right = mesh.locate_entities_boundary(msh, fdim, boundary_right)
    bc_left = fem.dirichletbc(ScalarType(0.), fem.locate_dofs_topological(V, fdim, boundary_facets_left), V)
    bc_right = fem.dirichletbc(ScalarType(1.), fem.locate_dofs_topological(V, fdim, boundary_facets_right), V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    a = inner(grad(u), grad(v)) * dx
    f = fem.Function(V)
    L = f*v*dx
    # To set petsc_options, see https://jorgensd.github.io/dolfinx-tutorial/chapter4/solvers.html
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left, bc_right], 
                                      petsc_options={"ksp_type": "bicg", "pc_type": "none", 
                                      "ksp_rtol": 1e-10, "ksp_atol": 1e-10, "ksp_max_it": 10000})
    start_time = time.time()
    uh = problem.solve()
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")
    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 
    uh.name = 'sol'
    file = io.VTKFile(msh.comm, os.path.join(crt_dir, "linear_poisson/fenicsx/sol.pvd"), "w")  
    file.write_function(uh, 0) 
    return solve_time


def nonlinear_poisson(N):
    """Deprecated.
    """
    L = 1.
    msh = mesh.create_box(comm=MPI.COMM_WORLD, 
                          points=((0., 0., 0.), (L, L, L)), 
                          n=(N, N, N), 
                          cell_type=mesh.CellType.hexahedron)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))

    def boundary_left(x):
        return np.isclose(x[0], 0)

    def boundary_right(x):
        return np.isclose(x[0], L)

    fdim = msh.topology.dim - 1
    boundary_facets_left = mesh.locate_entities_boundary(msh, fdim, boundary_left)
    boundary_facets_right = mesh.locate_entities_boundary(msh, fdim, boundary_right)
    bc_left = fem.dirichletbc(ScalarType(0.), fem.locate_dofs_topological(V, fdim, boundary_facets_left), V)
    bc_right = fem.dirichletbc(ScalarType(1.), fem.locate_dofs_topological(V, fdim, boundary_facets_right), V)

    uh = fem.Function(V)
    v = ufl.TestFunction(V)
    F_res = (1 + uh**2)*inner(grad(uh), grad(v))*dx
    problem = dolfinx.fem.petsc.NonlinearProblem(F_res, uh, [bc_left, bc_right])
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
    print(f"Time elapsed {solve_time}")
    print(f"Number of interations: {n:d}")
    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    uh.name = 'sol'
    file = io.VTKFile(msh.comm, os.path.join(crt_dir, "nonlinear_poisson/fenicsx/sol.pvd", "w"))
    file.write_function(uh, 0) 
    return solve_time


def linear_elasticity_cube(N):
    L = 1
    E = 70e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, L, L])],
                      [N,N,N], cell_type=mesh.CellType.hexahedron)
    V = fem.VectorFunctionSpace(msh, ("CG", 1))

    def boundary_left(x):
        return np.isclose(x[0], 0)

    def boundary_right(x):
        return np.isclose(x[0], L)

    fdim = msh.topology.dim - 1
    boundary_facets_left = mesh.locate_entities_boundary(msh, fdim, boundary_left)
    boundary_facets_right = mesh.locate_entities_boundary(msh, fdim, boundary_right)

    marked_facets = boundary_facets_right
    marked_values = np.full(len(boundary_facets_right), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_right[sorted_facets], marked_values[sorted_facets])
    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)

    u_left = np.array([1., 1., 1.], dtype=ScalarType)
    bc_left = fem.dirichletbc(u_left, fem.locate_dofs_topological(V, fdim, boundary_facets_left), V)

    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, ScalarType((0, 10., 10.)))
    t = fem.Constant(msh, ScalarType((10., 0., 0.)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(t, v) * ds(2)

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left], 
                                      petsc_options={"ksp_type": "bicg", "pc_type": "none", 
                                      "ksp_rtol": 1e-10, "ksp_atol": 1e-10, "ksp_max_it": 10000})

    start_time = time.time()
    uh = problem.solve()
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    uh.name = 'sol'
    file = io.VTKFile(msh.comm, os.path.join(crt_dir, "linear_elasticity_cube/fenicsx/sol.pvd"), "w")  
    file.write_function(uh, 0)
    return solve_time


def linear_elasticity_cylinder():
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = os.path.join(data_dir, f"msh/cylinder.xdmf")
    out_mesh.write(xdmf_file)
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")
    E = 70e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    H = 10.
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

    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)

    u_top = np.array([0, 0, 1], dtype=ScalarType)
    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, boundary_facets_top), V)
 
    u_bottom = np.array([0, 0, 0], dtype=ScalarType)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets_bottom), V)
 
    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    x = ufl.SpatialCoordinate(msh)
    f = ufl.as_vector((1e3*x[0], 2e3*x[1], 3e3*x[2]))
    t = ufl.as_vector((1e3*x[0]**2 + 1e3*x[1]**2, 0., 0.))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(t, v) * ds(2)

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_bottom], 
                                      petsc_options={"ksp_type": "bicg", "pc_type": "none", 
                                      "ksp_rtol": 1e-10, "ksp_atol": 1e-10, "ksp_max_it": 10000})
    start_time = time.time()
    uh = problem.solve()
    uh.name = 'sol'
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    file = io.VTKFile(msh.comm, os.path.join(crt_dir, "linear_elasticity_cylinder/fenicsx/sol.pvd"), "w")  
    file.write_function(uh, 0) 

    uh = fem.Function(V)
    uh.x.array[:] = 1.
    surface_area = fem.assemble_scalar(fem.form(uh[0]*ds(2)))
    np.save(os.path.join(crt_dir, "linear_elasticity_cylinder/fenicsx/surface_area.npy"), surface_area)

    return solve_time


def hyperelasticity():
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = os.path.join(data_dir, "msh/cylinder.xdmf")
    out_mesh.write(xdmf_file)
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")
 
    E = 1e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    kappa = E/(3.*(1. - 2.*nu))

    H = 10.

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

    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"} 
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=msh, metadata=metadata)
    normal = ufl.FacetNormal(msh)

    u_top = np.array([0, 0, 1.], dtype=ScalarType)
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
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    traction = fem.assemble_scalar(fem.form(ufl.dot(P, normal)[2]*ds(2)))
    print(f"traction = {traction}")
    np.save(os.path.join(crt_dir, "hyperelasticity/fenicsx/traction.npy"), traction)

    uh.name = 'sol'
    file = io.VTKFile(msh.comm, os.path.join(crt_dir, "hyperelasticity/fenicsx/sol.pvd"), "w")  
    file.write_function(uh, 0) 
    return solve_time


def plasticity(disps, path, case):
    meshio_mesh = cylinder_mesh_gmsh(data_dir)
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = os.path.join(data_dir, "msh/cylinder.xdmf")
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
        print(f"\nStep {i} in {len(disps)}")

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
        print(f"avg_stress_sig = {avg_stress_sig}")
        avg_stresses.append(avg_stress_sig)

    end_time = time.time()

    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")
    print(f"max of sol = {np.max(u_crt.x.array)}")
    print(f"min of sol = {np.min(u_crt.x.array)}") 

    avg_stresses = np.array(avg_stresses)

    print(f"Volume averaged stress = {avg_stresses}")

    sig.x.array[:] = quad_interpolation(new_sig, W).x.array
    avg_stress_quad = fem.assemble_scalar(fem.form(sig[2, 2]*dxm))/vol
    avg_stress_node = fem.assemble_scalar(fem.form(stress_fn(u_crt)[2, 2]*dxm))/vol
    print(avg_stress_quad)
    print(avg_stress_node)

    u_crt.name = 'sol'

    if case == 'test':
        np.save(os.path.join(path, 'avg_stresses.npy'), avg_stresses)
        np.save(os.path.join(path, 'disps.npy'), disps)
        file = io.VTKFile(msh.comm, os.path.join(path, 'sol.pvd'), "w")
    else:
        np.save(os.path.join(path, 'numpy/plasticity/fenicsx/avg_stresses.npy'), avg_stresses)
        np.save(os.path.join(path, 'numpy/plasticity/fenicsx/disps.npy'), disps)
        file = io.VTKFile(msh.comm, os.path.join(path, 'vtk/plasticity/fenicsx/sol.pvd'), "w")

    file.write_function(u_crt, 0) 
    return solve_time


def generate_ground_truth_results_for_tests():
    linear_poisson(10)
    linear_elasticity_cube(10)
    linear_elasticity_cylinder()
    hyperelasticity()
    plasticity(np.array([0., 0.05, 0.1, 0.05, 0.]), os.path.join(crt_dir, f"plasticity/fenicsx/"), 'test')


if __name__ == "__main__":
    generate_ground_truth_results_for_tests()
