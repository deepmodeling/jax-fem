# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

from numpy import array
from petsc4py import PETSc
from slepc4py import SLEPc

class HyperelasticityProblem(BifurcationProblem):
    def mesh(self, comm):
        return RectangleMesh(comm, Point(0, 0), Point(1, 0.1), 40, 40)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 1)

        # Construct rigid body modes used in algebraic multigrid preconditioner later on
        rbms = [Constant((0, 1)),
                Constant((1, 0)),
                Expression(("-x[1]", "x[0]"), degree=1, mpi_comm=mesh.mpi_comm())]
        self.rbms = [interpolate(rbm, V) for rbm in rbms]

        return V

    def parameters(self):
        eps = Constant(0)

        return [(eps, "eps", r"$\epsilon$")]

    def residual(self, u, params, v):
        B   = Constant((0.0, -1000)) # Body force per unit volume

        # Kinematics
        I = Identity(2)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        # Elasticity parameters
        E, nu = 1000000.0, 0.3
        mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

        # Total potential energy
        Energy = psi*dx - dot(B, u)*dx #- dot(T, u)*ds
        F = derivative(Energy, u, v)

        return F

    def boundary_conditions(self, V, params):
        eps = params[0]
        left  = CompiledSubDomain("(std::abs(x[0])       < DOLFIN_EPS) && on_boundary", mpi_comm=V.mesh().mpi_comm())
        right = CompiledSubDomain("(std::abs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary", mpi_comm=V.mesh().mpi_comm())

        bcl = DirichletBC(V, (0.0,  0.0), left)
        bcr = DirichletBC(V, (-eps, 0.0), right)

        return [bcl, bcr]

    def functionals(self):
        def pointeval(u, params):
            return u((0.25, 0.05))[1]

        return [(pointeval, "pointeval", r"$u_1(0.25, 0.05)$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant((0, 0)), V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of eps.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        eps = params[0]
        if eps < 0.03:
            return 1
        if eps < 0.07:
            return 3
        if eps < 0.12:
            return 5
        if eps < 0.18:
            return 7
        if eps < 0.20:
            return 9
        return float("inf")

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

    def solver(self, problem, params, solver_params, prefix="", **kwargs):
        # Set the rigid body modes for use in AMG

        s = SNUFLSolver(problem, solver_parameters=solver_params, prefix=prefix, **kwargs)
        snes = s.snes
        snes.setFromOptions()

        if snes.ksp.type != "preonly":
            # Convert rigid body modes (computed in self.function_space above) to PETSc Vec
            rbms = list(map(vec, self.rbms))

            # Create the PETSc nullspace
            nullsp = PETSc.NullSpace().create(vectors=rbms, constant=False, comm=snes.comm)

            (A, P) = snes.ksp.getOperators()
            A.setNearNullSpace(nullsp)
            P.setNearNullSpace(nullsp)

        return s

    def compute_stability(self, params, branchid, u, hint=None):
        V = u.function_space()
        trial = TrialFunction(V)
        test  = TestFunction(V)

        bcs = self.boundary_conditions(V, params)
        comm = V.mesh().mpi_comm()

        F = self.residual(u, map(Constant, params), test)
        J = derivative(F, u, trial)
        b = inner(Constant((1, 0)), test)*dx # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = PETScMatrix(comm)
        asm = SystemAssembler(J, b, bcs)
        asm.assemble(A)

        # Build the mass matrix for the RHS of the generalised eigenproblem
        B = PETScMatrix(comm)
        asm = SystemAssembler(inner(test, trial)*dx, b, bcs)
        asm.assemble(B)
        [bc.zero(B) for bc in bcs]

        # Create the SLEPc eigensolver
        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(A.mat(), B.mat())
        eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
        eps.setProblemType(eps.ProblemType.GHEP)
        eps.setFromOptions()

        # If we have a hint, use it - it's the eigenfunctions from the previous solve
        if hint is not None:
            initial_space = [vec(x) for x in hint]
            eps.setInitialSpace(initial_space)

        if eps.st.ksp.type != "preonly":
            # Convert rigid body modes (computed in self.function_space above) to PETSc Vec
            rbms = map(vec, self.rbms)

            # Create the PETSc nullspace
            nullsp = PETSc.NullSpace().create(vectors=rbms, constant=False, comm=comm)

            (A, P) = eps.st.ksp.getOperators()
            A.setNearNullSpace(nullsp)
            P.setNearNullSpace(nullsp)

        # Solve the eigenproblem
        eps.solve()

        eigenvalues = []
        eigenfunctions = []
        eigenfunction = Function(V, name="Eigenfunction")

        for i in range(eps.getConverged()):
            lmbda = eps.getEigenvalue(i)
            assert lmbda.imag == 0
            eigenvalues.append(lmbda.real)

            eps.getEigenvector(i, vec(eigenfunction))
            eigenfunctions.append(eigenfunction.copy(deepcopy=True))

        if min(eigenvalues) < 0:
            is_stable = False
        else:
            is_stable = True

        d = {"stable": is_stable,
             "eigenvalues": eigenvalues,
             "eigenfunctions": eigenfunctions,
             "hint": eigenfunctions}

        return d

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_atol": 1.0e-7,
               "snes_rtol": 1.0e-10,
               "snes_max_linear_solve_fail": 100,
               "snes_monitor": None,
               "snes_converged_reason": None,
               "ksp_type": "gmres",
               "ksp_monitor_cancel": None,
               "ksp_converged_reason": None,
               "ksp_max_it": 2000,
               "pc_type": "gamg",
               "pc_factor_mat_solver_package": "mumps",
               "pc_factor_mat_solver_type": "mumps",
               "eps_type": "krylovschur",
               "eps_target": -1,
               "eps_monitor_all": None,
               "eps_converged_reason": None,
               "eps_nev": 1,
               "st_type": "sinvert",
               "st_ksp_type": "preonly",
               "st_pc_type": "lu",
               "st_pc_factor_mat_solver_package": "mumps",
               "st_pc_factor_mat_solver_type": "mumps",
               }


if __name__ == "__main__":
    dc = DeflatedContinuation(problem=HyperelasticityProblem(), teamsize=1, verbose=True, clear_output=True)
    params = list(arange(0.0, 0.2, 0.001)) + [0.2]
    dc.run(values={"eps": params})