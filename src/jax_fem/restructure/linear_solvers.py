"""
This folder contains the linear solvers used in solving the FEM problems.
Lineax is used here as a common interface for JAX and PETSc linear solvers.

TODO 1: Wrap the PETSc solver, using petsc4py and a pure_callback.
TODO -1: Add documentation, consistent typing and tests.
"""

from lineax import AbstractLinearSolver, AbstractLinearOperator
from typing import Any, Optional

# This structure is inspired by:
# https://github.com/google/lineax/issues/44
# and the concrete implementation for GMRES:
# https://github.com/google/lineax/blob/main/lineax/_solver/gmres.py


class PETScSolver(AbstractLinearSolver):
    rtol: float
    atol: float

    # Ensure that inputs are correct
    def __check_init__(self):
        # Perform checks on the inputs such as rtol, atol
        pass

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        """Perform initial computations on the operator, e.g. convert it to a
        PETSc compatible format.

        Parameters
        ----------
        operator : AbstractLinearOperator
            Linear operator (A in Ax=b)
        options : dict[str, Any]
            Dictionary of extra options to pass to the solver (here we can pass
            the PETSc options like preconditioner, ksp_type, etc.)

        Returns
        -------
        ~_SolverState - a state object that will be passed to methods.

        Note: The solver state should be immutable, and be designed so that it
        can keep the inputs that we do not wish to recalculate when trying to
        solve for a new vector.
        """

        pass

    def compute(self, state, vector, options):
        """
        For documentation, see:
        https://docs.kidger.site/lineax/api/solvers/#lineax.AbstractLinearSolver
        """
        pass

        # Eventually, it should return a tuple like described above


# There is no need to define an extra solver for JAX, since we can use the ones
# that are built-in with Lineax. However, we need to ensure that all the inputs
# are in the correct format.

# There is a clear need for preconditioners - these should also be implemented
# in here. For pure lineax solve, we may use preconditioners defined as
# AbstractLinearOperators (see docstring for GMRES), for PETSc, we can just
# pass the options to the solver. For a smoother experience, we may want to
# have a check for available Lineax preconditioners.
# We should simply implement the Jacobi preconditioner for now to match the
# existing JAX_FEM code.

# Functions from original solver.py that use the linear solver, e.g.
# linear_guess_solve(), should be in a separate file - this folder is
# reserved for the linear solvers backend.
