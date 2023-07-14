from jaxopt import GaussNewton, LevenbergMarquardt, ScipyRootFinding
import scipy

from jax_am.fem.solver import get_flatten_fn, apply_bc


def external_solve(problem, initial_guess, params):
    def constraint_fn(dofs, pms):
        """c(u, p)
        """
        problem.set_params(pms)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    # gn = LevenbergMarquardt(residual_fun=constraint_fn, verbose=True)
    # gn_sol = gn.run(initial_guess.reshape(-1), pms=params).params


    gn = ScipyRootFinding(optimality_fun=constraint_fn, method='lm')
    gn_sol = gn.run(initial_guess.reshape(-1), pms=params).params


    return gn_sol.reshape(initial_guess.shape)


