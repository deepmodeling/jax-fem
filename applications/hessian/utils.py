import jax.numpy as np


def compute_l2_norm_error(problem, sol_list_pred, sol_list_true):
    u_pred_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_pred[0]) # (num_cells, num_quads, vec)
    u_true_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_true[0]) # (num_cells, num_quads, vec)
    l2_error = np.sqrt(np.sum((u_pred_quad - u_true_quad)**2 * problem.fes[0].JxW[:, :, None]))
    return l2_error
