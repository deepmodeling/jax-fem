"""Testing the ad_wrappers
1. Check original ad_wrapper against finite differences
2. Check new ad_wrapper_jvp against finite differences
    upto second order (time consuming!)
    Only test reverse mode since it is derived from
    the specified forward mode
3. Check that hessian-vector product is not nan
4. ToDO: Original ad-wrapper has issues in computing accurate gradeints
Check_grads reference:
https://github.com/google/jax/blob/fe30d3fd4bbf92b890c97a75c4c47f4275ab77d1/jax/_src/public_test_util.py#L249
"""
import pytest
import jax
import jax.numpy as np
from jax.test_util import check_grads
from tests_for_fem.elasticity2d_code import Elasticity
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh
from jax_am.fem.solver import ad_wrapper
from jax_am.fem.autodiff_utils import ad_wrapper_jvp

_A_TOL_GRADIENT = 1e-2

def test_ad_wrapper_jvp():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 20., 10.
    meshio_mesh = rectangle_mesh(Nx=20, Ny=10, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
    def load_location(point):
        return np.logical_and(np.isclose(point[0], Lx, atol=1e-5),
                              np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))
    def dirichlet_val(point):
        return 0.
    def neumann_val(point):
        return np.array([0., -100.])

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
    neumann_bc_info = [[load_location], [neumann_val]]
    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type,
                         dirichlet_bc_info=dirichlet_bc_info,
                         neumann_bc_info=neumann_bc_info)
    ######################################
    # Case A: NO petsc - Linear
    fwd_orig = ad_wrapper(problem, linear=True, use_petsc=False)
    fwd_new = ad_wrapper_jvp(problem, linear=True, use_petsc=False)
    input_params = np.ones((200, 1))*0.5

    def objective(params, use_original=True):
        if use_original:
            return np.sum(fwd_orig(params))
        else:
            return np.sum(fwd_new(params))

    # -------------Test baseline - Original ad_wrapper
    print("*******Checking original against finite differences...")
    check_grads(objective, (np.ones((200, 1))*0.5, ),
                order=1, modes=['rev'], eps=1e-4,
                atol=_A_TOL_GRADIENT)

    # Check second-order gradients - Time consuming (~5 min)
    # Change to new ad_wrapper as default
    def objective(params, use_original=False):
        if use_original:
            return np.sum(fwd_orig(params))
        else:
            return np.sum(fwd_new(params))

    print("*******Checking new against finite differences upto 2nd order...")
    check_grads(objective, (np.ones((200, 1))*0.5, ),
                order=2, modes=['rev'], eps=1e-4,
                atol=_A_TOL_GRADIENT)

    # -------------For hessian-vector product
    new_obj = lambda params: objective(params, False)
    hvp = jax.jvp(jax.grad(new_obj), (input_params, ), ((input_params,)))[1]
    assert np.any(np.isnan(hvp)) == False

    ######################################

    # Case B: With petsc - Linear
    fwd_orig = ad_wrapper(problem, linear=True, use_petsc=True)
    fwd_new = ad_wrapper_jvp(problem, linear=True, use_petsc=True)
    input_params = np.ones((200, 1))*0.5

    def objective(params, use_original=True):
        if use_original:
            return np.sum(fwd_orig(params))
        else:
            return np.sum(fwd_new(params))

    # -------------Test baseline - Original ad_wrapper
    print("*******Checking original for PETSc against finite differences...")
    check_grads(objective, (np.ones((200, 1))*0.5, ),
                order=1, modes=['rev'], eps=1e-4,
                atol=_A_TOL_GRADIENT)

    # Check second-order gradients - Time consuming (~5 min)
    # Change to new ad_wrapper as default
    def objective(params, use_original=False):
        if use_original:
            return np.sum(fwd_orig(params))
        else:
            return np.sum(fwd_new(params))

    print("*******Checking new for PETSc against finite differences upto 2nd order...")
    check_grads(objective, (np.ones((200, 1))*0.5, ),
                order=2, modes=['rev'], eps=1e-4,
                atol=_A_TOL_GRADIENT)

    # -------------For hessian-vector product
    new_obj = lambda params: objective(params, False)
    hvp = jax.jvp(jax.grad(new_obj), (input_params, ), ((input_params,)))[1]
    assert np.any(np.isnan(hvp)) == False


# test_ad_wrapper_jvp()