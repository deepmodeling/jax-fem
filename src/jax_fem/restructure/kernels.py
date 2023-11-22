"""
Here, we will define the kernels used within the solver. These should be
operators that represent a piece of physics, just like in MOOSE:
https://mooseframework.inl.gov/syntax/Kernels/
"""
import jax
import jax.numpy as np

from typing import NamedTuple
import numpy as np
from abc import ABC, abstractmethod

from typing import NamedTuple
import numpy as np
from abc import ABC, abstractmethod


class KernelData(NamedTuple):
    """
    NamedTuple to store and pass data to Kernel methods.

    Attributes
    ----------
    u : np.ndarray
        Value of the variable the kernel operates on.
    grad_u : np.ndarray
        Gradient of the variable.
    test : np.ndarray
        Value of the test functions.
    grad_test : np.ndarray
        Gradient of the test functions.
    phi : np.ndarray
        Value of the trial functions.
    grad_phi : np.ndarray
        Gradient of the trial functions.
    q_point : np.ndarray
        XYZ coordinates of the current quadrature point.
    current_elem : object
        Pointer to the current element being operated on.
    """


class Kernel(ABC):
    """
    Base class for defining kernels in the FEM solver.
    Kernels represent operators or terms in the weak form of a PDE.

    Methods to be overridden in subclasses:
    - compute_qp_residual: Computes the residual at a quadrature point.
    - compute_qp_jacobian (optional): Computes the Jacobian at a quadrature point.
    - compute_qp_offdiag_jacobian (optional): Computes the off-diagonal Jacobian at a quadrature point.

    Attributes
    ----------
    mesh : object
        The mesh object associated with the FEM problem.
    shape_functions : object
        Shape functions associated with the elements of the mesh.
    quadrature_points : object
        Quadrature points for numerical integration.
    """

    def __init__(self, mesh, shape_functions, quadrature_points):
        """
        Initialize the kernel with mesh, shape functions, and quadrature points.

        Parameters
        ----------
        mesh : object
            The mesh object associated with the FEM problem.
        shape_functions : object
            Shape functions associated with the elements of the mesh.
        quadrature_points : object
            Quadrature points for numerical integration.
        """
        self.mesh = mesh
        self.shape_functions = shape_functions
        self.quadrature_points = quadrature_points

    @abstractmethod
    def compute_qp_residual(self, data: KernelData) -> np.ndarray:
        """
        Compute the residual at a quadrature point. Must be overridden in subclasses.

        Parameters
        ----------
        data : KernelData
            NamedTuple containing all necessary data for the computation.

        Returns
        -------
        np.ndarray
            Residual value at the quadrature point.
        """
        pass

    # Similar structure for compute_qp_jacobian and compute_qp_offdiag_jacobian


# def laplace_kernel(
#     cell_sol,
#     cell_shape_grads,
#     cell_v_grads_JxW,
#     vec,
#     dim,
#     tensor_map,
#     *cell_internal_vars
# ):
#     # Note: maybe it's easier to feed u_grads directly instead of calcuating
#     # them here?

#     u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
#     u_grads = np.sum(u_grads, axis=1)
#     u_grads_reshape = u_grads.reshape(-1, vec, dim)
#     u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(
#         u_grads.shape
#     )
#     val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
#     return val

# etc. for other kernels
