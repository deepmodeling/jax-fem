"""Not working! This file is also outdated. We need some help of implementing periodci B.C.
"""
import jax
import jax.numpy as np
import os
import meshio
import gmsh

from jax_fem.models import LinearPoisson
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh_gmsh, get_meshio_cell_type
from jax_fem.utils import save_sol, modify_vtu_file
from jax_fem.basis import get_elements


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def gmsh_mesh(data_dir, degree):
    """
    Reference:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_8_3/tutorial/python/t1.py
    """
    msh_dir = os.path.join(data_dir, f'msh')
    os.makedirs(msh_dir, exist_ok=True)
    file_path = os.path.join(msh_dir, f't1.msh')

    gmsh.initialize()
    gmsh.model.add("t1")
    lc = 1e-2
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1., 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1., 1., 0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0, 1., 0, lc)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, p4, 3)
    gmsh.model.geo.addLine(4, 1, p4)
    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(file_path)

    return file_path


def problem():
    """

    Reference:
    https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/periodic/python/documentation.html
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ele_type = 'TRI6'
    _, _, _, _, degree, _ = get_elements(ele_type)
    msh_file_path = gmsh_mesh(data_dir, degree)
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(msh_file_path)
    # TODO:
    mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict[cell_type])

    def body_force(x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        val = x[0]*np.sin(5.0*np.pi*x[1]) + 1.0*np.exp(-(dx*dx + dy*dy)/0.02)
        return np.array([val])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], 1., atol=1e-5)

    def dirichlet_val(point):
        return 0.

    def mapping_x(point_A):
        point_B = point_A + np.array([1., 0])
        return point_B

    location_fns_A = [left] 
    location_fns_B = [right] 
    mappings = [mapping_x]
    vecs = [0]
    periodic_bc_info = [location_fns_A, location_fns_B, mappings, vecs]

    dirichlet_bc_info = [[bottom, top], 
                         [0]*2, 
                         [dirichlet_val, dirichlet_val]]

    problem = LinearPoisson(mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                            periodic_bc_info=periodic_bc_info, source_info=body_force)

    sol = solver(problem, linear=True, use_petsc=True)

    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_file = os.path.join(vtk_dir, f"u.vtu")
    save_sol(problem, sol, vtk_file)


if __name__ == "__main__":
    problem()
