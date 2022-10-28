import jax
import jax.numpy as np
import numpy as onp
import meshio
import gmsh
import os

from jax_am.fem.jax_fem import Mesh, LinearElasticity
from jax_am.fem.solver import solver
# from jax_am.fem.generate_mesh import box_mesh
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, cell_type, lag_order, data_dir):
    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'box_order_2.msh')
    generate = True
    if generate:
        offset_x = 0.
        offset_y = 0.
        offset_z = 0.
        domain_x = Lx
        domain_y = Ly
        domain_z = Lz

        gmsh.initialize()
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
        if cell_type.startswith('tetra'):
            Rec2d = False  # tris or quads
            Rec3d = False  # tets, prisms or hexas
        else:
            Rec2d = True
            Rec3d = True
        p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
        l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
        s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
        v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(lag_order)
        gmsh.write(msh_file)
        gmsh.finalize()
      
    mesh = meshio.read(msh_file)
    points = mesh.points # (num_total_nodes, dim)

    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})

    return out_mesh


def problem():
    """Can be used to test the memory limit of JAX-FEM
    """
    problem_name = f'linear_elasticity'
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data') # applications/fem/debug/data

    ele_type = 'tetrahedron'
    lag_order = 2

    if ele_type == 'tetrahedron' and lag_order == 1:
        cell_type = 'tetra'
    elif ele_type == 'tetrahedron' and lag_order == 2:
        cell_type = 'tetra10'
    elif ele_type == 'hexahedron' and lag_order == 1:
        cell_type = 'hexahedron'
    elif ele_type == 'hexahedron' and lag_order == 2:
        cell_type = 'hexahedron27'
    else:
        raise NotImplementedError

    meshio_mesh = box_mesh(10, 10, 10, 1., 1., 1., cell_type, lag_order, data_dir)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # inp_dir = os.path.join(data_dir, 'inp')
    # os.makedirs(inp_dir, exist_ok=True)
    # inp_file = os.path.join(inp_dir, 'box_order_2.inp')
    # meshio_mesh.write(inp_file)
    # abaqus_tet = meshio.read(os.path.join(inp_dir, 'tet.inp'))
    # print(meshio_mesh.points[meshio_mesh.cells_dict[cell_type][0]])
    # exit()

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
 
    problem = LinearElasticity(problem_name, mesh, ele_type=ele_type, lag_order=lag_order, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, linear=True, precond=True)
    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    vtk_file = os.path.join(vtk_dir, f"{problem_name}/u.vtu")
    save_sol(problem, sol, vtk_file, cell_type=cell_type)

    physical_quad_points = problem.get_physical_quad_points()

    # print(physical_quad_points[0])

if __name__ == "__main__":
    problem()
