import os
import gmsh
import numpy as onp
import meshio


def box_mesh(Nx, Ny, Nz, Lx=1., Ly=1., Lz=1.):
    """References:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/hex.py
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_1/tutorial/python/t1.py
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_1/tutorial/python/t3.py
    """
    mesh_file = f"jax_am/fem/data/msh/box.msh"
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
        Rec2d = True  # tris or quads
        Rec3d = True  # tets, prisms or hexas
        p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
        l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
        s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
        v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_file)
        gmsh.finalize()
      
    mesh = meshio.read(mesh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict['hexahedron'] # (num_cells, num_nodes)

    out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return out_mesh


def cylinder_mesh(R=5, H=10, circle_mesh=5, hight_mesh=20, rect_ratio=0.4):
    """By Xinxin Wu at PKU in July, 2022
    Reference: https://www.researchgate.net/post/How_can_I_create_a_structured_mesh_using_a_transfinite_volume_in_gmsh
    R: radius
    H: hight
    circle_mesh:num of meshs in circle lines
    hight_mesh:num of meshs in hight
    rect_ratio: rect length/R
    """
    mesh_name = f"jax_am/fem/data/msh/cylinder"
    rect_coor = R*rect_ratio
    geo_file = mesh_name + ".geo"
    mesh_file = mesh_name + ".msh"
    
    string='''
        Point(1) = {{0, 0, 0, 1.0}};
        Point(2) = {{-{rect_coor}, {rect_coor}, 0, 1.0}};
        Point(3) = {{{rect_coor}, {rect_coor}, 0, 1.0}};
        Point(4) = {{{rect_coor}, -{rect_coor}, 0, 1.0}};
        Point(5) = {{-{rect_coor}, -{rect_coor}, 0, 1.0}};
        Point(6) = {{{R}*Cos(3*Pi/4), {R}*Sin(3*Pi/4), 0, 1.0}};
        Point(7) = {{{R}*Cos(Pi/4), {R}*Sin(Pi/4), 0, 1.0}};
        Point(8) = {{{R}*Cos(-Pi/4), {R}*Sin(-Pi/4), 0, 1.0}};
        Point(9) = {{{R}*Cos(-3*Pi/4), {R}*Sin(-3*Pi/4), 0, 1.0}};

        Line(1) = {{2, 3}};
        Line(2) = {{3, 4}};
        Line(3) = {{4, 5}};
        Line(4) = {{5, 2}};
        Line(5) = {{2, 6}};
        Line(6) = {{3, 7}};
        Line(7) = {{4, 8}};
        Line(8) = {{5, 9}};

        Circle(9) = {{6, 1, 7}};
        Circle(10) = {{7, 1, 8}};
        Circle(11) = {{8, 1, 9}};
        Circle(12) = {{9, 1, 6}};

        Curve Loop(1) = {{1, 2, 3, 4}};
        Plane Surface(1) = {{1}};
        Curve Loop(2) = {{1, 6, -9, -5}};
        Plane Surface(2) = {{2}};
        Curve Loop(3) = {{2, 7, -10, -6}};
        Plane Surface(3) = {{3}};
        Curve Loop(4) = {{3, 8, -11, -7}};
        Plane Surface(4) = {{4}};
        Curve Loop(5) = {{4, 5, -12, -8}};
        Plane Surface(5) = {{5}};

        Transfinite Curve {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} = {circle_mesh} Using Progression 1;

        Transfinite Surface {{1}};
        Transfinite Surface {{2}};
        Transfinite Surface {{3}};
        Transfinite Surface {{4}};
        Transfinite Surface {{5}};
        Recombine Surface {{1, 2, 3, 4, 5}};

        Extrude {{0, 0, {H}}} {{
          Surface{{1:5}}; Layers {{{hight_mesh}}}; Recombine;
        }}

        Mesh 3;'''.format(R=R, H=H, rect_coor=rect_coor, circle_mesh=circle_mesh, hight_mesh=hight_mesh, mesh_file=mesh_file)

    with open(geo_file, "w") as f:
        f.write(string)
    os.system("gmsh -3 {geo_file} -o {mesh_file} -format msh2".format(geo_file=geo_file, mesh_file=mesh_file))

    mesh = meshio.read(mesh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict['hexahedron'] # (num_cells, num_nodes)

    # The mesh somehow has two redundant points...
    points = onp.vstack((points[1:14], points[15:]))
    cells = onp.where(cells > 14, cells - 2, cells - 1)

    # print(points[:10])
    # print(f"Number of total vertices = {len(points)}")
    # print(f"Number of total cells = {len(cells)}")
    # print(onp.take(points, cells[:3], axis=0))

    out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return out_mesh


if __name__=="__main__":
    cylinder_mesh(R=5, H=10, circle_mesh=5, hight_mesh=20, rect_ratio=0.4)
