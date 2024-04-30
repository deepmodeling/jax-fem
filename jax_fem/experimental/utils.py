import meshio
import json
import os
import numpy as onp

from jax_fem.generate_mesh import get_meshio_cell_type


def save_sol(fe, sol, sol_file, cell_infos=None, point_infos=None):
    cell_type = get_meshio_cell_type(fe.ele_type)
    sol_dir = os.path.dirname(sol_file)
    os.makedirs(sol_dir, exist_ok=True)
    out_mesh = meshio.Mesh(points=fe.points, cells={cell_type: fe.cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    if cell_infos is not None:
        for cell_info in cell_infos:
            name, data = cell_info
            # TODO: vector-valued cell data
            assert data.shape == (fe.num_cells,), f"cell data wrong shape, get {data.shape}, while num_cells = {fe.num_cells}"
            out_mesh.cell_data[name] = [onp.array(data, dtype=onp.float32)]
    if point_infos is not None:
        for point_info in point_infos:
            name, data = point_info
            assert len(data) == len(sol), "point data wrong shape!"
            out_mesh.point_data[name] = onp.array(data, dtype=onp.float32)
    out_mesh.write(sol_file)


def modify_vtu_file(input_file_path, output_file_path):
    """Convert version 2.2 of vtu file to version 1.0
    meshio does not accept version 2.2, raising error of
    meshio._exceptions.ReadError: Unknown VTU file version '2.2'.
    """
    fin = open(input_file_path, "r")
    fout = open(output_file_path, "w")
    for line in fin:
        fout.write(line.replace('<VTKFile type="UnstructuredGrid" version="2.2">', '<VTKFile type="UnstructuredGrid" version="1.0">'))
    fin.close()
    fout.close()


def read_abaqus_and_write_vtk(abaqus_file, vtk_file):
    """Used for a quick inspection. Paraview can't open .inp file so we convert it to .vtu
    """
    meshio_mesh = meshio.read(abaqus_file)
    meshio_mesh.write(vtk_file)


def json_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args