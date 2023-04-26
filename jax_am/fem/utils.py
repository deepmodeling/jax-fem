import meshio
import os
import numpy as onp


def save_sol(problem, sol, sol_file, cell_infos=None, point_infos=None, cell_type='hexahedron'):
    # TODO: infer cell_type from problem
    sol_dir = os.path.dirname(sol_file)
    os.makedirs(sol_dir, exist_ok=True)
    out_mesh = meshio.Mesh(points=problem.points, cells={cell_type: problem.cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    if cell_infos is not None:
        for cell_info in cell_infos:
            name, data = cell_info
            # TODO: vector-valued cell data
            assert data.shape == (problem.num_cells,), f"cell data wrong shape, get {data.shape}, while num_cells = {problem.num_cells}"
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
