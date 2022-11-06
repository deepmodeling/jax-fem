import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import meshio
import time

from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data_dir = os.path.join(os.path.dirname(__file__), 'data') 


def fn():

 
    meshio_mesh = meshio.read(os.path.join(data_dir, 'msh/part.msh'))

    print(os.path.join(data_dir, 'msh/part.msh'))
    print(meshio_mesh)

    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={'tetra10': meshio_mesh.cells_dict['tetra10']})

    # out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={'tetra': meshio_mesh.cells_dict['tetra']})


    print(len(out_mesh.points))
 
    out_mesh.write(os.path.join(data_dir, 'vtk/bracket.vtu'))


if __name__ == "__main__":
    fn()