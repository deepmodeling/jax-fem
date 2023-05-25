```python
import jax
import jax.numpy as np
import os

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh


class Poisson(FEM):
    def get_tensor_map(self):
        return lambda x: x
```

$$
\begin{align*}
\nabla u \cdot \nabla v =
\end{align*}
$$



