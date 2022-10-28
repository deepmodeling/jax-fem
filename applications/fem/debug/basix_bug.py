import basix
import numpy as onp

lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron, 2, basix.LagrangeVariant.equispaced)
point = onp.array([[0., 0.5, 0.]])
values = lagrange.tabulate(0, point)[0, 0, :, 0]
print(values)
print(onp.argwhere(values > 1e-10))
