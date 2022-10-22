import scipy
import numpy as onp

points = onp.random.normal(size=(1000000, 3))
kd_tree = scipy.spatial.KDTree(points)

for i in range(len(points)):
    if i % 10000 == 0:
        print(f"i = {i}")
    # print(f"query point = {points[i]}")
    neighbors = kd_tree.query(points[i], 10)
    # print(neighbors)