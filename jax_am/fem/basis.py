import basix
import numpy as onp


def get_full_integration_poly_degree(ele_type, lag_order, dim):
    """Only works for weak forms of (grad_u, grad_v)
    Reference:
    https://zhuanlan.zhihu.com/p/521630645
    """
    if ele_type == 'hexahedron' or ele_type == 'quadrilateral':
        return 2 * (dim*lag_order - 1)

    if ele_type == 'tetrahedron' or ele_type == 'triangle':
        return 2 * (dim*(lag_order - 1) - 1)

def get_elements(ele_type, lag_order):
    """Order is different for basix and meshio
    References
    https://github.com/FEniCS/basix
    http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_65.php
    """
    if (ele_type, lag_order) == ('hexahedron', 1):
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 2x2x2
    elif (ele_type, lag_order) == ('hexahedron', 2):

        # re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 15, 
        #             14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 
                    17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26]

        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 10 # 6x6x6, full integration
    elif (ele_type, lag_order) == ('tetrahedron', 1):
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0 # 1, full integration
    elif (ele_type, lag_order) == ('tetrahedron', 2):
        # re_order = [0, 1, 2, 3, 9, 6, 8, 7, 4, 5]
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2 # 4, full integration
    else:
        raise NotImplementedError

    return basix_ele, basix_face_ele, gauss_order, re_order


def reorder_inds(inds, re_order):
    new_inds = []
    for ind in inds.reshape(-1): 
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type, lag_order):
    """TODO: Add comments

    Returns
    ------- 
    shape_values: ndarray
        (8, 8) = (num_quads, num_nodes)
    shape_grads_ref: ndarray
        (8, 8, 3) = (num_quads, num_nodes, dim)
    weights: ndarray
        (8,) = (num_quads,)
    """
    basix_ele, basix_face_ele, gauss_order, re_order = get_elements(ele_type, lag_order)
    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)  
    lagrange = basix.create_element(basix.ElementFamily.P, basix_ele, lag_order, basix.LagrangeVariant.equispaced)
    vals_and_grads = lagrange.tabulate(1, quad_points)[:, :, re_order, :]
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    print(f"ele_type = {ele_type}, lag_order = {lag_order}, quad_points.shape= {quad_points.shape}")
    return shape_values, shape_grads_ref, weights


def get_face_shape_vals_and_grads(ele_type, lag_order):
    """TODO: Add comments

    Returns
    ------- 
    face_shape_vals: ndarray
        (6, 4, 8) = (num_faces, num_face_quads, num_nodes)
    face_shape_grads_ref: ndarray
        (6, 4, 3) = (num_faces, num_face_quads, num_nodes, dim)
    face_weights: ndarray
        (6, 4) = (num_faces, num_face_quads)
    face_normals:ndarray
        (6, 3) = (num_faces, dim)
    face_inds: ndarray
        (6, 4) = (num_faces, num_face_quads)
    """
    basix_ele, basix_face_ele, gauss_order, re_order = get_elements(ele_type, lag_order)

    # TODO: Check if this is correct.
    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_lag_order = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_lag_order, basix.LagrangeVariant.equispaced)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    facets = basix.cell.sub_entity_connectivity(basix_ele)[2] 
    # Map face points
    # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    lagrange = basix.create_element(basix.ElementFamily.P, basix_ele, lag_order, basix.LagrangeVariant.equispaced)
    vals_and_grads = lagrange.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))

    print(f"face_quad_points.shape = {face_quad_points.shape}")

    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds


if __name__ == "__main__":
    pass
