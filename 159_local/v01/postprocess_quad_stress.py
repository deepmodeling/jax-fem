import argparse
import os
import re

import numpy as onp


STRESS_COMPONENTS = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("xy", 0, 1),
    ("yz", 1, 2),
    ("xz", 0, 2),
)
STRESS_FIELD_RE = re.compile(r"^stress_quad(?P<quad>\d*)_(?P<component>xx|yy|zz|xy|yz|xz)$")
VM_FIELD_RE = re.compile(r"^vm_quad(?P<quad>\d*)$")


def _as_array(cell_data_value):
    if isinstance(cell_data_value, list):
        if len(cell_data_value) != 1:
            raise ValueError("Only single-cell-block VTU files are supported")
        return onp.asarray(cell_data_value[0])
    return onp.asarray(cell_data_value)


def _quad_id(text):
    return 0 if text == "" else int(text)


def extract_quad_stress(cell_data):
    stress_fields = {}
    vm_fields = {}
    for name, value in cell_data.items():
        stress_match = STRESS_FIELD_RE.match(name)
        if stress_match:
            quad_idx = _quad_id(stress_match.group("quad"))
            stress_fields.setdefault(quad_idx, {})[stress_match.group("component")] = _as_array(value)
            continue
        vm_match = VM_FIELD_RE.match(name)
        if vm_match:
            vm_fields[_quad_id(vm_match.group("quad"))] = _as_array(value)

    quad_ids = sorted(stress_fields)
    if not quad_ids:
        raise ValueError("No stress_quad* fields found")
    if quad_ids != sorted(vm_fields):
        raise ValueError("stress_quad* and vm_quad* fields use different quadrature ids")

    num_cells = len(next(iter(stress_fields[quad_ids[0]].values())))
    stress_quad = onp.zeros((num_cells, len(quad_ids), 3, 3), dtype=onp.float64)
    vm_quad = onp.zeros((num_cells, len(quad_ids)), dtype=onp.float64)

    for out_idx, quad_idx in enumerate(quad_ids):
        components = stress_fields[quad_idx]
        missing = [name for name, _, _ in STRESS_COMPONENTS if name not in components]
        if missing:
            raise ValueError(f"Missing stress components for quad {quad_idx}: {missing}")
        for name, row, col in STRESS_COMPONENTS:
            arr = onp.asarray(components[name], dtype=onp.float64)
            if arr.shape != (num_cells,):
                raise ValueError(f"Field stress_quad{quad_idx}_{name} has shape {arr.shape}; expected {(num_cells,)}")
            stress_quad[:, out_idx, row, col] = arr
            stress_quad[:, out_idx, col, row] = arr
        vm_arr = onp.asarray(vm_fields[quad_idx], dtype=onp.float64)
        if vm_arr.shape != (num_cells,):
            raise ValueError(f"Field vm_quad{quad_idx} has shape {vm_arr.shape}; expected {(num_cells,)}")
        vm_quad[:, out_idx] = vm_arr

    return stress_quad, vm_quad


def summarize_quad_stress(stress_quad, vm_quad):
    stress_quad = onp.asarray(stress_quad, dtype=onp.float64)
    vm_quad = onp.asarray(vm_quad, dtype=onp.float64)
    return {
        "stress_mean": onp.mean(stress_quad, axis=1),
        "von_mises_mean": onp.mean(vm_quad, axis=1),
        "von_mises_max": onp.max(vm_quad, axis=1),
        "von_mises_p95": onp.percentile(vm_quad, 95.0, axis=1),
    }


def recover_nodal_averaged_cell_data(cells, cell_data, num_nodes):
    cells = onp.asarray(cells, dtype=onp.int64)
    data = onp.asarray(cell_data, dtype=onp.float64)
    nodal_sum = onp.zeros((num_nodes,) + data.shape[1:], dtype=onp.float64)
    nodal_count = onp.zeros((num_nodes,), dtype=onp.float64)
    for local_node in range(cells.shape[1]):
        onp.add.at(nodal_sum, cells[:, local_node], data)
        onp.add.at(nodal_count, cells[:, local_node], 1.0)
    shape = (num_nodes,) + (1,) * len(data.shape[1:])
    nodal_count = onp.maximum(nodal_count.reshape(shape), 1.0)
    return nodal_sum / nodal_count


def derived_cell_data(stress_stats, include_compat_aliases=True):
    stress_mean = stress_stats["stress_mean"]
    fields = {
        "von_mises_mean": stress_stats["von_mises_mean"],
        "von_mises_max": stress_stats["von_mises_max"],
        "von_mises_p95": stress_stats["von_mises_p95"],
    }
    for name, row, col in STRESS_COMPONENTS:
        fields[f"stress_{name}_mean"] = stress_mean[:, row, col]
        if include_compat_aliases:
            fields[f"stress_{name}"] = stress_mean[:, row, col]
    if include_compat_aliases:
        fields["von_mises"] = stress_stats["von_mises_mean"]
    return fields


def derived_point_data(cells, num_nodes, stress_stats):
    stress_nodal = recover_nodal_averaged_cell_data(cells, stress_stats["stress_mean"], num_nodes)
    vm_nodal = recover_nodal_averaged_cell_data(cells, stress_stats["von_mises_mean"], num_nodes)
    fields = {"recovered_von_mises_mean": vm_nodal}
    for name, row, col in STRESS_COMPONENTS:
        fields[f"recovered_stress_{name}_mean"] = stress_nodal[:, row, col]
    return fields


def postprocess_mesh(mesh, include_compat_aliases=True):
    if len(mesh.cells) != 1:
        raise ValueError("Only single-cell-block VTU files are supported")
    stress_quad, vm_quad = extract_quad_stress(mesh.cell_data)
    stress_stats = summarize_quad_stress(stress_quad, vm_quad)
    for name, values in derived_cell_data(stress_stats, include_compat_aliases).items():
        mesh.cell_data[name] = [onp.asarray(values, dtype=onp.float32)]
    for name, values in derived_point_data(mesh.cells[0].data, len(mesh.points), stress_stats).items():
        mesh.point_data[name] = onp.asarray(values, dtype=onp.float32)
    return mesh


def output_path_for(input_path, output_dir, suffix):
    base, ext = os.path.splitext(os.path.basename(input_path))
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    return os.path.join(output_dir, f"{base}{suffix}{ext}")


def postprocess_file(input_path, output_path=None, output_dir=None, suffix="_post", include_compat_aliases=True):
    import meshio

    mesh = meshio.read(input_path)
    postprocess_mesh(mesh, include_compat_aliases=include_compat_aliases)
    if output_path is None:
        output_path = output_path_for(input_path, output_dir, suffix)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.write(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate derived stress fields from raw stress_quad*/vm_quad* VTU fields.")
    parser.add_argument("inputs", nargs="+", help="Input .vtu files written by inp_thermal_stress_oneway_xbuild_p0p1_fixed.py")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--suffix", default="_post")
    parser.add_argument("--no-compat-aliases", dest="compat_aliases", action="store_false")
    args = parser.parse_args()
    for input_path in args.inputs:
        out = postprocess_file(
            input_path,
            output_dir=args.output_dir,
            suffix=args.suffix,
            include_compat_aliases=args.compat_aliases,
        )
        print(out)


if __name__ == "__main__":
    main()
