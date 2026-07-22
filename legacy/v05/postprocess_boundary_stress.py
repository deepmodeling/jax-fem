"""v05 boundary-stress postprocessor.

Quantifies edge/interface stress concentration in fast-scan / physfix outputs:
principal stresses, yield utilization (with hardening + saturation), distance
to the lateral free surface, and part-plate interface tractions recovered from
the elastic-foundation springs (t = -k_s * u on base faces).

Physics background: Mercelis & Kruth 2006 (M-shaped height profile),
Parry et al. 2016 (free-edge in-plane concentration), Zaeh & Branner 2010
(interface-perimeter shear/peel driving delamination).
"""
import argparse
import csv
import glob
import os
import re

import meshio
import numpy as onp
from scipy.spatial import cKDTree

TET_FACES = onp.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
COMPS = ("xx", "yy", "zz", "xy", "yz", "xz")


def tensor_from_quads(m, num_cells):
    """Stack per-quad stress tensors -> (num_quads, num_cells, 3, 3)."""
    tensors = []
    q = 0
    while f"stress_quad{q}_xx" in m.cell_data:
        c = {k: onp.asarray(m.cell_data[f"stress_quad{q}_{k}"][0]) for k in COMPS}
        S = onp.zeros((num_cells, 3, 3))
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = c["xx"], c["yy"], c["zz"]
        S[:, 0, 1] = S[:, 1, 0] = c["xy"]
        S[:, 1, 2] = S[:, 2, 1] = c["yz"]
        S[:, 0, 2] = S[:, 2, 0] = c["xz"]
        tensors.append(S)
        q += 1
    return onp.stack(tensors)


def von_mises(S):
    dev = S - onp.trace(S, axis1=-2, axis2=-1)[..., None, None] / 3.0 * onp.eye(3)
    return onp.sqrt(1.5 * onp.sum(dev * dev, axis=(-2, -1)))


def exterior_faces(cells):
    nc = len(cells)
    faces = cells[:, TET_FACES]                        # (nc, 4, 3)
    keys = onp.sort(faces.reshape(nc * 4, 3), axis=1)
    _, inverse, counts = onp.unique(keys, axis=0, return_inverse=True, return_counts=True)
    ext = (counts[inverse] == 1)
    face_nodes = faces.reshape(nc * 4, 3)[ext]
    owner_cell = onp.repeat(onp.arange(nc), 4)[ext]
    return face_nodes, owner_cell


def analyze(vtu_path, out_dir, tag, args, springs):
    m = meshio.read(vtu_path)
    pts = m.points
    cells = m.cells[0].data
    nc = len(cells)
    printed = onp.asarray(m.cell_data["printed"][0]) > 0.5
    layer_id = onp.asarray(m.cell_data["layer_id"][0]).astype(int)
    eqp = onp.asarray(m.cell_data["eq_plastic_strain"][0])
    u = onp.asarray(m.point_data["u"])

    S_quads = tensor_from_quads(m, nc)                 # (Q, nc, 3, 3)
    S_mean = S_quads.mean(axis=0)
    vm_cell = von_mises(S_mean)
    vm_quad_max = von_mises(S_quads).max(axis=0)
    w = onp.linalg.eigvalsh(S_mean)                    # ascending
    sigma1, sigma3 = w[:, 2], w[:, 0]

    yield_eff = onp.minimum(args.yield0 + args.hardening * eqp, args.saturation)
    utilization = vm_quad_max / onp.maximum(yield_eff, 1.0)
    plastic = (eqp > 1e-6).astype(float)

    # geometry: lateral free surface vs base plane
    face_nodes, owner = exterior_faces(cells)
    fx = pts[face_nodes][:, :, 0]
    x_min = pts[:, 0].min()
    is_base = onp.all(onp.abs(fx - x_min) <= args.base_tol, axis=1)
    lateral_centroids = pts[face_nodes[~is_base]].mean(axis=1)
    centroids = pts[cells].mean(axis=1)
    edge_distance = cKDTree(lateral_centroids).query(centroids)[0]

    aug = meshio.Mesh(
        pts, [("tetra", cells)],
        cell_data={
            "vm_cell": [vm_cell], "vm_quad_max": [vm_quad_max],
            "sigma1": [sigma1], "sigma3": [sigma3],
            "yield_utilization": [utilization], "plastic": [plastic],
            "edge_distance": [edge_distance],
            "printed": [printed.astype(float)], "layer_id": [layer_id.astype(float)],
        },
    )
    aug_path = os.path.join(out_dir, f"v05_boundary_{tag}.vtu")
    aug.write(aug_path)

    # profiles
    with open(os.path.join(out_dir, f"profile_height_{tag}.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["layer", "n", "vm_mean_MPa", "vm_p95_MPa", "sigma1_max_MPa",
                       "plastic_frac", "util_max"])
        for L in range(1, layer_id[printed].max() + 1):
            s = printed & (layer_id == L)
            if not s.any():
                continue
            wcsv.writerow([L, int(s.sum()), round(vm_cell[s].mean() / 1e6, 1),
                           round(onp.percentile(vm_cell[s], 95) / 1e6, 1),
                           round(sigma1[s].max() / 1e6, 1),
                           round(float(plastic[s].mean()), 4),
                           round(float(utilization[s].max()), 3)])

    bins = [0.0, 2.5e-3, 5e-3, 10e-3, 20e-3, 40e-3, 1.0]
    with open(os.path.join(out_dir, f"profile_edge_{tag}.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["edge_dist_mm", "n", "vm_mean_MPa", "vm_p95_MPa",
                       "sigma1_mean_MPa", "plastic_frac", "util_mean"])
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = printed & (edge_distance >= lo) & (edge_distance < hi)
            if not s.any():
                continue
            wcsv.writerow([f"{lo*1e3:.1f}-{hi*1e3:.1f}", int(s.sum()),
                           round(vm_cell[s].mean() / 1e6, 1),
                           round(onp.percentile(vm_cell[s], 95) / 1e6, 1),
                           round(sigma1[s].mean() / 1e6, 1),
                           round(float(plastic[s].mean()), 4),
                           round(float(utilization[s].mean()), 3)])

    # interface tractions from foundation springs (constrained state only)
    peel_stats = None
    if springs and args.foundation_stiffness > 0:
        bf = face_nodes[is_base]
        u_face = u[bf].mean(axis=1)                    # (nbase, 3)
        peel = args.foundation_stiffness * u_face[:, 0]
        shear = args.foundation_stiffness * onp.hypot(u_face[:, 1], u_face[:, 2])
        fc = pts[bf].mean(axis=1)
        center = fc.mean(axis=0)
        r = onp.hypot(fc[:, 1] - center[1], fc[:, 2] - center[2])
        rim = r > onp.percentile(r, 80)
        with open(os.path.join(out_dir, f"interface_tractions_{tag}.csv"), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["x", "y", "z", "peel_MPa", "shear_MPa"])
            for i in range(len(fc)):
                wcsv.writerow([*(round(v, 6) for v in fc[i]),
                               round(peel[i] / 1e6, 2), round(shear[i] / 1e6, 2)])
        peel_stats = (
            f"  interface faces: {len(fc)}\n"
            f"  peel  sigma_n  : max={peel.max()/1e6:7.1f} MPa (tension=delamination driver), "
            f"min={peel.min()/1e6:7.1f} MPa\n"
            f"  shear tau      : max={shear.max()/1e6:7.1f} MPa\n"
            f"  rim (outer 20% radius) vs center: "
            f"peel_mean {peel[rim].mean()/1e6:6.1f} / {peel[~rim].mean()/1e6:6.1f} MPa,  "
            f"shear_mean {shear[rim].mean()/1e6:6.1f} / {shear[~rim].mean()/1e6:6.1f} MPa"
        )

    # edge-vs-interior comparison + hotspots
    near = printed & (edge_distance < 2.5e-3)
    far = printed & (edge_distance >= 10e-3)
    lines = [
        f"=== {tag}: {os.path.basename(vtu_path)} ===",
        f"printed cells: {int(printed.sum())}",
        f"edge band (<2.5mm): n={int(near.sum())}  vm_mean={vm_cell[near].mean()/1e6:6.1f} MPa  "
        f"util_mean={utilization[near].mean():.3f}  plastic_frac={plastic[near].mean():.4f}",
        f"interior (>=10mm) : n={int(far.sum())}  vm_mean={vm_cell[far].mean()/1e6:6.1f} MPa  "
        f"util_mean={utilization[far].mean():.3f}  plastic_frac={plastic[far].mean():.4f}",
    ]
    order = onp.argsort(-utilization * printed)
    lines.append(f"top {args.top_n} yield-utilization hotspots:")
    for i in order[: args.top_n]:
        lines.append(
            f"  U={utilization[i]:.3f} vm_qmax={vm_quad_max[i]/1e6:7.1f} MPa "
            f"sigma1={sigma1[i]/1e6:7.1f} MPa layer={layer_id[i]:3d} "
            f"edge_dist={edge_distance[i]*1e3:5.1f} mm at ({centroids[i][0]:.4f}, "
            f"{centroids[i][1]:.4f}, {centroids[i][2]:.4f})"
        )
    if peel_stats:
        lines.append("interface tractions (foundation springs):")
        lines.append(peel_stats)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, f"report_{tag}.txt"), "w") as f:
        f.write(report + "\n")
    return aug_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--foundation-stiffness", type=float, default=1.0e12)
    ap.add_argument("--yield0", type=float, default=955.0e6)
    ap.add_argument("--hardening", type=float, default=1.45e9)
    ap.add_argument("--saturation", type=float, default=1.15e9)
    ap.add_argument("--base-tol", type=float, default=1.0e-4)
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()

    out_dir = os.path.join(args.run_dir, "v05_boundary")
    os.makedirs(out_dir, exist_ok=True)

    steps = [p for p in sorted(glob.glob(os.path.join(args.run_dir, "step_*.vtu")))
             if re.search(r"step_\d+", p)]
    if steps:
        analyze(steps[-1], out_dir, "constrained", args, springs=True)
        print()
    release = os.path.join(args.run_dir, "release.vtu")
    if os.path.exists(release):
        analyze(release, out_dir, "released", args, springs=False)
    print(f"\noutputs -> {out_dir}")


if __name__ == "__main__":
    main()
