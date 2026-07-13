"""Compare whole-layer flash vs stride-64 track fine-scan (both v05 full-91)."""
import re

import meshio
import numpy as np

RUNS = {
    "flash (fast-scan)": "/home/user/work/159/output/fastscan_flash_v05_full91_20260709_174441",
    "track stride-64": "/tmp/finescan_smoke",
}

for name, out in RUNS.items():
    rel = meshio.read(f"{out}/release.vtu")
    printed = np.asarray(rel.cell_data["printed"][0]) > 0.5
    vm_keys = sorted(k for k in rel.cell_data if k.startswith("vm_quad"))
    vm = np.max(np.stack([np.asarray(rel.cell_data[k][0]) for k in vm_keys]), axis=0)
    eqp = np.asarray(rel.cell_data["eq_plastic_strain"][0])
    u = np.asarray(rel.point_data["u"])
    log = f"{out}/stdout.log" if name.startswith("flash") else f"{out}/driver.log"
    tmins = []
    try:
        with open(log) as f:
            for line in f:
                m = re.search(r"T_min=([-0-9.e+]+)", line)
                if m:
                    tmins.append(float(m.group(1)))
    except OSError:
        pass
    tmin = min(tmins) if tmins else float("nan")
    plastic = (eqp[printed] > 1e-6).sum()
    print(f"{name}:")
    print(f"  release |u|_max = {np.linalg.norm(u, axis=1).max()*1e3:.4f} mm")
    print(f"  released vm: p99={np.percentile(vm[printed],99)/1e6:7.1f}  "
          f"p99.9={np.percentile(vm[printed],99.9)/1e6:7.1f} MPa (峰值不引用—网格门控未做)")
    print(f"  plastic cells = {plastic}  eqp_max = {eqp[printed].max():.4f}")
    print(f"  build-phase summary T_min = {tmin:.1f} K")
