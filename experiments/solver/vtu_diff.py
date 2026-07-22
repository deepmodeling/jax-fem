"""Compare temperature fields between two bench-run VTU files."""
import sys
import numpy as np
import meshio

ref, test = sys.argv[1], sys.argv[2]
mr = meshio.read(ref)
mt = meshio.read(test)
keys = sorted(set(mr.point_data) | set(mt.point_data))
out = []
for k in keys:
    a, b = mr.point_data.get(k), mt.point_data.get(k)
    if a is None or b is None:
        out.append(f"{k}: MISSING")
        continue
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    out.append(f"{k}: max_abs={d:.3e}")
print("; ".join(out))
