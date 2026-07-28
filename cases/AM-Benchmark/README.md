# AM-Benchmark — NIST AM-Bench 2018 AMB2018-01

Validation of the LPBF thermo-mechanical solver against **real measurement data**,
replacing the code-to-code Kaess 2023 reproduction.

**Working document: [`PREREQUISITES.md`](PREREQUISITES.md)** — the decision log,
verified facts, registered conflicts and data gaps. Read that first; this file is
orientation only.

## Why this benchmark

`CHAL-AMB2018-01-PD` measures **part deflection after the part is partially
separated from the build plate by wire EDM** — the same quantity the existing
release/warping machinery already computes, so the solver capability transfers.

`CHAL-AMB2018-01-RS` adds residual elastic strain tensors measured by neutron and
synchrotron X-ray diffraction inside the structure.

Unlike Kaess 2023, the build conditions are published, so there is no blind
assumption budget to spend on unknown anchors, scan start corners or cut locations.

Governing paper: Phan et al. (2019), *IMMI* 8(3):318-334,
doi 10.1007/s40192-019-00149-0.

## Status

| | |
|---|---|
| Source data | **downloaded and checksum-verified** — 63 files, 16.3 MB, 30/30 NIST-supplied SHA-256 match |
| Geometry | **verified** — both STLs watertight, Euler 2, orientation consistent, zero degenerate triangles |
| Design decisions | **D-01 … D-07 settled** (see PREREQUISITES.md section 0) |
| Material data | source scope decided (Option A); acquisition not started |
| Mesh | not started |
| Solver | **not run since the `codex/r3-optimization` merge** — `tests/benchmarks/` has not been executed |

## Directory layout

```
references/geometry/      AMB2018_01_Part.STL, AMB2018_01_Build.STL (+ .sha256)
references/material/      IN625 and 15-5 mill certificates (chemistry and PSD only —
                          NOT thermophysical properties)
references/measurements/  neutron and EDXRD residual strain results, numeric + figures
references/docs/          Phan 2019 full paper, scan-strategy and chamber videos
inputs/                   frozen, hashed inputs derived from references/
derived/                  meshes, scan paths, material tables generated here
```

`references/` holds bytes exactly as downloaded and is never edited. Everything in
`inputs/` is derived, hashed and recorded.

## Key geometry (measured from the official STL, not from the paper)

Part 75.000 x 5.000 x 12.500 mm. Legs 0 -> 5.0 mm; 45 degree overhang 5.0 -> 7.5 mm;
constant-section bridge 7.5 -> 12.0 mm; ridges 12.0 -> 12.5 mm.

12 legs in 4 repeats of (5.0, 0.5, 2.5) mm with uniform 2.0 mm gaps, 14.0 mm period,
plus a 19.0 mm end block. **The 0.5 mm thin leg is the minimum feature and sets the
mesh floor.**

11 ridges, 1.000 mm wide, 7.000 mm pitch, centres at x = 0.5, 7.5, 14.5, 21.5, 28.5,
35.5, 42.5, 49.5, 56.5, 63.5, 70.5 mm. **These are the CMM measurement targets — the
validation metric lands on these x positions.**

Ridge numbering is inverted between the NIST prose and every figure. **Key the model
on the x coordinate, never the ridge index.** (Conflict B4.)

## Environment

```
/home/user/miniconda3/envs/jax-fem-env/bin/python
```

jax 0.10.2 (CPU, x64), petsc4py 3.25.1, fenics-basix 0.10.0, pypardiso 0.4.7 +
MKL 2026.1.0, gmsh 4.15.2, meshio 5.3.5, pymupdf 1.28.0. jaxlib is CPU-only; the
RTX 5080 is unused until `jax[cuda12]` is installed.

Conda and pip are pointed at the TUNA mirrors — direct `files.pythonhosted.org`,
`conda.anaconda.org` and GitHub release assets are unreachable from this network.
