# Local Thermal-Mechanical INP Example

This folder contains local examples for running JAX-FEM thermal-mechanical workflows on an imported Abaqus `*.inp` volume mesh.

The scripts are adapted from `applications/thermal_mechanical`, but they use an external tetrahedral mesh instead of generating the box mesh internally.

## Files

| File | Purpose |
| --- | --- |
| `smoke_test.py` | Reads an Abaqus `*.inp` tetrahedral mesh, optionally extracts a connected submesh, builds a small-strain elasticity problem, writes `initial_guess.vtu`, and can run a small mechanics solve. |
| `run.py` | Runs one-way transient thermal to linear thermoelastic stress on an imported `TET4` mesh. The heat source can scan along a selected axis. |

## Relation to the Official Example

The official `applications/thermal_mechanical/example.py` solves a one-way coupled LPBF-style problem:

1. solve transient temperature `T`;
2. use temperature change to solve displacement `u`;
3. write VTK/VTU outputs for ParaView.

The local `run.py` keeps that same high-level workflow, but differs in several practical ways:

| Aspect | Official application | Local example |
| --- | --- | --- |
| Mesh | Generated `HEX8` box mesh | Imported Abaqus `C3D4` / meshio `tetra` mesh |
| Element type | `HEX8` | `TET4` |
| Thermal source | Surface Gaussian laser flux on top boundary | Volumetric Gaussian heat source plus optional top surface flux when top faces are detected |
| Mechanics | Plasticity and phase state | Linear thermoelasticity |
| Input geometry | Internal box parameters | External `--inp` file |

## Initial Conditions and Boundary Conditions

`run.py` does not read Abaqus material, load, step, or initial-condition cards from the `*.inp` file. The `*.inp` file is used as mesh input only.

Current initial conditions:

| Quantity | Definition |
| --- | --- |
| Temperature | `T_old = ambient` at every node; default `--ambient 300.0` |
| Displacement guess | zero vector at every node |
| Thermal Dirichlet BC | `bottom` nodes fixed to `ambient` |
| Mechanical Dirichlet BC | `bottom` nodes fixed in `ux`, `uy`, `uz` |

The boundary selectors are coordinate-based:

| Selector | Logic |
| --- | --- |
| `bottom` | `z == min(z)` within tolerance |
| `top` | `z == max(z)` within tolerance |
| `walls` | `x == min/max(x)` or `y == min/max(y)` within tolerance |

For a complex CAD-derived mesh, check the startup log:

```text
thermal_boundary_face_counts: [...]
thermal_dirichlet_node_counts: [...]
mechanical_dirichlet_node_counts: [...]
```

If `top` faces or `bottom` nodes are unexpectedly near zero, the coordinate-based selectors do not match the real physical boundary well.

## Mesh Input

The practical solver input is:

```text
Mesh(points, cells, ele_type="TET4")
```

The default mesh path is:

```text
/home/user/work/159/schema/0119_c3d4_only.inp
```

`--max-cells` controls whether a submesh is extracted:

| Value | Behavior |
| --- | --- |
| `--max-cells 0` | Use the full imported `*.inp` tetrahedral mesh |
| `--max-cells N` | Use a connected submesh of up to `N` tetrahedral cells |

Use `--max-cells 0` when you want to preserve the full imported model shape.

## Running from WSL

Activate the conda environment first:

```bash
cd ~/work/159/jax-fem
source ~/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Because the scripts in this folder import helpers from `examples_local`, run them with:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/run.py --help
```

## Smoke Test

Read a small submesh and write the initial displacement guess:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/smoke_test.py \
  --max-cells 200 \
  --output-dir ~/work/159/output/inp_initial_guess_smoke
```

Run a small mechanics solve:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/smoke_test.py \
  --max-cells 50 \
  --solve \
  --output-dir ~/work/159/output/inp_initial_guess_smoke_solve50
```

Outputs:

```text
initial_guess.vtu
solved.vtu
```

## Thermal-Mechanical Scan

A small quick run:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/run.py \
  --max-cells 500 \
  --steps 5 \
  --laser-power 100000000 \
  --output-dir ~/work/159/output/inp_thermal_stress_500cells_test5
```

A full imported mesh run:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/run.py \
  --max-cells 0 \
  --steps 50 \
  --laser-power 100000000 \
  --output-dir ~/work/159/output/inp_thermal_stress_fullmesh_50steps
```

Outputs are written as:

```text
step_0000.vtu
step_0001.vtu
...
```

Each output contains:

| Field | Location | Meaning |
| --- | --- | --- |
| `sol` | point data | temperature |
| `u` | point data | displacement vector |
| `dT` | cell data | mean temperature increment |
| `stress_xx` | cell data | cell-mean xx stress |
| `von_mises` | cell data | von Mises stress |

## Laser Scan Controls

The laser source is controlled by these parameters:

| Parameter | Meaning |
| --- | --- |
| `--laser-power` | Effective heat power used by the source |
| `--beam-radius` | Gaussian beam radius; if omitted, inferred from the mesh bounding box |
| `--source-depth` | Depth of the volumetric heat source; if omitted, inferred from beam radius and model span |
| `--scan-axis` | Axis to scan along: `x`, `y`, or `z` |
| `--scan-start` | Absolute coordinate where scanning begins along `--scan-axis` |
| `--scan-end` | Absolute coordinate where scanning ends along `--scan-axis` |
| `--scan-start-frac` | Bounding-box fraction used when `--scan-start` is omitted; default `0.25` |
| `--scan-end-frac` | Bounding-box fraction used when `--scan-end` is omitted; default `0.75` |
| `--scan-fixed-x` | Fixed x coordinate for non-scan directions |
| `--scan-fixed-y` | Fixed y coordinate for non-scan directions |
| `--scan-fixed-z` | Fixed z coordinate for non-scan directions |
| `--scan-speed` | Physical scan speed. If greater than zero, the source moves by `scan_speed * dt` per step and turns off after reaching the end. |

If `--scan-speed 0`, the source moves uniformly from start to end over `--steps`.

### Bounding-Box Scan

Scan along the x axis from 25% to 75% of the model bounding box:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/run.py \
  --max-cells 0 \
  --steps 50 \
  --laser-power 100000000 \
  --scan-axis x \
  --scan-start-frac 0.25 \
  --scan-end-frac 0.75 \
  --output-dir ~/work/159/output/inp_thermal_stress_fullmesh_scan_x
```

### Real-Coordinate Scan

Use this form when you know the laser path in the same coordinate system and units as the `*.inp` mesh:

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/run.py \
  --max-cells 0 \
  --steps 50 \
  --dt 1e-4 \
  --laser-power 100000000 \
  --beam-radius 0.002 \
  --source-depth 0.001 \
  --scan-axis x \
  --scan-start 3.70 \
  --scan-end 3.90 \
  --scan-fixed-y -0.23 \
  --scan-fixed-z -0.20 \
  --output-dir ~/work/159/output/inp_thermal_stress_fullmesh_scan_x_realcoords
```

### Speed-Based Scan

If you know the scan speed, set `--scan-speed`. The laser advances by:

```text
scan_speed * dt
```

per time step. After reaching `--scan-end`, `laser_switch` becomes `0`.

```bash
PYTHONPATH=examples_local python examples_local/thermal_mechanical/run.py \
  --max-cells 0 \
  --steps 100 \
  --dt 1e-4 \
  --laser-power 100000000 \
  --beam-radius 0.002 \
  --source-depth 0.001 \
  --scan-axis x \
  --scan-start 3.70 \
  --scan-end 3.90 \
  --scan-speed 0.5 \
  --scan-fixed-y -0.23 \
  --scan-fixed-z -0.20 \
  --output-dir ~/work/159/output/inp_thermal_stress_fullmesh_scan_x_speed
```

## Practical Notes

- Keep mesh coordinates, material parameters, `beam-radius`, `source-depth`, `scan-speed`, and `dt` in a consistent unit system.
- The default material parameters are SI-like values. If the `*.inp` coordinates are in millimeters, rescale either the mesh or all physical parameters consistently.
- For large full-mesh runs, start with `--steps 1` or `--steps 5` before launching long scans.
- `--mechanics-every 5` can reduce cost by solving mechanics every five thermal steps.
- Use ParaView's `Warp By Vector` with vector field `u` and an amplified scale factor to inspect deformation.
- If the startup log reports `thermal_boundary_face_counts: [0, ...]`, top-surface flux is not active for the current boundary selector; the volumetric heat source still moves along the scan path.

## Visualization

Open the generated `step_*.vtu` files in ParaView.

Suggested views:

- color by `sol` for temperature;
- color by `von_mises` for stress;
- apply `Warp By Vector` using `u` to visualize deformation.
