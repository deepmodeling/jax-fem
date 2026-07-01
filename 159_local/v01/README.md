# INP thermal stress one-way x-build example

This directory documents the local script:

```text
159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py
```

The script runs a one-way coupled additive-manufacturing thermal stress simulation on an existing Abaqus-style TET4 `.inp` volume mesh. It first solves the transient temperature field `T`, then uses that temperature field to solve displacement `u` and stress. Mechanical deformation does not feed back into the thermal solve.

This README describes the implementation in the local script. It should not be read as a calibrated industrial LPBF model without separate material, heat-source, and residual-stress validation.

## Scope

The current implementation is intended for a full imported tetrahedral body mesh, not for CAD meshing. CAD formats such as STEP/STP or Parasolid XT must be meshed upstream into a supported volume mesh before this script can run.

Main assumptions:

- Mesh input is read through `read_tet4_inp(...)` and converted to `Mesh(points, cells, ele_type="TET4")`.
- The build direction is controlled by `--build-axis` and `--base-side`.
- The default AM setup is `build_axis=x` and `base_side=min`, so layers advance from `x_min` toward `x_max`.
- The thermal problem is transient.
- The mechanics problem is quasi-static at selected thermal steps.
- Coupling is one-way: `T -> dT -> u/stress`.

## Quick Start

Run from the repository root:

```bash
cd /home/user/work/159/jax-fem

python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 500 \
  --layers 2 \
  --hatch-lines-per-layer 2 \
  --scan-steps-per-layer 3 \
  --laser-power 100000000 \
  --output-dir /home/user/work/159/output/inp_xbuild_smoke
```

For a mesh stored in millimeters, scale the coordinates to meters explicitly:

```bash
python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  --inp /path/to/model.inp \
  --mesh-length-scale 1e-3 \
  --build-axis x \
  --base-side min \
  --layers 50 \
  --hatch-lines-per-layer 4 \
  --scan-steps-per-layer 50 \
  --laser-power 300 \
  --absorptivity 0.35 \
  --output-dir /home/user/work/159/output/my_case
```

The script writes `used_config.json` into the output directory, plus `.vtu` files for visualization in ParaView.

## Model Setup

The script builds two `jax_fem.Problem` subclasses:

| Class | Unknown | Purpose |
| --- | --- | --- |
| `TransientThermal` | scalar `T` | transient heat conduction with laser body source, active mask, latent heat approximation, convection/radiation losses |
| `ThermoMechanical` | vector `u` | small-strain thermal stress with linear elasticity by default and optional simplified `j2_plastic` mode |

At startup, the script:

1. Reads TET4 cells from `.inp`.
2. Scales coordinates with `--mesh-length-scale`.
3. Detects `bottom`, `exposed`, and `walls` from the mesh bounding box.
4. Builds synthetic raster or CSV-driven laser step states.
5. Initializes temperature, displacement, activation step, activation temperature, and maximum temperature history.
6. Marches over scan/cooling steps.
7. Solves thermal first, then mechanics when `--mechanics-every` requests it.

## Governing Equations

### Thermal Problem

The implemented thermal residual corresponds to a backward-Euler heat equation:

```math
\rho c_{p,\mathrm{eff}} \frac{T^n - T^{n-1}}{\Delta t}
- \nabla \cdot (k \nabla T^n)
- q_\mathrm{laser}
+ q_\mathrm{front-loss}
= 0
```

The thermal weak form represented by the script is:

```math
\int_\Omega k \nabla T^n \cdot \nabla w \, d\Omega
+ \int_\Omega
\left[
\rho c_{p,\mathrm{eff}} \frac{T^n - T^{n-1}}{\Delta t}
- q_\mathrm{laser}
+ q_\mathrm{front-loss}
\right] w \, d\Omega
+ \int_\Gamma q_\mathrm{surface} w \, d\Gamma
= 0
```

The JAX-FEM callback mapping is:

| Callback | Implemented term |
| --- | --- |
| `TransientThermal.get_tensor_map()` | `k * grad(T)` diffusion term |
| `TransientThermal.get_mass_map()` | time term, laser body source, moving-front loss |
| `TransientThermal.get_surface_maps()` | external convection/radiation surface contribution |

The laser source is a volumetric approximation:

```math
q_\mathrm{laser}
=
\frac{2 P_\mathrm{eff}}{\pi r_b^2 d}
\exp\left(-\frac{2r^2}{r_b^2}\right)
\exp\left(-\frac{\mathrm{depth}}{d}\right)
\chi_{\mathrm{depth}\ge 0}
\chi_\mathrm{active}
\chi_\mathrm{laser-on}
```

where:

- `P_eff = absorptivity * laser_power`
- `r_b = beam_radius`
- `d = source_depth`
- `r` is the distance from the laser center in the scan plane
- `depth` is measured along the build axis behind the laser center

`laser_power` is therefore the numerical strength of this volumetric heat source. It is not automatically a calibrated physical machine laser power unless the model has been calibrated against melt pool, temperature, or distortion data.

### Mechanical Problem

The mechanics solve is quasi-static:

```math
-\nabla \cdot \sigma = 0
```

with weak form:

```math
\int_\Omega \sigma : \nabla \delta u \, d\Omega = 0
```

The default constitutive model is small-strain isotropic thermoelasticity:

```math
\varepsilon = \frac{1}{2}(\nabla u + \nabla u^T)
```

```math
\varepsilon_\mathrm{th} = \alpha \Delta T I
```

```math
\sigma_\mathrm{trial}
=
\lambda \mathrm{tr}(\varepsilon - \varepsilon_\mathrm{th})I
+ 2\mu(\varepsilon - \varepsilon_\mathrm{th})
```

The stress returned to the weak form is scaled by the active mechanics factor:

```math
\sigma = f_\mathrm{active} \sigma_\mathrm{trial}
```

If `--mechanics-model j2_plastic` is selected, the script applies a simplified deviatoric stress clipping and updates `eq_plastic_strain`. This is not a full production return-mapping plasticity implementation.

The mechanical temperature increment is computed as `dT = T - activation_temperature`. Here `activation_temperature` is a frozen first-activation temperature snapshot for each cell. It is not a proof that the cell is physically stress-free at that temperature.

The optional `j2_plastic` mode is a simplified numerical option and is not suitable for residual-stress prediction without a proper return-mapping model, validation, and calibration.

## Physical Substitutions and Weak Representations

The script uses several practical approximations:

| Physical item | Implementation |
| --- | --- |
| Laser heat input | volumetric Gaussian/exponential source; `laser_power` is a numerical heat-source strength unless calibrated |
| Moving build-front loss | heuristic volumetric damping/sink near the build front; not a real boundary or geometric interface condition |
| Latent heat | equivalent heat capacity inside `[solidus_temperature, liquidus_temperature]` |
| Powder or void region | inactive cells keep weak thermal/mechanical material properties instead of being deleted from the mesh; `void` is weak material, not true geometric void |
| Element activation | `active_cell = previous_active OR raw_active_cell`, so activation is monotonic; there is no element insertion, deletion, or true birth/death |
| Thermal stress reference | each cell records a frozen first-activation temperature snapshot |
| Mechanical temperature increment | `dT = T - activation_temperature`, with inactive cells zeroed by `active_quad` |
| Substrate/support | classified from build-axis distance, not from Abaqus elsets |

## Scan Path and Layer Activation

Raster scanning is generated synthetically from:

- `--layers`
- `--hatch-lines-per-layer`
- `--scan-steps-per-layer`
- `--scan-axis`
- `--scan-start-frac`, `--scan-end-frac`
- `--hatch-start-frac`, `--hatch-end-frac`
- `--serpentine` / `--no-serpentine`

This synthetic raster generator is not a machine G-code interpreter. The build front advances one layer at a time along the build axis. A cell becomes active once the current build front has passed its centroid in the build direction. Substrate and support cells are active from the start.

For measured or externally generated toolpaths, use `--path-file`. This is a simple CSV path input, not a general G-code parser. The CSV must contain:

```text
time,x,y,z,power,laser_on,layer,hatch,mode
```

Use `--path-length-scale` if the path coordinates use a different unit scale from the mesh.

## Important Parameters

### Mesh and Units

| Option | Meaning |
| --- | --- |
| `--inp` | Abaqus-style TET4 input file |
| `--max-cells` | optional cell limit for smoke tests |
| `--mesh-length-scale` | coordinate scaling applied immediately after reading the mesh |
| `--build-axis` | layer build axis: `x`, `y`, or `z` |
| `--base-side` | base side along build axis: `min` or `max` |

### Thermal Material

| Option | Meaning |
| --- | --- |
| `--rho`, `--cp`, `--conductivity` | fallback solid constants |
| `--rho-solid`, `--cp-solid`, `--conductivity-solid` | explicit solid constants |
| `--rho-powder`, `--cp-powder`, `--conductivity-powder` | inactive powder constants |
| `--powder-mode powder|void` | inactive material treatment |
| `--k-table-solid`, `--cp-table-solid` | temperature-dependent solid property CSV |
| `--k-table-powder`, `--cp-table-powder` | temperature-dependent powder property CSV |
| `--latent-heat` | enables equivalent heat capacity when phase temperatures are set |

Property-table CSV files must have columns:

```text
T,value
```

### Laser Source

| Option | Meaning |
| --- | --- |
| `--laser-power` | numerical heat-source strength used before absorptivity scaling |
| `--absorptivity` | multiplier applied to `laser_power` as `P_eff = absorptivity * laser_power` |
| `--beam-radius` | Gaussian in-plane beam radius; if `0`, estimated from part size |
| `--source-depth` | one-sided exponential depth scale; if `0`, estimated from beam radius and build span |

### Boundary Conditions

| Option | Meaning |
| --- | --- |
| `--ambient` | ambient/reference environment temperature |
| `--preheat-temperature` | initial temperature; defaults to ambient |
| `--bottom-temperature` | fixed bottom temperature; defaults to initial temperature |
| `--bottom-thermal-bc fixed|convection` | fixed bottom temperature or external loss on bottom |
| `--convection-h` | convection coefficient |
| `--emissivity` | radiation emissivity; `0` disables radiation |
| `--front-surface-loss-h` | enables moving-front volumetric loss when greater than zero |

### Mechanics

| Option | Meaning |
| --- | --- |
| `--young`, `--poisson`, `--alpha` | fallback thermoelastic constants |
| `--E-table`, `--alpha-table`, `--poisson-table` | temperature-dependent mechanics property CSV |
| `--mechanics-model linear_elastic|j2_plastic` | mechanics model |
| `--yield-table`, `--hardening-table` | required/used for `j2_plastic` |
| `--inactive-mechanics-factor` | weak mechanics contribution of inactive cells |

### Output Cadence

| Option | Meaning |
| --- | --- |
| `--mechanics-every` | run mechanics every N global steps; `0` means thermal-only output |
| `--thermal-output-every` | save thermal states every N steps; `0` disables this cadence |
| `--mechanics-output-every` | save mechanics-valid states every N mechanics solves |
| `--summary-every` | print runtime summaries every N steps |
| `--cooling-steps` | append laser-off cooling steps |
| `--release-after-cooling` | after cooling, solve mechanics with minimal anchor constraints |

## Output Fields

Each saved `.vtu` includes point fields:

- `T`
- `u`

and cell fields:

- `active`
- `layer_id`
- `activation_step`
- `activation_temperature`
- `material_state`
- `dT`
- `stress_quad_xx`, `stress_quad_yy`, `stress_quad_zz`
- `stress_quad_xy`, `stress_quad_yz`, `stress_quad_xz`
- `vm_quad`
- `eq_plastic_strain`
- `max_temperature_history`
- `mechanics_valid`
- `mechanics_source_step`
- `mode_id`

The solver output intentionally keeps only raw quadrature stress fields and `vm_quad`. For `TET4`, jax-fem uses one quadrature point, so each raw quadrature field is a cell scalar. If a future element has multiple quadrature points, the raw fields are written with a quadrature index, for example `stress_quad0_xx` and `vm_quad0`.

Derived quantities are not written by the solver. Run the postprocess step to create mean/max/p95 and recovered nodal fields:

```bash
python 159_local/postprocess_quad_stress.py \
  /home/user/work/159/output/readme_smoke/step_000000_scan.vtu
```

The postprocess output adds cell fields:

- `stress_xx_mean`, `stress_yy_mean`, `stress_zz_mean`
- `stress_xy_mean`, `stress_yz_mean`, `stress_xz_mean`
- `stress_xx`, `stress_yy`, `stress_zz`
- `stress_xy`, `stress_yz`, `stress_xz`
- `von_mises_mean`
- `von_mises_max`
- `von_mises_p95`
- `von_mises`

and point fields:

- `recovered_von_mises_mean`
- `recovered_stress_xx_mean`, `recovered_stress_yy_mean`, `recovered_stress_zz_mean`
- `recovered_stress_xy_mean`, `recovered_stress_yz_mean`, `recovered_stress_xz_mean`

The postprocess computes:

```text
stress_quad / vm_quad -> mean / max / p95 / nodal averaged recovery
```

The legacy `von_mises` field is produced only by postprocess as a compatibility alias for `von_mises_mean`. The `recovered_*` point fields are a minimum stress-recovery aid: each cell-mean stress value is averaged onto its connected nodes. This is useful for smoother visualization, but it is not an L2 projection and should not be treated as mesh-independent peak-stress evidence.

`material_state` uses:

| Value | Meaning |
| --- | --- |
| `0` | void/inactive weak material |
| `1` | powder/inactive powder |
| `2` | active solid |
| `3` | active mushy zone |
| `4` | active liquid |
| `5` | substrate |
| `6` | support |

## Practical Notes

- Check `raw_pmin/raw_pmax` and scaled coordinate ranges printed at startup. Wrong units are the most common cause of unrealistic beam radius, source depth, and power density.
- If the imported `.inp` already has the desired volume mesh, do not remesh it inside this script. The solver consumes the existing TET4 cells.
- The bottom and side boundary selectors are coordinate-based bounding-box predicates. If the part has a complex support or fixture definition, this script does not currently read Abaqus elsets for that purpose.
- `--absorptivity 0` is a useful thermal-source sanity check: the laser should no longer add heat.
- Increase output frequency carefully for large meshes; `.vtu` files can become large.

## Minimal Smoke Tests

Syntax check:

```bash
python -m py_compile \
  159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  159_local/postprocess_quad_stress.py
```

Stress post-processing unit checks:

```bash
python -m unittest tests.test_local_stress_postprocessing
```

This checks:

- uniaxial stress: `von_mises ~= axial stress`
- pure shear: `von_mises ~= sqrt(3) * shear stress`
- nonlinear post-processing: `mean(von_mises_quad)` is not replaced by `von_mises(mean(stress_quad))`
- solver output contains only raw `stress_quad*` and `vm_quad*` stress fields
- postprocess derives `von_mises_mean`, `von_mises_max`, `von_mises_p95`, stress means, compatibility aliases, and nodal recovered fields
- postprocess recovery: adjacent cell data are averaged to shared nodes
- free thermal expansion at the constitutive-map level gives near-zero stress
- fully constrained heating at the constitutive-map level gives nonzero compressive thermal stress
- bottom-fixed thermal expansion on a small TET4 box concentrates von Mises stress near the fixed base when real `jax`/`jax-fem` are available

On Windows without `jax`, the bottom-fixed integration test is skipped. In the WSL conda environment, use CPU mode to avoid GPU memory preallocation during this verification:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m unittest tests.test_local_stress_postprocessing
```

Small path/activation test:

```bash
python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  --max-cells 500 \
  --layers 2 \
  --hatch-lines-per-layer 2 \
  --scan-steps-per-layer 3 \
  --mechanics-every 1 \
  --output-dir /home/user/work/159/output/readme_smoke
```

The bottom-fixed field sanity check above is intentionally small and verifies the post-processing and constraint trend. It is not a calibrated residual-stress benchmark.

Thermal-only larger run:

```bash
python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  --inp /path/to/model.inp \
  --mesh-length-scale 1e-3 \
  --layers 50 \
  --hatch-lines-per-layer 4 \
  --scan-steps-per-layer 50 \
  --mechanics-every 0 \
  --thermal-output-every 10 \
  --output-dir /home/user/work/159/output/readme_thermal_only
```
