# P0/P1 fixed version validation commands

Assume the patched script is placed at:

```bash
/home/user/work/159/jax-fem/159_local/inp_thermal_stress_oneway_xbuild.py
```

## 1. Replace script

```bash
cp /path/to/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  /home/user/work/159/jax-fem/159_local/inp_thermal_stress_oneway_xbuild.py
```

## 2. Syntax and help

```bash
cd /home/user/work/159/jax-fem
python -m py_compile 159_local/inp_thermal_stress_oneway_xbuild.py
python 159_local/inp_thermal_stress_oneway_xbuild.py --help | grep -E "path-length-scale|bottom-temperature|front-surface-loss|mechanics-every|mechanics_valid"
```

## 3. 20-cell smoke test

```bash
cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 20 \
  --mesh-length-scale 1.0 \
  --layers 5 \
  --hatch-lines-per-layer 2 \
  --scan-steps-per-layer 3 \
  --dt 1e-3 \
  --build-axis x \
  --base-side min \
  --scan-axis auto \
  --laser-power 100 \
  --absorptivity 0.35 \
  --mechanics-model linear_elastic \
  --mechanics-every 3 \
  --thermal-output-every 3 \
  --mechanics-output-every 3 \
  --summary-every 1 \
  --output-dir /home/user/work/159/output/check_p0p1_20cells
```

Expected checks:

```text
active_cells must be monotonically non-decreasing.
used_config.json exists.
VTU fields include mechanics_valid, mechanics_source_step, mode_id.
```

## 4. Absorptivity zero test

```bash
cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 20 \
  --layers 2 \
  --hatch-lines-per-layer 1 \
  --scan-steps-per-layer 2 \
  --laser-power 100 \
  --absorptivity 0 \
  --mechanics-every 0 \
  --thermal-output-every 1 \
  --summary-every 1 \
  --output-dir /home/user/work/159/output/check_p0p1_abs0
```

Expected:

```text
T_min and T_max remain at ambient/preheat temperature.
mechanics_valid = 0 in thermal-only outputs.
```

## 5. Preheat + fixed bottom temperature consistency

```bash
cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 20 \
  --layers 1 \
  --hatch-lines-per-layer 1 \
  --scan-steps-per-layer 1 \
  --preheat-temperature 473.15 \
  --bottom-thermal-bc fixed \
  --laser-power 0 \
  --absorptivity 0 \
  --mechanics-every 0 \
  --thermal-output-every 1 \
  --summary-every 1 \
  --output-dir /home/user/work/159/output/check_p0p1_preheat_bottom
```

Expected startup log:

```text
bottom_temperature_effective: 473.15
```

To force the bottom to 300 K while the part is preheated:

```bash
--bottom-temperature 300
```

## 6. Path-file scaling, monotonic active, and cooling append test

Create a tiny path file in raw mesh units. If your mesh is in mm and you use `--mesh-length-scale 1e-3`, this path is also scaled by default.

```bash
cat > /home/user/work/159/output/check_path_p0p1.csv <<'CSV'
time,x,y,z,power,laser_on,layer,hatch,mode
0.000,0.0,0.0,0.0,100,1,1,1,path
0.001,1.0,0.0,0.0,100,1,1,1,path
0.002,0.5,0.0,0.0,100,1,1,1,path
CSV
```

Run:

```bash
cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 20 \
  --mesh-length-scale 1e-3 \
  --path-file /home/user/work/159/output/check_path_p0p1.csv \
  --layers 2 \
  --cooling-steps 2 \
  --laser-power 100 \
  --absorptivity 0.35 \
  --mechanics-every 0 \
  --thermal-output-every 1 \
  --summary-every 1 \
  --output-dir /home/user/work/159/output/check_p0p1_path_scale_cooling
```

Expected:

```text
path_length_scale: 0.001
cooling steps appear after path rows.
active_cells do not decrease even though the third path x is lower than the second.
```

## 7. Path-file non-monotonic time must fail

```bash
cat > /home/user/work/159/output/check_bad_time_p0p1.csv <<'CSV'
time,x,y,z,power,laser_on,layer,hatch,mode
0.000,0.0,0.0,0.0,100,1,1,1,path
0.000,1.0,0.0,0.0,100,1,1,1,path
CSV

cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 20 \
  --path-file /home/user/work/159/output/check_bad_time_p0p1.csv \
  --output-dir /home/user/work/159/output/check_p0p1_bad_time
```

Expected error:

```text
--path-file time must be strictly increasing
```

## 8. 500-cell acceptance run

```bash
cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 500 \
  --mesh-length-scale 1.0 \
  --layers 10 \
  --hatch-lines-per-layer 3 \
  --scan-steps-per-layer 5 \
  --dt 1e-3 \
  --build-axis x \
  --base-side min \
  --scan-axis auto \
  --laser-power 100 \
  --absorptivity 0.35 \
  --powder-mode powder \
  --mechanics-model linear_elastic \
  --mechanics-every 10 \
  --thermal-output-every 10 \
  --mechanics-output-every 10 \
  --summary-every 5 \
  --output-dir /home/user/work/159/output/check_p0p1_500cells
```

## 9. Optional moving-front heat-loss approximation

```bash
cd /home/user/work/159/jax-fem
python 159_local/inp_thermal_stress_oneway_xbuild.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 500 \
  --layers 5 \
  --hatch-lines-per-layer 2 \
  --scan-steps-per-layer 5 \
  --laser-power 100 \
  --absorptivity 0.35 \
  --front-surface-loss-h 10 \
  --front-surface-loss-radiation \
  --thermal-output-every 5 \
  --mechanics-every 0 \
  --summary-every 1 \
  --output-dir /home/user/work/159/output/check_p0p1_front_loss
```

Expected startup log:

```text
front_surface_loss_h: 10.0
front_surface_loss_thickness: <source_depth if not explicitly specified>
front_surface_loss_radiation: True
```
