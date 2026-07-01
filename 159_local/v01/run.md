python 159_local/inp_thermal_stress_oneway_xbuild_layers_hatch.py   --inp /home/user/work/159/schema/0119_c3d4_only.inp   --max-cells 0   --layers 50   --hatch-lines-per-layer 10   --scan-steps-per-layer 20   --dt 1e-3   --build-axis x   --base-side min   --scan-axis auto   --laser-power 100000000   --mechanics-every 200   --output-dir /home/user/work/159/output/xbuild_50layers_full_hatch10_scan20

$ python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py   --inp /home/user/work/159/schema/0119_c3d4_only.inp   --max-cells 0   --mesh-length-scale 1.0   --build-axis x   --base-side min   --scan-axis auto   --layers 50   --hatch-lines-per-layer 10   --scan-steps-per-layer 20   --dt 1e-4   --laser-power 200   --absorptivity 0.35   --beam-radius 5e-5   --source-depth 5e-5   --powder-mode powder   --rho 7800   --cp 500   --conductivity 20   --rho-powder 3900   --cp-powder 500   --conductivity-powder 1.0   --ambient 300   --preheat-temperature 300   --convection-h 10   --emissivity 0.3   --bottom-thermal-bc convection   --solidus-temperature 1600   --liquidus-temperature 1700   --latent-heat 2.7e5   --mechanics-model linear_elastic   --young 2.0e11   --poisson 0.3   --alpha 1.2e-5   --cooling-steps 500   --release-after-cooling   --mechanics-every 200   --thermal-output-every 200   --mechanics-output-every 200   --summary-every 50   --output-dir /home/user/work/159/output/formal_p0p1_xbuild_50layers_fullmesh_m

cd /home/user/work/159/jax-fem

python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 0 \
  --mesh-length-scale 1.0 \
  --build-axis x \
  --base-side min \
  --scan-axis auto \
  --layers 50 \
  --hatch-lines-per-layer 10 \
  --scan-steps-per-layer 20 \
  --dt 1e-3 \
  --laser-power 100000000 \
  --absorptivity 1.0 \
  --powder-mode void \
  --ambient 300 \
  --preheat-temperature 300 \
  --convection-h 10 \
  --emissivity 0 \
  --bottom-thermal-bc fixed \
  --latent-heat 0 \
  --mechanics-model linear_elastic \
  --cooling-steps 0 \
  --mechanics-every 200 \
  --thermal-output-every 0 \
  --mechanics-output-every 200 \
  --summary-every 50 \
  --output-dir /home/user/work/159/output/debug_p0p1_old_like

  $ python 159_local/inp_thermal_stress_oneway_xbuild_p0p1_fixed.py
  --inp /home/user/work/159/schema/0119_c3d4_only.inp
  --max-cells 0
  --mesh-length-scale 1.0
  --build-axis x
  --base-side min
  --scan-axis auto
  --layers 50
  --hatch-lines-per-layer 10
  --scan-steps-per-layer 20
  --dt 1e-3
  --laser-power 100000000
  --absorptivity 1.0
  --powder-mode void
  --ambient 300
  --preheat-temperature 300
  --convection-h 10
  --emissivity 0
  --bottom-thermal-bc fixed
  --latent-heat 0
  --mechanics-model linear_elastic
  --cooling-steps 0
  --mechanics-every 200
  --thermal-output-every 0
  --mechanics-output-every 200
  --summary-every 50
  --output-dir /home/user/work/159/output/debug_p0p1_current_quad