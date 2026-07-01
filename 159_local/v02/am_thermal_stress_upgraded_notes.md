# 3D 粉末打印热-力耦合代码升级说明

## 已完成的代码修改

输出文件：`am_thermal_stress_upgraded.py`

### 1. 扫描路径升级

新增参数：

```bash
--layer-thickness
--hatch-spacing
--scan-pattern raster
--scan-rotation-per-layer
--jump-speed
--path-output
```

主要改动：

- `--layer-thickness` 优先于 `--layers`，用于按真实铺粉层厚自动计算层数。
- `--hatch-spacing` 优先于 `--hatch-lines-per-layer`，用于按真实 hatch 间距生成扫描线。
- `--scan-rotation-per-layer` 支持层间扫描方向旋转，例如 0°、90°、67°。
- `--jump-speed` 支持 hatch 之间的 laser-off jump，不再只能用原地 dwell 表示空行程。
- 每次运行会输出 `path_used.csv`，记录实际离散后的路径点、时间、模式、激光开关和前沿坐标。
- 外部 `--path-file` 支持可选 `front_coord` 列，用于把“激光位置”和“层激活前沿”解耦。

推荐路径文件格式：

```csv
time,x,y,z,front_coord,power,laser_on,layer,hatch,scan_id,mode
```

如果没有 `front_coord`，程序会退回使用 `laser_center[build_axis]` 作为激活前沿，并打印 warning。

---

### 2. 材料状态升级

新增积分点相态编码：

```python
STATE_VOID      = 0.0
STATE_POWDER    = 1.0
STATE_SOLID     = 2.0
STATE_MUSHY     = 3.0
STATE_LIQUID    = 4.0
STATE_SUBSTRATE = 5.0
STATE_SUPPORT   = 6.0
```

主要改动：

- 新增 `phase_quad` 作为积分点历史变量。
- `material_state` 不再只是根据单元平均温度临时显示，而是由 `phase_quad` 历史决定。
- 热物性函数现在区分 `void / powder / solid / mushy / liquid / substrate / support`。
- 新增液态材料属性输入：

```bash
--rho-liquid
--cp-liquid
--conductivity-liquid
--k-table-liquid
--cp-table-liquid
```

相态转化逻辑：

```text
未激活 void -> 激活后 powder
powder/solid -> T >= liquidus -> liquid
solid/liquid -> solidus <= T < liquidus -> mushy
liquid/mushy -> T < solidus -> solid
```

---

### 3. 力学参考温度升级

主要改动：

- 新增 `T_ref_quad`，表示积分点的应力自由参考温度。
- 热应变改为基于 `T_quad - T_ref_quad`，而不是基于铺粉激活温度。
- `T_ref_quad` 在 `liquid/mushy -> solid` 的凝固瞬间写入。
- 输出新增：

```text
stress_free_temperature
solidification_step
```

这比原来的 `activation_temperature` 更适合残余应力模拟，因为粉末铺上去并不等于形成承载应力的固体；真正的应力自由状态应在凝固时刻建立。

新增力学相态参数：

```bash
--mushy-mechanics-factor
--liquid-mechanics-factor
--reset-plastic-on-melt / --no-reset-plastic-on-melt
```

---

## 推荐运行命令

以 `yz` 为打印平面、沿 `+x` 方向堆叠为例：

```bash
python am_thermal_stress_upgraded.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --build-axis x \
  --base-side min \
  --layer-thickness 5e-5 \
  --hatch-spacing 8e-5 \
  --scan-axis y \
  --scan-rotation-per-layer 67 \
  --jump-speed 1.0 \
  --scan-speed 0.5 \
  --auto-scan-steps-from-speed \
  --laser-power 200 \
  --absorptivity 0.35 \
  --beam-radius 5e-5 \
  --source-depth 5e-5 \
  --powder-mode powder \
  --solidus-temperature 1878 \
  --liquidus-temperature 1928 \
  --latent-heat 2.9e5 \
  --mechanics-every 20 \
  --thermal-output-every 20 \
  --mechanics-output-every 20 \
  --cooling-steps 100 \
  --output-dir /home/user/work/159/output/upgraded_powder_scan
```

---

## 特定材料铺粉模拟需要提供的材料参数

是的，如果你要做特定材料的铺粉模拟，最好分别给出粉末态、固态、液态属性。最低限度建议如下。

### 热学属性

| 状态 | 必要参数 | 代码参数 |
|---|---|---|
| 粉末 | 密度、比热、有效导热率 | `--rho-powder`, `--cp-powder`, `--conductivity-powder`, `--k-table-powder`, `--cp-table-powder` |
| 固体 | 密度、比热、导热率 | `--rho-solid`, `--cp-solid`, `--conductivity-solid`, `--k-table-solid`, `--cp-table-solid` |
| 液体 | 密度、比热、导热率 | `--rho-liquid`, `--cp-liquid`, `--conductivity-liquid`, `--k-table-liquid`, `--cp-table-liquid` |
| 相变 | solidus、liquidus、潜热 | `--solidus-temperature`, `--liquidus-temperature`, `--latent-heat` |
| 表面/热源 | 发射率、吸收率 | `--emissivity`, `--absorptivity` |

温度表格 CSV 格式统一为：

```csv
T,value
300,15.0
500,17.2
800,21.5
```

---

### 力学属性

| 属性 | 代码参数 |
|---|---|
| 弹性模量 | `--young` 或 `--E-table` |
| 泊松比 | `--poisson` 或 `--poisson-table` |
| 热膨胀系数 | `--alpha` 或 `--alpha-table` |
| 屈服强度 | `--yield-table`，仅 `--mechanics-model j2_plastic` 需要 |
| 硬化模量 | `--hardening-table` |
| 液态/糊状区承载能力 | `--liquid-mechanics-factor`, `--mushy-mechanics-factor` |

液态通常不应承担固态残余应力，因此代码默认用很小的 `liquid_mechanics_factor` 表示近似应力释放。

---

## 下一步建议

1. 先用 1–3 层、小网格验证 `path_used.csv` 是否符合预期。
2. 在 ParaView 中检查 `material_state`、`stress_free_temperature`、`solidification_step`。
3. 用 `--scan-rotation-per-layer 0 / 90 / 67` 做三组对比。
4. 材料参数不完整时，先做敏感性分析，不要直接宣称定量残余应力精度。
