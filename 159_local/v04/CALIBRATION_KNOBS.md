# 标定参数清单（后期导入真实值前必读）

状态：2026-07-09。当前所有值为**工程估计**，模型链路已验证但定量结果
依赖下述标定。导入真实值的推荐方式见文末。

## 一级旋钮（决定应力量级，必须标定）

### 1. 基板等效刚度 k_s —— `bottom_foundation_stiffness`

| | |
|---|---|
| 当前值 | `1.0e12` Pa/m（Winkler 地基模量，~25mm 钢板量级的粗估） |
| 生效条件 | `bottom_mechanics_bc = "elastic"` |
| 控制什么 | 近基板 3–5 层的应力量级与塑性区大小；全局回弹形态基本不敏感（实测 release 变形对刚夹持/弹性地基不变，均 1.33mm） |
| 方向性 | k_s ↑ → 层 1 应力 ↑（k_s→∞ 退化为刚性夹持，层 1 均值从 409 回到 ~1053 MPa） |
| 标定数据 | 首选：基板切除前的近基板残余应力测量（钻孔法/轮廓法）；次选：打印中基板背面应变片 |
| 物理参考 | 均匀受压圆板近似 k_s ≈ E_plate/t_plate：钢 210GPa/25mm ≈ 8.4e12；含夹具柔度取低一档，1e11–1e13 扫描 |

### 2. 应力松弛温度 T_relax —— `stress_relaxation_temperature`

| | |
|---|---|
| 当前值 | `1100.0` K（Ti64 常用区间 1073–1173K 的中值） |
| 配套 | `activation_reset_temperature` 应与其一致（=1100），新层以无应力热态进入 |
| 控制什么 | **整体残余应力量级的第一决定因子**：应力 ∝ 受约束冷却区间 ΔT = T_relax − T_final 的弹塑性响应 |
| 方向性 | T_relax ↑ → 全场应力 ↑（近似线性，直到屈服封顶接管） |
| 标定数据 | 全场残余应力测量（中子衍射/轮廓法）或 release 后变形量对比 |
| 物理含义 | 高于此温度视为应力瞬时松弛（蠕变/回复足够快）。Ti64 应力消除退火 ~1003–1073K，快速冷却下有效值偏高 |

## 二级旋钮（已有工程值，按需精化）

| 参数 | 当前值 | 控制什么 | 备注 |
|---|---|---|---|
| `yield_saturation_stress` | 1.15e9 Pa | 硬化封顶（~UTS） | 有拉伸曲线就换真值 |
| `hardening_table` | *_ext.csv 估计 | 塑性应变→应力斜率 | 出处列标 `verify` 的行需核对 |
| `old_layer_cooling_h` | 1.0e4 W/(m³K) | 移动窗口外旧层降温速率 | 影响层间温度史；有热电偶数据可标定 |
| `front_surface_loss_h` | 0（关闭） | 打印顶面对流近似 | 顶面是内部界面，外表面选择器覆盖不到；精化热史时开启并标定 |
| `conductivity_powder` | 1.0 W/mK | 粉床侧向散热 | 真实 0.1–0.3；仅逐道模式敏感 |
| `absorptivity` / `source_depth` | 0.5 / 5e-4 | 激光能量沉积 | 仅逐道模式（physfix 脚本）使用；fast-scan 不用激光 |
| `conductivity_liquid` | 28.7 W/mK | 熔池 Marangoni 等效 | 仅熔池路线（当前不可达）相关 |

## 如何导入真实值

**方式一（推荐）：写进材料 config JSON**。所有键都支持 config 读取，复制
`materials/Ti-6Al-4V/ti64_material_config_physfix.json` 为
`..._calibrated.json`，追加：

```json
{
  "bottom_mechanics_bc": "elastic",
  "bottom_foundation_stiffness": 8.4e12,
  "stress_relaxation_temperature": 1050.0,
  "activation_reset_temperature": 1050.0,
  "yield_saturation_stress": 1.10e9
}
```

然后运行脚本换 `--config` 指向新文件（CLI 显式传参仍会覆盖 config，
注意 `run_macro_fastscan_flash.sh` 里已显式写了这几个参数——标定时
直接改脚本或用附加参数覆盖，argparse 取最后一次出现的值）。

**方式二：命令行覆盖**（脚本末尾追加参数即可）：

```bash
bash run_macro_fastscan_flash.sh \
  --bottom-foundation-stiffness 8.4e12 \
  --stress-relaxation-temperature 1050 \
  --activation-reset-temperature 1050
```

## 标定流程建议

fast-scan 全件 54 分钟/次，参数扫描可行：
1. 固定 T_relax=1100，扫 k_s ∈ {1e11, 1e12, 1e13}，对比近基板应力/背面应变；
2. 固定标出的 k_s，扫 T_relax ∈ {1000, 1100, 1200}，对比全场应力量级或
   release 变形；
3. 逐道模式（physfix 脚本）只在标定完成后做局部高保真验证。
