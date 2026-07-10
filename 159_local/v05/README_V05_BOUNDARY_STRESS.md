# v05 —— 边界应力分布探究（boundary stress analysis）

日期：2026-07-09。目标：定量回答"边界处受约束是否产生更大的应力/屈服"，
并把零件–基板界面的分层风险量化出来。v05 不改 v03/v04 求解器（它们已经
过 147 个测试锁定），而是新增一个**边界应力后处理器**，直接消费
fast-scan / physfix 运行的 VTU 输出。

## 物理预期（文献线索——凭记忆列出，引用前请核对卷期页码）

| 文献 | 对本工作的输入 |
|---|---|
| Mercelis & Kruth, *Rapid Prototyping J.* 2006 | LPBF 残余应力经典解析模型：沿高度 "M" 形分布，顶面/近基板受拉、中部受压；切离基板后的应力重分布 |
| Parry, Ashcroft & Wildman, *Additive Manufacturing* 2016 | 热梯度机制（TGM）+ 扫描策略各向异性；**自由边缘处面内应力集中** |
| Zaeh & Branner, *Production Engineering* 2010 | **零件–基板界面边缘的剪切/剥离应力峰**驱动分层与开裂——即用户提出的"边界约束屈服"的最工程化形态 |
| Denlinger, Heigel & Michaleris, *J. Mater. Process. Technol.* 2015 | Ti-6Al-4V 的应力松弛效应、层间停留时间的影响（支持我们的 T_relax 旋钮） |
| Hodge, Ferencz & Solberg, *Comput. Mech.* 2014 | 部件级热-力耦合 LPBF 建模范式（quiet element / 激活法，与本仓库路线一致） |
| Liang et al., *Additive Manufacturing* 2018 | 修正固有应变法——我们 fast-scan flash 模式的近亲，可用于交叉验证 |

理论要点：
1. **自由侧表面**：面外牵引力为零 → 面内主应力在边缘旋转、法向分量卸载，
   等效应力在边缘一个特征长度内**升高**（约束释放的补偿），屈服带常沿边缘。
2. **基板界面周边**：零件收缩被基板拖拽，界面剪应力 τ 在周边达到峰值
   （中心趋零），叠加边角的剥离（peel）正应力——分层判据。
3. 沿高度的 "M" 形：底部受基板约束高拉，中部被后续层压缩，顶面再次受拉。

## v05 组件

| 文件 | 作用 |
|---|---|
| `postprocess_boundary_stress.py` | 核心后处理器，见下 |
| `run_v05_boundary_analysis.sh` | 对一个输出目录跑全套分析 |

### 后处理器输出

输入：运行目录（需要 `step_*<final>*.vtu`（约束态，弹簧激活）+ `release.vtu`）。

新增单元场（写入 `v05_boundary_<state>.vtu`）：
- `vm_cell` / `vm_quad_max`：单元均值与积分点最大 von Mises
- `sigma1` / `sigma3`：最大/最小主应力（拉正压负；开裂看 σ1）
- `yield_utilization`：vm / σy_eff(eqp)，σy_eff 含硬化与饱和帽。**≥1 即屈服面上**
- `edge_distance`：单元到最近**侧向自由表面**的距离（基板面除外）——
  画"应力 vs 边距"就能定量回答边界效应
- `plastic`：eqp > 1e-6 指示

界面牵引力（约束态专属，来自弹性地基 t = -k_s·u）：
- `interface_tractions.csv`：每个基板面片的剥离正应力 σ_peel = k_s·u_x
  （正=拉离基板）与剪应力 τ = k_s·√(u_y²+u_z²)，含面心坐标
- 周边 vs 中心统计——分层风险图

剖面 CSV：
- `profile_height.csv`：逐层 vm/σ1/屈服率统计（验证 "M" 形）
- `profile_edge.csv`：按边距分箱的应力统计（验证边缘集中）

文本报告：屈服热点 Top-N（坐标、层号、边距、利用率），
边缘带（<1 单元尺寸）与内部的应力/屈服率对比。

## 用法

```bash
bash 159_local/v05/run_v05_boundary_analysis.sh \
  /home/user/work/159/output/fastscan_flash_full91_<ts> \
  --foundation-stiffness 1.0e12
```

材料参数默认与 `ti64_material_config_physfix.json` 一致
（yield 955 MPa@300K、H=1.45 GPa、饱和 1.15 GPa），标定后请用 CLI 覆盖。
