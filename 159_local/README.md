# Macro-Scale LPBF Thermal-Mechanical Residual Stress Model (159_local)

维护中的最新版本：**v05**（2026-07-10）。本文档描述当前模型的物理表述、两个
维护级运行入口与结果查看方式；历史演进见文末版本表。物理审查/修复台账见
`v03/V03_PHYSICS_REVIEW_AND_FIXES.md`，标定旋钮见 `v04/CALIBRATION_KNOBS.md`。

## Formulation

### Governing Equations

零件网格 $\Omega$（TET4，实测网格 197k 单元）内求解单向耦合的瞬态热传导与
准静态力学平衡。材料按激活状态分区：已打印 $\Omega_p(t)$、当前粉末层、以及
未铺粉的 void 区（quiet-element 法：void 单元保留在网格中，导热缩小
$10^{-6}$、热质量保持全值以避免虚假扩散与病态方程）。

热方程（隐式 Euler，4 点积分保证质量阵满秩）：

$$
\begin{align*}
\rho C_p \frac{T^n - T^{n-1}}{\Delta t} &= \nabla \cdot (k \nabla T^n) + q_{\textrm{laser}} - q_{\textrm{sink}} &&\textrm{in } \Omega, \\
T^n &= T_{\textrm{plate}} &&\textrm{on } \Gamma_{\textrm{base}}, \\
k\nabla T^n \cdot \boldsymbol{n} &= \chi_p \big( q_{\textrm{conv}} + q_{\textrm{rad}} \big) &&\textrm{on } \Gamma_{\textrm{ext}},
\end{align*}
$$

其中 $\Gamma_{\textrm{ext}}$ 为**网格外表面**（任意曲面零件，按"面仅属于一个
单元"判定），$\chi_p$ 为 printed 掩码（void 面无物理表面、不参与换热）；
$q_{\textrm{laser}}$ 为体积高斯热源（面内 $e^{-2r^2/r_b^2}$、沿堆积方向指数
衰减，仅逐道模式启用）；$q_{\textrm{sink}}$ 为移动热窗口外旧层的体积冷却
近似。层间铺粉停留（recoat，默认 10 s）与终冷用大步长隐式步覆盖。

### Consolidation-on-Activation（宏观固化模型）

能量核算表明 1 mm 宏观层参数化下熔池不可解析（能量密度差 ~25×），故采用
部件级标准做法：层激活即固化，应力自由参考温度取**松弛温度** $T_{\textrm{relax}}$
（标定旋钮，Ti64 取 1073–1173 K），新层以 $T_{\textrm{relax}}$ 无应力热态
进入（避免 GPa 级载荷阶跃），残余应力来自受约束的冷却收缩：

$$
\boldsymbol{\varepsilon}_{\textrm{th}} = \alpha(T)\,\big(T - T_{\textrm{relax}}\big)\,\boldsymbol{I}.
$$

### Mechanics: Incremental J2 with Stored Plastic Strain (v05)

力学为小应变增量 J2 塑性，**塑性应变张量 $\boldsymbol{\varepsilon}_p$ 按积分点
存储**（v05 核心升级：仅存标量 eqp 时不相容应变无法锁入，卸约束后残余应力
恒为零）：

```math
\begin{align*}
\boldsymbol{\varepsilon}_e &= \boldsymbol{\varepsilon}(\boldsymbol{u}) - \boldsymbol{\varepsilon}_{\textrm{th}} - \boldsymbol{\varepsilon}_p^{\,n-1}, \\
\boldsymbol{\sigma}_{\textrm{tr}} &= \lambda\,\textrm{tr}(\boldsymbol{\varepsilon}_e)\boldsymbol{I} + 2\mu\,\boldsymbol{\varepsilon}_e, \qquad
s_{\textrm{eq}} = \sqrt{\tfrac{3}{2}\,\boldsymbol{s}:\boldsymbol{s}}, \\
\Delta\bar{\varepsilon}_p &= \frac{\big\langle s_{\textrm{eq}} - \sigma_y(\bar{\varepsilon}_p^{\,n-1}, T)\big\rangle_+}{3\mu + H_{\textrm{eff}}}, \qquad
\sigma_y = \min\!\big(\sigma_{y0}(T) + H(T)\,\bar{\varepsilon}_p,\; \sigma_{\textrm{sat}}\big), \\
\boldsymbol{\sigma}^n &= \boldsymbol{\sigma}_{\textrm{tr}} - 3\mu\,\Delta\bar{\varepsilon}_p\,\frac{\boldsymbol{s}}{s_{\textrm{eq}}}, \qquad
\boldsymbol{\varepsilon}_p^{\,n} = \boldsymbol{\varepsilon}_p^{\,n-1} + \tfrac{3}{2}\Delta\bar{\varepsilon}_p\,\frac{\boldsymbol{s}}{s_{\textrm{eq}}}.
\end{align*}
```

径向返回给出一致切线（Newton + line search + 迭代上限）；硬化饱和帽
$\sigma_{\textrm{sat}}$（~UTS）防止线性硬化在夹持奇异区外推出非物理应力。

边界条件：底面为 **Winkler 弹性地基**（基板有限柔度，
$\boldsymbol{t} = -k_s\,\boldsymbol{u}$，$k_s$ 为一级标定旋钮；刚性夹持是
$k_s \to \infty$ 极限），侧面/顶面自由。**release**（切离基板）：终冷后以
3-2-1 锚点重解，$\boldsymbol{\varepsilon}_p$ 自建造态继承——锁入不相容应变
给出真实残余应力与回弹。

## Execution

两个维护级入口（均从任意目录直接运行，物理内核一致 = v05 全套修复）：

**1. 快速扫描 —— 全件 91 层，约 1 小时**（层聚合 flash 模式，无逐道激光；
用于全件应力/变形形态、k_s / T_relax 标定扫描、日常迭代）：

```bash
bash /home/user/work/159/jax-fem/159_local/run_fast_scan.sh
```

**2. 全件精细扫描 —— 91 层逐道路径，约 24 小时**（真实扫描路径按
STRIDE=16 抽稀至 5.2 万热步，保留道次级热瞬变；能量守恒由 dt 自动放大保证）：

```bash
STRIDE=16 MECH_EVERY=300 \
  bash /home/user/work/159/jax-fem/159_local/run_full_fine_scan_24h.sh
```

STRIDE=1 为全分辨率（~264 h，待求解器效率阶段）。两个入口均接受所有
v03/v04/v05 CLI 覆盖参数（argparse 取最后出现的值）。

运行后的边界应力分析（主应力/屈服利用率/边距剖面/基板界面剥离与剪切）：

```bash
bash 159_local/v05/run_v05_boundary_analysis.sh <输出目录>
```

## Results

用 *ParaView* 查看（先 Threshold `printed >= 0.5`）：

| 文件 | 内容 |
|---|---|
| `release.vtu` | 切离基板后的最终残余应力 + 回弹变形（`u` + Warp） |
| `step_<末步>_cooling.vtu` | 冷却完成、仍在基板上的约束态应力 |
| `v05_boundary/v05_boundary_released.vtu` | 预合成的 `vm_quad_max` / `sigma1` / `yield_utilization` / `edge_distance` |
| `v05_boundary/interface_tractions_constrained.csv` | 基板界面剥离/剪切牵引力（分层风险） |
| `v05_boundary/report_*.txt` | 屈服热点 Top-N 文本报告 |

注意：中间步 VTU 中 `mechanics_valid=0` 的文件应力为全零占位（该步无力学
求解），两个维护入口的输出节奏已对齐、不产生此类文件。定量使用前必须完成
`v04/CALIBRATION_KNOBS.md` 的 $k_s$ 与 $T_{\textrm{relax}}$ 标定，材料表
高温段估计行（`materials/Ti-6Al-4V/*_ext.csv` 中 source 含 `verify`）需
替换为实测/文献值（补数优先级见 `materials/Ti-6Al-4V/TODO.md`）。

## Version Lineage

| 版本 | 内容 |
|---|---|
| v01/v02 | 网格读取（`read_tet4_inp`）与早期原型 |
| v03 | 完整热-力主求解器 + 2026-07-08 物理修复（外表面换热、底面容差、recoat、冷却、激活重置、松弛温度、径向返回、积分阶、弹性地基、饱和硬化…详见修复台账） |
| v04 | XLA/性能层（装配 batch 修复 3.4×、residual-only 检查、26 段 profiling、benchmark harness）+ flash 快扫模式 |
| **v05** | **增量塑性（ε_p 张量、release 状态继承）+ 边界应力分析套件**；两个维护入口指向此版 |

## References

[1] Mercelis, P. & Kruth, J.-P. Residual stresses in selective laser sintering and selective laser melting. *Rapid Prototyping Journal* (2006). — M 形应力分布与解析模型。
[2] Denlinger, E., Heigel, J. & Michaleris, P. Residual stress and distortion modeling of electron beam direct manufacturing Ti-6Al-4V. / 及其 2015 层间停留时间研究 — 应力松弛温度机制。
[3] Parry, L., Ashcroft, I. & Wildman, R. Understanding the effect of laser scan strategy on residual stress in selective laser melting. *Additive Manufacturing* (2016). — 自由边缘应力集中。
[4] Zaeh, M. & Branner, G. Investigations on residual stresses and deformations in selective laser melting. *Production Engineering* (2010). — 基板界面分层。
[5] Hodge, N., Ferencz, R. & Solberg, J. Implementation of a thermomechanical model for the simulation of selective laser melting. *Computational Mechanics* (2014). — 部件级激活法范式。
[6] Liao, S. et al. Efficient GPU-accelerated thermomechanical solver for residual stress prediction in additive manufacturing. *Computational Mechanics* (2023). — jax-fem 路线的 GPU 求解参考。
[7] Mills, K.C. *Recommended Values of Thermophysical Properties for Selected Commercial Alloys* (2002). — 材料热物性金标准。

（文献条目凭记忆整理，正式引用前请核对卷期页码。）
