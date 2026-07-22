# V07 消融实验 — PARDISO 求解链优化选型

日期：2026-07-17。承接 `V07_CPU_MULTICORE_SOLVER.md`（pardiso 已为新默认候选，
68 ms/step）。本轮回答"pardiso 之后还有多少"：对求解链逐项消融，
real-slice 档（197k TET4 / 52,735 thermal dofs），repeat=2 discard-first=1。

## 1. 消融阶梯设计

每级只在上一级基础上加一件事，保证差值可归因：

| 级 | 内容 |
|---|---|
| base | 现状：`pypardiso.spsolve` 每步（矩阵 copy + phase 12 + 全量比对 + phase 33 + 双份 ia/ja 转换） |
| nocmp | 绕过 pypardiso 记账：每步单次 phase 13 裸调用 |
| cache-idx | nocmp + 跨步缓存 1-based int32 索引（pattern 精确比对失效） |
| phase23 | cache-idx + 符号分析只做一次（首解 phase 13，之后 phase 23） |
| fp32ir | phase23 的 fp32 分解（iparm(28)=1）+ 手动 fp64 迭代精化 |

另加 MKL 线程数扫描（base，8/16/24/32）。实现：
`159_local/v07/pardiso_variants.py`，由 v04 wrapper 的 `V07_PARDISO_MODE`
环境变量选择；不设置时行为与 v04 现状完全一致（75 单测通过）。

## 2. 结果

数据：`output/05_bench/v07_ablation/`。精度列为最终步 VTU 温度场对
spsolve 基线（`bench_realslice2_20260715_123329`）的 max|dT|。

| arm | wall(s) | solver ms/step | max\|dT\| vs spsolve | 判定 |
|---|---|---|---|---|
| base MKL=8 | 2.16 | 63.8 | 0 | 线程数不敏感 |
| base MKL=16 | 2.27 | 70.1 | 0 | 〃 |
| base MKL=24 | 2.19 | 64.2 | 0 | 〃 |
| base MKL=32 | 2.23 | 73.0 | 0 | 〃 |
| nocmp | 2.49 | 100.5 | 0 | **负收益，弃** |
| cache-idx | 2.54 | 104.6 | 0 | **负收益，弃** |
| **phase23** | **2.06** | **49.6（-27%）** | **0（逐位一致）** | **最优** |
| fp32ir | 未进全量 bench | 微基准判死 | fp32 残差 1.2e-07（IR 可救） | **弃，见 §3** |

phase23 复用统计：8 步 8 解只做 1 次符号分析、0 次 pattern 重建——
**激活推进不改变矩阵 pattern**（ersatz 刚度使全网格 pattern 恒定），
lane 长跑中符号分析近乎一次性成本。

## 3. 负结果记录（消融的主要价值）

1. **nocmp/cache-idx 比 base 还慢**（100/105 vs 68 ms）：pypardiso 的
   copy/比对/索引转换开销可忽略；且同一 pt 句柄上"单次 phase 13"比
   pypardiso 的"phase 12 + phase 33"拆分更慢（MKL 内部行为，未深究）。
   直觉上的"Python 层开销"不是瓶颈，别再往这个方向投入。
2. **fp32 分解在 MKL 2026.1 上慢 ~10-30x**（55k 合成系统：fp32 phase13
   2261 ms vs fp64 213 ms；fp64+同一套手工 iparm 275 ms，排除 iparm
   配置问题）。iparm(28) 功能上生效（残差 1.2e-07 符合 fp32 精度），
   纯粹是这版 MKL 的 fp32 kernel 慢。RTX 5080 FP64 弱 + MKL FP32 慢，
   混合精度路线两头堵死。
3. **MKL 线程数 8→32 全平**（±5 ms 噪声带内）：52k dofs 矩阵并行
   饱和点很低。与其他任务共存时 `MKL_NUM_THREADS=8` 零代价。

## 4. 最优解与用法

**`V07_PARDISO_MODE=phase23`**，solver 68→50 ms/step，温度场与 spsolve
逐位一致：

```bash
V07_PARDISO_MODE=phase23 python ... --xla-linear-solver pardiso ...
```

lane1 外推：solver 49,910 s（spsolve 基线）→ pardiso ≈6,360 s →
phase23 ≈**4,640 s**；全程 15.6 h → **≈3.0 h（≈5.2x 端到端）**。

验收（沿用 V07 §5）：首个正式 lane 跑用
`lane1_first5_track_ref_20260714_093503` 做同配置 VTU 对照，要求
T max_abs=0 量级；同时看 stderr 的 `[v07-pardiso phase23]` 统计行确认
`analyze_calls` 远小于 `solves`。稳定后可把 phase23 转正进
`_PardisoCustomSolver` 默认路径。

## 5. 优化后的格局与下一步

phase23 落地后 real-slice 每步：solver 50 ms + **assembly 67 ms**（GPU）
+ global_matrix 23 ms——**瓶颈翻回装配侧**。后续排序：

1. **CPU/GPU 流水线重叠**：solver(CPU 50ms) 与 assembly(GPU 67ms) 规模
   相当且异硬件，但隐式步进有真依赖（第 n 步装配需要第 n-1 步收敛解），
   完整重叠需配合 lagged Jacobian。
2. **modified Newton / 回代复用**：phase 33 回代实测仅 ~10 ms（55k 合成
   系统）。lane1 每步 ~2 次牛顿迭代，若同步内冻结 Jacobian，第二次求解
   50→10 ms。改变收敛路径不改变收敛解，需 lane 档验证迭代数不涨。
   real-slice 档每步 1 次迭代，bench 上不可见，只能 lane 实测。
3. 装配侧继续抠（v04 已 3.4x，边际收益待评估）。

## 6. 热力耦合与真实一层验证（2026-07-17 下午补充）

### 6.1 耦合 bench（新增 real-slice-mech 档：真实网格 + mechanics-every 4）

| arm | wall(s) | solver(s) | 说明 |
|---|---|---|---|
| trad2（v03 batch 2048 + 全优化关 + spsolve） | 40.30 | 18.77 | 真传统基线 |
| trad（仅旗标关，batch 保留 v04 默认） | 24.00 | 19.10 | 装配优化删不干净的教训 |
| xla-spsolve | 23.68 | 18.93 | 力学 spsolve ~7.6s/次主导 |
| xla-pardiso | 5.08 | 0.62 | 力学非对称矩阵加速 ~30x |
| **xla-phase23** | **4.83** | **0.48** | **vs trad2 = 8.3x 端到端** |

结论：耦合档位下传统链的瓶颈是力学 spsolve（80% wall），PARDISO 对
158k 非对称 J2 切线的加速比热矩阵更大——热力耦合是本优化收益最大的场景。
注意：该 bench 热载荷过小（max|u|~1e-19 m），u 一致性在此档无意义，
力学正确性由 6.2 裁决。

### 6.2 真实一层求解（真实 inp + 真实路径，STRIDE=5，1,959 步 / 3,936 次求解，
完整物理：3000W、J2、physfix、弹性地基、release）

| arm | wall | solver | 单次求解 |
|---|---|---|---|
| spsolve | 2,264 s（37.7 min） | 1,934 s | 491 ms |
| **phase23** | **433 s（7.2 min）** | **104 s** | **26 ms** |

- **端到端 5.2x**；solver 18.6x。长跑稳态单次求解 26 ms **优于** 8 步
  bench 的 50 ms——bench 把一次性 phase-13 摊进了 8 步均值，长跑摊销后
  几乎纯 phase 23。
- 复用统计：3,936 次求解仅 2 次符号分析（热/力学 pattern 各一次）、
  0 次重建——激活/recoat/cooling/release 全流程 pattern 稳定。
- 正确性：**T、sol、eq_plastic_strain 及全部激活/历史记账字段逐位一致
  （max_abs=0）**；u max 差 3.5e-7 m（rel 9.0e-6）、应力 rel ~5e-6、
  vm rel 1.2e-6——量级在力学 Newton `--mechanics-rel-tol 1e-5` 的容差带
  内，来源是不同线性求解器下的牛顿迭代路径差异，非线性求解器精度损失。
  需要更紧一致性时收紧 mech rel-tol 即可。
- 每层实测（stride-5）：spsolve ~38 min/层 → phase23 **~7 min/层**；
  stride-1 外推 ~3.0 h/层 → **~35 min/层**（lane1 实测 3.1 h/层交叉验证
  吻合）。全件 91 层 stride-1：264 h → 估 **~55 h**。
- 数据：`output/finescan_v05_full91_v07real_{spsolve,phase23}_20260717_154359/`
  与 `output/05_bench/v07_coupled/`。
- 顺手修复：`run_full_fine_scan_24h.sh` 的 PATH_CSV 指向 output 重组前
  旧位置（已改 `01_dev_early/`）；v03/v04 历史脚本同病未动。

## 7. 关联文件

- 变体实现：`159_local/v07/pardiso_variants.py`
- wrapper 接线：v04 `_PardisoCustomSolver._maybe_v07_variant()`
- bench 数据：`output/05_bench/v07_ablation/{threads_*,mode_*}.json`
- 微基准（phase/fp32 定时）：本文 §3，脚本未入库（/tmp/v07_fp32_check.py）
