# V07 跨工况横向对比 — jax-fem 标准样例 × phase23

日期：2026-07-17，2026-07-20 收尾（scal/spsolve 定案 + 解一致性核查完成，
见 §1.4 与 §1.3 更新）。剩余可选扩展见 §4.3。
承接 `V07_ABLATION.md`（AM 热力耦合已验证：真实一层 5.2x，T/eqp 逐位一致）。

目的：回答"优化是否只对 AM 工况有效"。方法：`bench_apps.py` 在
`jax_fem.solver.linear_solver` 层统一拦截，样例代码零改动，每 arm
输出快照到 `output/05_bench/v07_apps/{case}_{arm}_out/`。

## 1. 已有结果（三工况三种结局）

### 1.1 phase_field_fracture（交错 u/d，7,617 次求解，~1 万 dofs）— 大胜

| arm | wall(s) | 求解累计(s) |
|---|---|---|
| baseline（样例自带 spsolve） | 1,496 | 356.9 |
| **phase23** | 1,359 | **43.4（8.2x）** |

7,618 解 / 2 次符号分析（u、d 两套 pattern）/ 0 重建——多 pattern
缓存在交错求解下正常。wall 仅 1.1x：矩阵太小，瓶颈在装配与交错循环。

### 1.2 wave（200 隐式步，定矩阵，~几千 dofs 2D）— 回代捷径生效

| arm | 求解累计(s) |
|---|---|
| baseline（jax bicgstab GPU，库默认） | 20.1 |
| spsolve | 5.1 |
| **phase23** | **2.3（8.7x vs 默认）** |

backsolve_hits 199/200：全程 1 次分解 + 199 次 phase 33 纯回代
（~11 ms/步）。定矩阵时间步进是 phase23 最优场景。
注意精度信号：baseline(迭代法) 与 spsolve 的 Min u 差 3.2e-5
（0.680520 vs 0.680489），迭代容差 200 步累积——直接法阵营内部
（spsolve vs phase23）的对比待做（§4.2）。

### 1.3 scalability（3D 超弹性 50³，~397k dofs，2 次牛顿解）— 诚实负结果

| arm | 求解累计(s) |
|---|---|
| baseline（petsc bcgsl+ilu） | **7.5** |
| phase23 | 46.7（~23s/次分解，输 6x） |
| spsolve | **OOM 不可行**（2026-07-20 定案：SuperLU 填充 21.9 GB 实存 /
  142 GB 虚拟，19 min 后被内核 OOM 杀，dmesg 有记录） |

397k 3D 矢量问题 LU 填充过重，且良态问题 petsc 迭代 3.8s/解就收敛，
2 次求解也摊不掉符号分析。注意分档：PARDISO 在此规模**慢但可行且精度
正常**（vs petsc rel 1.4e-7），传统 spsolve 直接死于内存。

### 1.4 解一致性核查（2026-07-20，全部通过）

| 对比 | 差值 | 判定 |
|---|---|---|
| pff 力-位移曲线 baseline vs phase23 | rel 1.7e-8 | 交错非线性路径差异，验证曲线实质相同 |
| wave 终态 spsolve vs phase23 | **max_abs = 0（逐位一致）** | 1 次分解+199 次回代后仍逐位复现直接法 |
| wave 终态 库默认迭代 vs phase23 | 3.2e-5 | 差异归属迭代法容差累积（直接法阵营内部为 0） |
| scal u 场 petsc vs phase23（397k） | rel 1.4e-7 | petsc 迭代容差内，PARDISO 精度正常 |

脚本：`/tmp/v07_accuracy.py` 逻辑已并入本目录 `vtu_diff.py` 用法（pff 用
npz 直接比 forces）。

## 2. 适用边界结论（定稿）

phase23/PARDISO 直接法路线的甜区 = **中等规模（≲20 万 dofs）×
大量重复求解 × 非对称/条件数差/定矩阵**：

- AM 热力耦合（52k 热 + 158k 力学、~2 解/步 × 万步级）：全占 → 5-8x
- 交错多场（相场断裂类）：占"重复求解+多 pattern" → solver 8x
- 定矩阵步进（波动/线性瞬态）：最优场景 → 一次分解全程回代
- 大规模良态 3D 静力学（scalability 类）：**不适用，留给迭代法**

## 2.5 全量扩展（2026-07-20 下午，"全部都做"轮）

### A. 六个补充工况（baseline vs phase23，零改动拦截）

| 工况 | baseline solver | phase23 solver | 解一致性 | 结论 |
|---|---|---|---|---|
| **thermal_mechanical_full** | 54.2s / **490 次牛顿解** | **3.2s / 200 次** | rel 6.1e-8 | 双重收益：单解 17x + **牛顿迭代数 490→200**（petsc ilu 解不准导致牛顿多走 2.4x）；wall 161→80s |
| stokes（鞍点） | 0.58s（tfqmr+LU） | 0.62s | rel 2.4e-17 | **鲁棒性达标**：加权匹配扛住零对角块，速度平、精度机器级 |
| dendrite（361 步瞬态相场） | 62.9s（GPU bicgstab） | 67.5s | rel 1.2e-3* | **平/小负**：矩阵逐步变+良态，两边 ~0.18s/解打平；*界面演化放大迭代容差累积，非求解器错误 |
| serendipity（高阶单元） | 0.89s | 0.67s | rel 2.3e-15 | 小胜 |
| arc_length | 0.89s | 0.60s | **0.0（逐位）** | 全程仅 2 次线性解，信号弱 |
| periodic_bc | 两 arm 均在求解前抛 JAX 异常 | 同 | — | 样例自身问题（baseline 也挂），与求解器无关 |

### B. 规模阶梯（3D 超弹性，2 次牛顿解，solver 累计秒）

| dofs | petsc | phase23 | spsolve（传统） |
|---|---|---|---|
| 27,783 | **0.40** | 0.61 | 7.79 |
| 89,373 | **1.30** | 2.28 | 121.4 |
| 206,763 | **3.57** | 12.1 | 超时（>30min） |
| 397,953 | **7.48** | 45.4 | 内核 OOM（前测） |

- 良态 3D 弹性问题上迭代法全程胜，直接法劣势随规模从 1.5x 扩大到 6x
  （3D LU 填充超线性）；四档 max|u| 全一致。
- **传统 spsolve 在每一档都差 20–90x，之后直接不可行**——AM 工况"从
  spsolve 出发"的对比里 phase23 的收益是真实的，只是它的对手不该是也
  不会是良态大 3D 问题上的迭代法。

### C. 伴随基准（89k dofs，iter = 完整 value_and_grad）

| arm | iter 均值 | 备注 |
|---|---|---|
| petsc | 4.03s | 伴随=显式 A_T 从头解 |
| phase23 | 3.85s | 伴随仍分解 A_T |
| **phase23T（转置复用）** | 3.84s | 伴随=iparm(12) 转置解，**不构造 A_T、不做符号分析**；梯度与 phase23 **同至 12 位**、与 petsc 差 3e-9 |

诚实标注：本例收益仅 ~4%，原因是 `implicit_vjp` 在收敛点重新装配的 A
与牛顿最后一次分解（倒数第二迭代点）数值不同 → 转置回代前仍需一次数值
重分解（phase 22，~2.9s@89k）。**若接受"用牛顿末次分解做伴随"（lagged
adjoint，与 modified Newton 同族的近似）或在收敛点复用分解，伴随可降到
~0.1s 纯回代**——这是转置复用的完整收益形态，留待与 modified Newton 一起
评估（涉及梯度近似误差的标定）。机制本身（iparm(12)、多 pattern 正反向
共存、正确性）已验证到位。

### 扩展轮结论增补

1. **直接法的准确性有二阶收益**：tmfull 里牛顿迭代数减半不是求解速度，
   是"每步解得准→牛顿路径短"，这类收益在 ilu 迭代法基线上普遍被低估。
2. 适用边界修正（更细）：甜区判据里"病态/非对称"权重高于"规模"——
   dendrite（良态瞬态）打平，tmfull（耦合非对称）大胜，两者规模相近。
3. 转置复用是基础设施级正确的改动，完整收益绑定 lagged-Jacobian 路线。

## 3. 工具与数据位置

- 通用 harness：`159_local/v07/bench_apps.py`
  （arm ∈ baseline/spsolve/pardiso/phase23；已处理 CUDA_VISIBLE_DEVICES
  被样例覆盖的问题、MPLBACKEND=Agg 由驱动脚本设置）
- 驱动脚本：`159_local/v07/run_apps_comparison.sh`（本轮 8 个 arm 的复现入口）
- 场对比工具：`159_local/v07/vtu_diff.py <ref.vtu> <test.vtu>`
- 计时 json + 输出快照：`output/05_bench/v07_apps/`
- 变体实现（含 backsolve 捷径与 iparm(12) 转置复用）：
  `159_local/v07/pardiso_variants.py`（`VariantSolver.solve_transposed`）
- 扩展轮脚本：`bench_adjoint.py`（伴随三臂）、`bench_scal_ladder.py`
  （规模阶梯）；数据 `output/05_bench/v07_{apps,ladder,adjoint}/`

## 4. 续跑清单

### 4.1 ~~补跑 scal/spsolve~~（已定案 2026-07-20：OOM 不可行，见 §1.3）

### 4.2 ~~解一致性核查~~（已完成 2026-07-20：全部通过，见 §1.4）

### 4.3 补充工况（可选扩展，均可用 bench_apps.py 直接跑）

1. **dendrite**（相场枝晶 + 热耦合，瞬态多步）——预期同 1.1/1.2 混合特征。
2. **forming / updated_lagrangian**（大变形接触/成形，强非线性牛顿）——
   updated_lagrangian 默认网格太小（9×2×2），需先放大网格再测。
3. **stokes**（鞍点系统，baseline=petsc tfqmr+lu）——测 PARDISO mtype=11
   加权匹配对零对角块的鲁棒性，正确性优先、规模其次。
4. **scalability 规模阶梯**（Nx=20/30/40/50 → ~25k/86k/205k/397k dofs）——
   把 §2 的"≲20 万 dofs"边界从单点变成曲线，找 direct/iterative 交叉点。

### 4.4 更远的后续（出自 V07_ABLATION.md §5）

- modified Newton（phase 33 回代复用）lane 档实测——solver 侧最后大项。
- CPU/GPU 流水线重叠（需 lagged Jacobian 配合）。
- 转正决策：phase23 是否设为 `--xla-linear-solver pardiso` 的默认行为
  （建议先过一次正式 lane 的 VTU 验收，见 V07_ABLATION.md §4）。

## 5. 复现注意

- 样例快照目录会被重跑覆盖，续跑前如需保留旧数据先改名。
- scalability 样例源码把 CUDA_VISIBLE_DEVICES 设成 "2"（本机只有 1 块
  GPU）——bench_apps.py 已通过预初始化 JAX 后端规避，直接跑样例本体则
  会静默退回 CPU，勿直接跑。
- wave 的 baseline 是迭代法，与直接法差 ~3e-5 属容差累积，不是 bug。
