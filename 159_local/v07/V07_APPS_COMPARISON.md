# V07 跨工况横向对比 — jax-fem 标准样例 × phase23

日期：2026-07-17。状态：**暂停中，7/8 arm 已完成**，续跑与收尾清单见 §4。
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
| spsolve | 未跑完即暂停（预期更慢，见 §4.1） |

397k 3D 矢量问题 LU 填充过重，且良态问题 petsc 迭代 3.8s/解就收敛，
2 次求解也摊不掉符号分析。

## 2. 当前适用边界结论（初版，待 §4 收尾后定稿）

phase23/PARDISO 直接法路线的甜区 = **中等规模（≲20 万 dofs）×
大量重复求解 × 非对称/条件数差/定矩阵**：

- AM 热力耦合（52k 热 + 158k 力学、~2 解/步 × 万步级）：全占 → 5-8x
- 交错多场（相场断裂类）：占"重复求解+多 pattern" → solver 8x
- 定矩阵步进（波动/线性瞬态）：最优场景 → 一次分解全程回代
- 大规模良态 3D 静力学（scalability 类）：**不适用，留给迭代法**

## 3. 工具与数据位置

- 通用 harness：`159_local/v07/bench_apps.py`
  （arm ∈ baseline/spsolve/pardiso/phase23；已处理 CUDA_VISIBLE_DEVICES
  被样例覆盖的问题、MPLBACKEND=Agg 由驱动脚本设置）
- 驱动脚本：`159_local/v07/run_apps_comparison.sh`（本轮 8 个 arm 的复现入口）
- 场对比工具：`159_local/v07/vtu_diff.py <ref.vtu> <test.vtu>`
- 计时 json + 输出快照：`output/05_bench/v07_apps/`
- 变体实现（本轮新增 backsolve 捷径）：`159_local/v07/pardiso_variants.py`

## 4. 续跑清单（按优先级）

### 4.1 补跑 scal/spsolve（传统直接法参照，预计很慢，给 1h 超时）

```bash
cd /home/user/work/159/jax-fem && source ~/miniforge3/etc/profile.d/conda.sh \
  && conda activate jax-fem-env && export PYTHONPATH=$PWD MPLBACKEND=Agg
timeout 3600 python 159_local/v07/bench_apps.py \
  applications/scalability/example_forward.py spsolve \
  output/05_bench/v07_apps/scal_spsolve.json
# 超时本身就是结论：记"SuperLU 397k 3D 不可行（>1h）"
```

### 4.2 解一致性核查（每工况一条判据）

```bash
# pff：力-位移验证曲线（对文献 ref 的判据）
python - <<'EOF'
import numpy as np
a = np.load('output/05_bench/v07_apps/pff_baseline_out/sol.npz')
b = np.load('output/05_bench/v07_apps/pff_phase23_out/sol.npz')
print('force max_abs_diff:', np.max(np.abs(a['forces']-b['forces'])))
EOF
# wave：终态场，直接法阵营内部应逐位或 ~1e-15
python 159_local/v07/vtu_diff.py \
  output/05_bench/v07_apps/wave_spsolve_out/vtk/u_199.vtk \
  output/05_bench/v07_apps/wave_phase23_out/vtk/u_199.vtk
# scal：phase23 vs petsc 基线（预期差 ~迭代容差）
python 159_local/v07/vtu_diff.py \
  output/05_bench/v07_apps/scal_baseline_out/vtk/u_classic_50x50x50.vtu \
  output/05_bench/v07_apps/scal_phase23_out/vtk/u_classic_50x50x50.vtu
```

### 4.3 补充工况（扩大覆盖面，均可用 bench_apps.py 直接跑）

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
