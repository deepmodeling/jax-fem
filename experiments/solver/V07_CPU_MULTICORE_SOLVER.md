# V07 — CPU 多核线性求解优化（pardiso 后端）

日期：2026-07-15。
定位：v04 wrapper 的线性求解器升级，不改任何物理；v05/v06 之上可直接叠加。
实现落点在 `159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py`
（v07 不新建 wrapper，本目录只承载方案文档与后续多核路线）。

## 1. 问题定性

lane1 first5 stride-1 参考解（197,266 TET4 / 52,735 thermal dofs，46,599 步，
93,552 次线性求解）的 profile 实证：

| 项 | 数值 |
|---|---|
| 总 wall | 56,042 s（15.6 h） |
| solver stage | 49,910 s = **89%** |
| 单次求解 | ≈533 ms（SciPy spsolve，单线程 SuperLU） |

机器有 32 逻辑核 + RTX 5080，但旧默认 spsolve 只用 1 核：
GPU 装配完矩阵后，31 个核和 GPU 一起等一个核做稀疏 LU。

历史包袱：v04 早期在 tiny/small/medium 档（20–500 cells）反复得出
"spsolve 最优、GPU/迭代法更慢"，但那个规模下单次直接解只要 2 ms，
任何多线程/GPU 开销都摊不平。**小档结论不能外推**——real-slice
（真实网格）档才是裁决依据。

## 2. 方案：MKL PARDISO 多线程直接法

选型对比（real-slice 档，真实 197k 网格，repeat=2 discard-first=1 warm 样本）：

| solver | wall(s) | solver stage /step | 最终 VTU max_abs_T_diff vs spsolve | 判定 |
|---|---|---|---|---|
| spsolve（基线） | 5.77 | 490 ms | — | 旧默认 |
| **pardiso** | **2.31** | **68 ms（7.1x）** | **0.000e+00（逐位一致）** | **新默认候选** |
| jax-precond (bicgstab+jacobi, GPU) | 2.57 | 104 ms（4.7x） | 2.1e-04 K | 备选；迭代容差误差 |
| jax-cg-no-check (GPU) | 4.77 | 390 ms | T_min 掉到 260 K，非物理 | 弃用 |
| jax-gmres (GPU) | 436 | 54 s | 发散 | 弃用 |

选 pardiso 而非 GPU 迭代法的理由：

1. **精度零损失**：直接法换直接法，T 场逐位一致。参考解（lane1 类）
   工作对精度敏感，迭代法 2e-4 K 级偏差在真实激光功率（ΔT~3000 K）
   下的相对行为未标定。
2. **更快**：68 ms vs 104 ms。RTX 5080 的 FP64 是 FP32 的 1/64，
   GPU 在 float64 直接法上无优势（v04 已定论），迭代法收敛次数
   又依赖矩阵条件数，鲁棒性不如直接法。
3. **力学方程同样适用**：J2 塑性切线矩阵非对称也能解
   （PARDISO mtype=11），medium 档热-力耦合验证 T 逐位一致、
   u 差 4.5e-13（机器噪声）。
4. GPU 继续做它擅长的装配（v04 的 3.4x 装配优化保留）。

## 3. 实现

- 依赖：`pip install pypardiso`（pypardiso 0.4.7 + MKL 2026.1.0，已装入
  `jax-fem-env`）。
- 接入点：`jax_fem/solver.py` 已有的 `custom_solver` 钩子
  （`linear_options['custom_solver']` 为可调用对象，签名
  `(A, b, x0, linear_options)`，A 是 PETSc AIJ）。**零库改动**。
- wrapper 侧（v04 文件）：
  - `_PardisoCustomSolver` 类：PETSc `getValuesCSR()` → SciPy CSR
    （索引 astype int32）→ `pypardiso.spsolve`；实例持有
    `PyPardisoSolver` 句柄跨步复用；`__deepcopy__` 返回自身，
    躲开选项重写的深拷贝。
  - `_SOLVER_CHOICE_TO_KEY` 增加 `"pardiso" -> "custom_solver"`；
    `LINEAR_SOLVER_KEYS` 增加 `custom_solver`；`_solver_label` 认
    `label` 属性；profile meta 的 `linear_solver_options` 对不可
    JSON 化的值写 label。
  - pypardiso 仅在求解时延迟 import，选项层保持纯 Python 可导入
    （CI 无 MKL 也能跑单测）。
  - 失败自动回退 spsolve 的既有逻辑对 pardiso 同样生效。
- bench：`bench_mech100_xla.py` `SOLVER_FLAGS` 增加 `pardiso` 档。
- 单测：`tests.test_v04_xla_wrapper` 74 tests OK。

## 4. 用法

```bash
# 任何 lane / fine-scan 脚本追加：
--xla-linear-solver pardiso

# 基准复核：
python 159_local/v04/bench_mech100_xla.py --tier real-slice \
  --solvers spsolve pardiso --repeat 2 --discard-first 1
```

预期收益（按 lane1 profile 外推）：solver 49,910 s → ≈7,000 s，
全程 15.6 h → **≈3.7 h（≈4.2x 端到端）**。

## 5. 验收与注意

- 换求解器后的首个正式运行，用 `04_lanes/lane1_first5_track_ref_20260714_093503`
  （spsolve 基线）做同配置 VTU 对照，要求 T max_abs=0 量级复现。
- medium 档 solver stage pardiso 反而略慢（0.57 s vs 0.32 s）——
  小矩阵多线程开销所致，**不要**据此回退默认；真实规模以 real-slice 为准。
- MKL 线程数默认取物理核；如需限制（与其他任务共存）用
  `MKL_NUM_THREADS` 环境变量。
- 首次调用有 MKL 初始化开销（tiny 档看到的 0.69 s），长跑摊销后可忽略。

## 6. 后续多核路线（未做）

1. **符号分解复用**：网格 sparsity 固定，PARDISO phase 11（analyze）
   可一次复用；PyPardisoSolver 按矩阵指纹缓存，激活拓扑变化时失效，
   需要测命中率再决定是否值得手工管理 phase。
2. **CPU/GPU 流水线重叠**：GPU 装配第 n 步时 CPU 解第 n-1 步
   （solver 降到 68 ms 后，装配 ~60 ms 与之相当，重叠收益上限 ~2x）。
3. 迭代法+warm start 作为 GPU 侧长线（需先解决 2e-4 偏差的标定）。

## 7. 关联记录

- 基准数据：`output/05_bench/bench_realslice2_20260715_123329/comparison.json`
  与 `bench_medium_pardiso_20260715_123440/`。
- 决策日志：`159_local/v04/XLA_UPGRADE_ROADMAP.md` 2026-07-15 条目。
- lane1 基线 profile：`output/04_lanes/lane1_first5_track_ref_20260714_093503/profile.json`。
