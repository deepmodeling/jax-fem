# mech100 XLA 升级路线图（v04）

目标：把 `am_thermal_stress_macro_intersection_mech100_XLA.py` 从"JAX/GPU 线性求解器尝试层"演进为"面向 GPU 高效执行的热-力求解流程"，同时保持物理流程和输出字段的字节级兼容。

## 当前状态（2026-07-07）

- v04 wrapper 已接回 v03 真实运行逻辑：加载
  `159_local/v03/am_thermal_stress_macro_intersection_mech100.py`，复用其
  `read_config()` / `build_parser()` / `main()`，不再依赖尚不存在的
  `default_solver_options()` / `run()` 接口。
- v04 通过 monkey-patch `base.solver` 改写 Newton 线性求解器配置，并保留
  `spsolve` fallback。fallback 发生时，benchmark 禁止声明 GPU speedup。
- profiling 已能输出固定 stage：
  `setup / activation / quad_state / material / history / postprocess /
  dof_to_quad / nonlinear_solve / nonlinear_solve_overhead /
  bc_initial_guess / residual_vector / residual_flatten / residual_bc /
  residual_projection / solver / conversion / transfer / local_assembly /
  global_matrix / assembly / cell_jacobian / cell_residual / face_jacobian /
  face_residual / residual_scatter / io / python_overhead`。
  其中 `setup` 以第一次 thermal/mechanical solve 调用为边界，记录 mesh /
  Problem 初始化、路径生成、启动写文件等首步之前的成本；
  `nonlinear_solve` 记录完整 `jax_fem.solver.solver(...)` 外层 Newton 调用，
  `nonlinear_solve_overhead` 是扣除 `local_assembly/global_matrix/solver/
  conversion/transfer/bc_initial_guess/residual_vector` 后的剩余外层
  overhead；`residual_flatten/residual_bc/residual_projection` 是
  `residual_vector` 的诊断明细，不参与二次扣减；`solver`、
  `local_assembly` 和 `global_matrix` 来自
  `jax_fem.solver._timing_record`，`assembly` 是两个 assembly 子阶段的汇总，
  `io` 来自 v03 输出函数包裹，
  `activation/quad_state/material/history/postprocess` 来自 v03 主循环纯 Python
  函数包裹，未归因 wall time 进入 `python_overhead`。
  JAX 路径已经把 PETSc/BCOO 稀疏转换记入 `conversion`，并把
  JAX iterative kernel 与 residual check 记入 `solver`，避免把外层
  `linear` 总时间重复计入 solver。当前实现已绕过 SciPy CSR 中间对象，
  并为固定 sparsity pattern 缓存 BCOO indices 和对角线位置；cache
  同时覆盖同一 PETSc Mat 对象和不同 Mat 对象但 CSR 结构相同的路径，
  后续只更新 values。
- tiny benchmark 已可端到端跑真实物理 driver：

  ```bash
  cd /home/user/work/159/jax-fem
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
  python 159_local/v04/bench_mech100_xla.py \
    --tier tiny --solvers spsolve jax-cg jax-cg-no-check \
    --repeat 2 --discard-first 1 \
    --out /tmp/v04_bench_tiny_repeat2 \
    --json /tmp/v04_bench_tiny_repeat2/combined.json
  ```

  当前 `jax` tiny 路线已能完成真实 run，不再因 `precond=False` 触发
  `NoneType` matmul fallback。benchmark harness 已支持 `--repeat` 和
  `--discard-first`，用于区分 JAX cold compile / cache warm-up 与稳态样本。
  单次顺序运行不能作为 GPU 加速证据。
- 新增保精度截断试跑入口
  `159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh`。该脚本
  对齐 v03 的 `run_macro_intersection_h60_mech100_XLA.sh`：同一 material
  config、同一 `.inp`、同一 h60 path CSV、同一 heat-source / activation /
  mechanics cadence，并保留 v03 的 `jax_solver(precond=False)`、GPU backend
  和 spsolve fallback；核心仿真差异仅为 `--max-print-layers 5`，另使用
  独立输出/profile 目录，避免覆盖 v03 全层结果。
  当前 h60 CSV 覆盖 91 层，前 5 层包含 46,459 个扫描步，因此该脚本是
  v03 参数不变的截断精度试跑，不是 smoke benchmark。
- JAX 求解器参数已经接到真实 `jax_fem.solver`：
  `--xla-jax-method {bicgstab,cg,gmres,spsolve}`、`--xla-jax-tol`、
  `--xla-jax-atol`、`--xla-jax-maxiter`，以及 GMRES 的
  `restart/solve_method`。新增的
  `--xla-jax-skip-residual-check` 可显式关闭每次 JAX solve 后额外的
  `A @ x - b` residual matvec；默认仍保持检查开启。关闭显式 residual
  check 时，`jax_solve()` 仍会检查 JAX iterative solver 返回的 `info`
  （`bicgstab/cg/gmres`），非零时直接抛出 `RuntimeError`，避免不收敛解静默
  进入后续热-力流程；`spsolve` 是 JAX experimental sparse direct solver，
  无 iterative `info` 返回，因此 residual check 由 CLI 显式控制。
  tiny repeat=2、discard-first=1 的 warm 样本显示，JAX 候选仍慢于 spsolve：

  | solver | wall(s) | solver(s) | conversion(s) |
  |---|---:|---:|---:|
  | spsolve | 1.30 | 0.002 | 0.000 |
  | jax-cg | 1.39 | 0.069 | 0.001 |
  | jax-cg-no-check | 1.40 | 0.072 | 0.001 |

  与上一版 profiling（conversion 约 `0.334s`）相比，直接 BCOO 构造
  和 sparse structure cache 把 tiny 档 warm conversion 降到约
  `0.001s`。但 JAX solver stage 仍约 `0.07s`，慢于 spsolve 的
  约 `0.002s`。因此下一步应继续处理 JAX kernel 编译/复用、求解器闭环和
  主循环固定开销，而不是直接上 full-loop `lax.scan`。
- Phase 2 第一版主循环缓存已落地：v04 wrapper 对 v03 的
  `compute_layer_on_scan_cells*` 和 `compute_moving_window_cells*`
  安装 activation mask cache。cache key 由当前层号、稳定 mesh/cell
  数组 identity、`active_window_below_layers/layers/layer_thickness`
  组成，覆盖 centroid/intersection 两类几何。一个真实 2 层 x 每层
  3 scan step smoke run 中，profile 记录 `activation_cache_misses=2`、
  `activation_cache_hits=4`，说明同层后续 step 已复用全 cell 布尔掩码。
- benchmark 新增 `small-loop` 档：80 cells、2 层、每层 8 scan step、
  mechanics 关闭，用于在不进入大力学成本的前提下观察主循环稳态。
  `setup` 拆分后的 `repeat=2, discard-first=1` warm 样本来自
  `/tmp/v04_bench_small_loop_setup_repeat2/combined.json`：

  | solver | wall(s) | setup(s) | solver(s) | conversion(s) | assembly(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | spsolve | 8.18 | 0.70 | 0.038 | 0.000 | 6.39 | 1.03 | 511 |
  | jax-cg-no-check | 9.22 | 0.67 | 1.646 | 0.010 | 5.79 | 1.10 | 576 |

  旧版 small-loop 表格没有 `setup` stage，导致 `python_overhead`
  同时包含启动/初始化成本和循环内 Python 成本；新 profile 显示 warm
  setup 约 `0.7s`，循环剩余 `python_overhead` 约 `64-68 ms/step`。

  该档记录 `activation_cache_misses=2`、`activation_cache_hits=14`，
  与 2 层 x 8 step 的预期一致。JAX conversion 已不是主瓶颈；
  `jax-cg-no-check` 的 solver stage 仍为约 `103 ms/step`，而
  spsolve 为约 `2.35 ms/step`。因此当前不能声明 GPU/JAX speedup，
  小规模默认解算器仍应保持 spsolve。

  2026-07-07 追加的 JAX sparse-cache 诊断显示，单纯按 PETSc Mat
  Python 对象缓存时，真实 small-loop 为 `jax_bcoo_cache_misses=16`、
  无 hit（`/tmp/v04_bench_small_loop_cache_diag/combined.json`）。升级为
  最近 CSR pattern cache 后，同档单次 cold run 为
  `jax_bcoo_cache_misses=1`、`jax_bcoo_cache_hits=15`，
  `conversion=0.021s`（`/tmp/v04_bench_small_loop_pattern_cache/combined.json`）。
  这证明 BCOO structure cache 已跨对象复用，但 conversion 仍只占总
  wall 的约 `0.15%`，后续主攻仍应是 assembly / JAX kernel / setup /
  Python 主循环。

  2026-07-07 进一步拆分主循环子阶段后，`repeat=2, discard-first=1`
  warm 样本来自
  `/tmp/v04_bench_small_loop_loopstages_repeat2_v2/combined.json`：

  | solver | wall(s) | setup | activation | quad_state | material | history | postprocess | solver | conversion | assembly | python_overhead | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | spsolve | 8.00 | 0.68 | 0.001 | 0.040 | 0.230 | 0.040 | 0.014 | 0.035 | 0.000 | 6.23 | 0.71 | 500 |
  | jax-cg-no-check | 9.25 | 0.70 | 0.001 | 0.036 | 0.222 | 0.044 | 0.011 | 1.217 | 0.005 | 6.29 | 0.70 | 578 |

  该 warm JAX 样本的 `jax_bcoo_cache_misses=0`、
  `jax_bcoo_cache_hits=16`，说明第二轮已完全复用 BCOO structure。
  `activation` 约 `0.001s/16 steps`，已经不是优先问题；`material +
  history + quad_state + postprocess` 合计约 `0.31s/16 steps`，可作为
  Phase 3 device-resident kernel 的候选，但当前最大项仍是
  `assembly`（约 `6.2s/16 steps`）和 JAX solver kernel（约
  `1.2s/16 steps`）。

  2026-07-07 继续拆分 `assembly` 后，默认 no-warm-start 的
  `repeat=2, discard-first=1` warm 样本来自
  `/tmp/v04_bench_small_loop_assembly_split_repeat2/combined.json`：

  | solver | wall(s) | linear_iterations | solver | conversion | local_assembly | global_matrix | assembly | python_overhead | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | spsolve | 7.71 | 16 | 0.042 | 0.000 | 5.841 | 0.006 | 5.848 | 0.799 | 482 |
  | jax-cg-no-check | 9.29 | 16 | 1.621 | 0.005 | 5.866 | 0.007 | 5.873 | 0.758 | 581 |

  这轮修正后 `linear_iterations` 不再恒为 0；16 个 thermal step
  对应 16 次 Newton 线性迭代。assembly 的主要成本几乎全部来自
  `local_assembly`（`Problem.newton_update` / cell residual-Jacobian
  计算），`global_matrix` 只有约 `0.007s/16 steps`，不是当前瓶颈。
  因此下一步应优先把局部单元 residual/Jacobian 与材料/quad state
  数据通路推向 fixed-shape JIT/device-resident，而不是继续优化 PETSc
  全局矩阵构造。

  2026-07-07 继续拆分 `local_assembly` 后，默认 no-warm-start 的
  `repeat=2, discard-first=1` warm 样本来自
  `/tmp/v04_bench_small_loop_local_detail_repeat2/combined.json`：

  | solver | wall(s) | linear_iterations | solver | local_assembly | cell_jacobian | face_jacobian | residual_scatter | global_matrix | python_overhead | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | spsolve | 8.31 | 16 | 0.043 | 6.459 | 6.014 | 0.174 | 0.177 | 0.004 | 0.740 | 519 |
  | jax-cg-no-check | 9.25 | 16 | 1.622 | 5.790 | 5.361 | 0.163 | 0.166 | 0.006 | 0.790 | 578 |

  这说明 `local_assembly` 内部的第一主项是 `cell_jacobian`
  （volume residual/Jacobian kernel），约占 `local_assembly` 的 92-93%。
  `face_jacobian` 和 `residual_scatter` 各约 `0.16-0.18s/16 steps`，
  暂不是优先瓶颈。下一步真正的 XLA 化应优先围绕 `Problem.split_and_compute_cell`
  中的 fixed-shape cell kernel、internal_vars device residency、chunk/vmap
  策略展开。

- 2026-07-07 追加了 `Problem.split_and_compute_cell` chunk 策略入口：
  `jax_fem.problem.Problem.cell_assembly_num_cuts` 在核心库里默认仍为历史值
  `20`，但新增 `cell_assembly_target_batch_size` 可按 cell 数自动计算 cuts。
  v04 wrapper 默认启用 `--xla-cell-target-batch-size 2048`，显式
  `--xla-cell-num-cuts N` 时关闭 auto 并回到固定 cuts；bench 同步支持
  `--cell-target-batch-size` 和 `--cell-num-cuts` 透传。这样 small-loop
  默认走 1 个 chunk，代表性大网格则按目标 batch size 分块，避免把所有
  cell 一次性塞入单个 batch。

  同机 `small-loop`、`repeat=2, discard-first=1` 对照：

  | setting | solver | wall(s) | solver | local_assembly | cell_jacobian | face_jacobian | residual_scatter | python_overhead | ms/step |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | default 20 cuts | spsolve | 8.24 | 0.041 | 6.433 | 5.987 | 0.176 | 0.173 | 0.708 | 515 |
  | default 20 cuts | jax-cg-no-check | 9.42 | 1.183 | 6.441 | 6.013 | 0.163 | 0.169 | 0.726 | 589 |
  | 1 cut | spsolve | 2.34 | 0.041 | 0.575 | 0.162 | 0.150 | 0.165 | 0.700 | 147 |
  | 1 cut | jax-cg-no-check | 3.56 | 1.198 | 0.617 | 0.177 | 0.164 | 0.172 | 0.711 | 223 |

  profile 文件：
  `/tmp/v04_bench_small_loop_cellcuts_default/combined.json` 与
  `/tmp/v04_bench_small_loop_cellcuts_1/combined.json`。最终 VTU 温度场
  比对显示 spsolve 的 default/1-cut 完全一致（`max_abs_T_diff=0`），
  说明 chunk 数改变不影响 direct baseline 的物理输出；`jax-cg-no-check`
  相对 spsolve 本身仍有数 K 量级误差，因此它仍只能作为热路径性能候选，
  不能作为数值验收基准。下一步应把该参数做成按 cell 数/显存预算选择的
  auto chunk 策略，再向 medium/representative 档验证。

  该 auto 策略已成为 v04 默认路径。默认 small-loop、不传任何 cell 参数的
  `repeat=2, discard-first=1` 样本来自
  `/tmp/v04_bench_small_loop_auto_chunk_default/combined.json`：

  | solver | wall(s) | chunking | target_batch | solver | local_assembly | cell_jacobian | python_overhead | ms/step |
  |---|---:|---|---:|---:|---:|---:|---:|---:|
  | spsolve | 2.33 | auto | 2048 | 0.039 | 0.571 | 0.162 | 0.704 | 146 |
  | jax-cg-no-check | 4.04 | auto | 2048 | 1.660 | 0.594 | 0.197 | 0.783 | 253 |

  `cell_assembly_chunking=auto_target_batch_size`、`cell_assembly_num_cuts=None`
  写入 profile meta。auto 默认的 spsolve 最终 VTU 温度场相对 legacy
  20-cuts spsolve 仍为 `max_abs_T_diff=0`。JAX 路线在该小规模档仍慢于
  spsolve，主要瓶颈转为 iterative solver 本身；因此 v04 默认求解器仍不应
  强制改为 JAX。

- 2026-07-07 追加了 Phase 3 的第一片 loop-side kernel JIT：
  v04 wrapper 默认启用 `--xla-jit-loop-kernels`，可用
  `--no-xla-jit-loop-kernels` 关闭；bench 同步支持 `--no-loop-jit`。
  该 patch 只在 property table 全为空的常见路径 JIT
  `thermal_material_quads`，只在线弹性路径 JIT
  `mechanics_material_quads`，并 JIT 相态/T_ref/eqp history update；
  有表格材料或 `j2_plastic` 力学材料时自动退回 v03 原函数。JIT kernel
  cache 提升到 v04 模块级别，同一进程内 benchmark repeat 不会反复为同一组
  material constants 创建新 kernel。

  同机 `small-loop`、spsolve、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | material(s) | history(s) | local_assembly(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | `--no-loop-jit` | 2.42 | 0.239 | 0.038 | 0.591 | 0.726 | 151 |
  | default loop JIT | 2.30 | 0.003 | 0.002 | 0.624 | 0.793 | 144 |

  profile 文件为
  `/tmp/v04_bench_small_loop_loopjit_disabled/combined.json` 与
  `/tmp/v04_bench_small_loop_loopjit_default/combined.json`。默认 JIT 样本记录
  `loop_kernel_jit_thermal_calls=16`、`loop_kernel_jit_history_calls=16`、
  `loop_kernel_jit_mechanics_calls=16`，thermal/history cache entries 均为 1
  （这是 thermal-only mechanics material skip 加入前的历史样本）。
  最终 VTU 温度场相对关闭 JIT 的 spsolve 输出为
  `max_abs_T_diff=0`、`mean_abs_T_diff=0`。这说明 material/history
  派发开销已基本消除；但 small-loop 总 wall 仍主要受 setup、local assembly
  和未细分 Python overhead 控制，因此下一步仍应优先处理 `cell_jacobian`
  与主循环剩余 host 逻辑。

- 2026-07-07 继续加了 thermal-only mechanics material skip：
  v03 主循环在 `--mechanics-every 0` 时仍会每步调用
  `mechanics_material_quads(...)`，但 `did_mechanics` 永远为 false，且没有
  release solve 时这些 mechanics material arrays 不会被使用。v04 现在默认
  启用 `--xla-skip-unused-mechanics-material`，可用
  `--no-xla-skip-unused-mechanics-material` 关闭；bench 同步支持
  `--no-skip-unused-mechanics-material`。为避免破坏 `release_after_cooling`
  后续复用 `mechanics_params` 的路径，该 skip 仅在
  `mechanics_every == 0 and not release_after_cooling` 时启用，并返回零分配
  占位引用，不创建无用 JAX arrays。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | mech_skip | material(s) | local_assembly(s) | solver(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | skip off | 1.792 | 0 | 0.0032 | 0.272 | 0.036 | 0.670 | 112.0 |
  | skip on | 1.690 | 16 | 0.0015 | 0.270 | 0.043 | 0.629 | 105.7 |

  profile 文件为
  `/tmp/v04_bench_small_loop_mech_skip_off_v2/combined.json` 与
  `/tmp/v04_bench_small_loop_mech_skip_on_v2/combined.json`。最终 VTU 对照中
  `T/sol/u` point data 均为 0；cell data 只有 `dT` 为
  `1.82e-12` 的舍入级差异，其余字段为 0。该片只优化 mechanics 完全关闭的
  thermal-only benchmark/run；有 release solve 或周期性 mechanics 的正式
  热-力耦合路径仍保留 v03/JIT mechanics material 计算。

- 2026-07-07 继续在 `Problem.split_and_compute_cell` 加了 single-cut fast
  path：当 auto chunking 得到 `num_cuts=1` 时，直接调用已有 cell
  `kernel`/`kernel_jac`，不再进入单元素 list + `vstack` 路径；多 chunk
  行为保持不变。新增单元测试在 residual-only 和 jacobian 两个分支中用
  会失败的 `vstack` stub 验证单 chunk 确实绕过拼接。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | local_assembly(s) | cell_jacobian(s) | face_jacobian(s) | residual_scatter(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | loop JIT before single-cut fast path | 2.30 | 0.624 | 0.158 | 0.175 | 0.188 | 144 |
  | single-cut fast path | 2.13 | 0.566 | 0.119 | 0.163 | 0.157 | 133 |

  profile 文件为
  `/tmp/v04_bench_small_loop_loopjit_default/combined.json` 与
  `/tmp/v04_bench_small_loop_singlecut_fastpath/combined.json`。最终 VTU
  温度场相对 fast path 前的 spsolve 输出仍为
  `max_abs_T_diff=0`、`mean_abs_T_diff=0`。这不是 full-loop XLA，
  但它把 auto chunking 的小/中规模路径从“单 chunk 仍拼接一次”改成了
  真正 fixed-shape 直通，为后续 fixed-shape cell kernel/device-resident
  改造减少一层 Python/JAX glue。

- 2026-07-07 继续加了空 surface fast path：当某个 boundary face set
  为空时，`Problem.compute_face` 直接返回正确 shape 的空 residual/Jacobian，
  不再调用 face JIT kernel；`compute_residual_vars_helper` 同时跳过空 face
  scatter。该优化针对 small-loop 当前真实状态
  `thermal_boundary_face_counts=[0, 0]`，避免每次 Newton 装配都对两个空
  surface map 触发无效 JAX 调用。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | local_assembly(s) | cell_jacobian(s) | face_jacobian(s) | residual_scatter(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | single-cut fast path | 2.13 | 0.566 | 0.119 | 0.163 | 0.157 | 133 |
  | empty surface fast path | 1.91 | 0.293 | 0.129 | 0.000 | 0.077 | 119 |

  profile 文件为
  `/tmp/v04_bench_small_loop_singlecut_fastpath/combined.json` 与
  `/tmp/v04_bench_small_loop_emptyface_fastpath/combined.json`。最终 VTU
  温度场相对 empty surface fast path 前的 spsolve 输出仍为
  `max_abs_T_diff=0`、`mean_abs_T_diff=0`。这说明当前 small-loop 的
  surface assembly 成本主要是空 boundary set 的框架开销；真实有外表面
  face 的代表性档仍需保留 face kernel 路径并单独验证。

- 2026-07-07 继续给 `compute_residual_vars_helper` 增加单变量 scatter
  fast path：当 `Problem.num_vars == 1` 时，cell/face residual 直接按
  `(num_cells, num_nodes, vec)` reshape 后 scatter，不再通过
  `jax.vmap(unflatten_fn_dof)` 解包 pytree。多变量耦合问题保持原通用路径。
  该片覆盖当前 thermal 主路径，也适用于单变量 mechanics/vector 问题。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | local_assembly(s) | cell_jacobian(s) | face_jacobian(s) | residual_scatter(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | empty surface fast path | 1.91 | 0.293 | 0.129 | 0.000 | 0.077 | 119 |
  | single-var scatter fast path | 1.79 | 0.286 | 0.125 | 0.000 | 0.054 | 112 |

  profile 文件为
  `/tmp/v04_bench_small_loop_emptyface_fastpath/combined.json` 与
  `/tmp/v04_bench_small_loop_singlevar_scatter_fastpath/combined.json`。最终
  VTU 温度场相对 single-var scatter fast path 前的 spsolve 输出仍为
  `max_abs_T_diff=0`、`mean_abs_T_diff=0`。这进一步把 small-loop 的
  local assembly 剩余成本压到 cell residual/Jacobian kernel 与主循环
  Python/setup 成本上。

- 2026-07-07 继续给 `Problem.compute_residual_vars` /
  `compute_newton_vars` 增加单变量 `cells_sol_flat` 快路径：
  当 `Problem.num_vars == 1` 时，`sol[cells]` 得到的
  `(num_cells, num_nodes, vec)` 直接 reshape 为
  `(num_cells, num_nodes * vec)`，不再通过
  `jax.vmap(lambda *x: ravel_pytree(x)[0])`。多变量耦合问题仍保留原
  pytree flatten 顺序。新增单元测试覆盖直接 helper、multi-var 顺序兼容、
  residual 入口和 newton 入口，并用 `ravel_pytree` fail stub 验证单变量
  路径没有回到旧实现。

  本轮验证：

  | check | result |
  |---|---|
  | `python -m unittest tests.test_jax_problem_cell_cuts` | 24 tests OK |
  | `python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 107 tests OK |
  | `python -m py_compile jax_fem/problem.py 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py tests/test_jax_problem_cell_cuts.py` | OK |
  | `RUN_ID=flatten_fastpath_dryrun 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh --xla-dry-run` | GPU / `precond=True` / fallback on |

  当前 post-change health sample 为 `small-loop`、spsolve、auto chunking、
  `repeat=2, discard-first=1`，来自
  `/tmp/v04_flatten_fastpath_bench/combined.json`：

  | wall(s) | setup(s) | local_assembly(s) | cell_jacobian(s) | residual_scatter(s) | dof_to_quad(s) | python_overhead(s) | ms/step |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1.634 | 0.700 | 0.233 | 0.106 | 0.056 | 0.029 | 0.598 | 102.1 |

  该片是保精度的单变量数据整理优化和 fixed-shape 路径铺垫；由于没有在同一
  代码点做 before/after 二分回退，本轮只把它记录为正确性受测的 micro-opt，
  不把它单独声明为端到端加速或 GPU speedup。

- 2026-07-07 继续给 `Problem.__post_init__` 增加单变量初始化快路径：
  单变量问题的 `cells_flat` 直接由 connectivity reshape 得到；
  cell/face 的全局 DOF 索引 `I/J` 直接由 `cells * vec + arange(vec)`
  生成，不再通过 `jax.vmap(find_ind)`。多变量耦合问题仍保留原通用
  `vmap + hstack` 路径。该片覆盖 v03/v04 的 thermal 标量问题和
  mechanics 三分量位移问题，目标是降低 `setup` 阶段固定初始化成本，
  不改变 FEM 方程、边界条件、材料参数或输出字段。

  本轮验证：

  | check | result |
  |---|---|
  | `python -m unittest tests.test_jax_problem_cell_cuts` | 24 tests OK |
  | `python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 107 tests OK |
  | `python -m py_compile jax_fem/problem.py 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py tests/test_jax_problem_cell_cuts.py` | OK |
  | `RUN_ID=init_fastpath_dryrun 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh --xla-dry-run` | GPU / `precond=True` / fallback on |
  | `git diff --check` | OK |

  当前 post-change health sample 为 `small-loop`、spsolve、auto chunking、
  `repeat=2, discard-first=1`，来自
  `/tmp/v04_init_fastpath_bench/combined.json`：

  | wall(s) | setup(s) | local_assembly(s) | cell_jacobian(s) | residual_scatter(s) | dof_to_quad(s) | python_overhead(s) | ms/step |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1.539 | 0.656 | 0.237 | 0.113 | 0.051 | 0.025 | 0.548 | 96.2 |

  最终 VTU 相对上一片
  `/tmp/v04_flatten_fastpath_bench/small-loop_spsolve_r01_run/step_000015_scan.vtu`
  的 point data `T/sol/u` 均为 `max_abs=0`。该片同样是保精度的
  setup micro-opt 和 fixed-shape 路径清理，不能单独作为 GPU speedup 证据。

- 2026-07-07 继续给 `Problem` 增加单变量 flat DOF unflatten 快路径：
  当 `Problem.num_vars == 1` 时，`unflatten_fn_dof` 和
  `unflatten_fn_sol_list` 直接 reshape 到 `(num_nodes, vec)` /
  `(num_total_nodes, vec)`，不再通过 dummy pytree 调用
  `jax.flatten_util.ravel_pytree` 生成 unflatten function。多变量耦合问题仍
  保留原 pytree unflatten 结构。该片针对 cell kernel 内部频繁调用的
  fixed-shape 单变量路径，目标是减少热标量和力学位移问题中的通用 pytree
  overhead，不改变 DOF 顺序、FEM 方程或输出字段。

  本轮验证：

  | check | result |
  |---|---|
  | `python -m unittest tests.test_jax_problem_cell_cuts` | 26 tests OK |
  | `python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 109 tests OK |
  | `python -m py_compile jax_fem/problem.py 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py tests/test_jax_problem_cell_cuts.py` | OK |
  | `RUN_ID=unflatten_fastpath_dryrun 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh --xla-dry-run` | GPU / `precond=True` / fallback on |

  当前 post-change health sample 为 `small-loop`、spsolve、auto chunking、
  `repeat=2, discard-first=1`，来自
  `/tmp/v04_unflatten_fastpath_bench/combined.json`：

  | wall(s) | setup(s) | local_assembly(s) | cell_jacobian(s) | residual_scatter(s) | dof_to_quad(s) | python_overhead(s) | ms/step |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1.573 | 0.657 | 0.244 | 0.118 | 0.053 | 0.026 | 0.572 | 98.3 |

  最终 VTU 相对上一片
  `/tmp/v04_init_fastpath_bench/small-loop_spsolve_r01_run/step_000015_scan.vtu`
  的 point data `T/sol/u` 均为 `max_abs=0`、`mean_abs=0`。该片继续作为
  保精度 fixed-shape micro-opt 记录；由于该基准仍是小循环健康样本，不把它
  单独声明为端到端 GPU 加速证据。

- 2026-07-07 继续拆分 solver 外层 Newton 调用开销：
  v04 wrapper 现在在 `base.solver(...)` patch 外层记录
  `nonlinear_solve`，并新增派生 stage `nonlinear_solve_overhead`，其定义为
  完整 `jax_fem.solver.solver(...)` 调用耗时减去
  `local_assembly/global_matrix/solver/conversion/transfer` 的增量。这样
  profile 可以区分“FEM Newton 外层 glue / BC / residual orchestration”
  和已经细分的局部装配、全局矩阵、线性求解器。`nonlinear_solve` 作为总量
  展示但不参与 `python_overhead` accounting，避免和内部子阶段重复计数；
  `nonlinear_solve_overhead` 参与 accounting，因此剩余 `python_overhead`
  更接近 v03 主循环自己的 path/output/bookkeeping 开销。

  本轮验证：

  | check | result |
  |---|---|
  | `python -m unittest tests.test_v04_xla_wrapper` | 70 tests OK |
  | `python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 110 tests OK |
  | `python -m py_compile 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py tests/test_v04_xla_wrapper.py` | OK |
  | `RUN_ID=nonlinear_profile_dryrun 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh --xla-dry-run` | GPU / `precond=True` / fallback on |

  当前 post-change health sample 为 `small-loop`、spsolve、auto chunking、
  `repeat=2, discard-first=1`，来自
  `/tmp/v04_nonlinear_profile_bench/combined.json`：

  | wall(s) | setup(s) | nonlinear_solve(s) | nonlinear_overhead(s) | local_assembly(s) | solver(s) | python_overhead(s) | ms/step |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1.505 | 0.611 | 0.700 | 0.445 | 0.240 | 0.012 | 0.107 | 94.1 |

  相对上一片的 profile 归因，原先约 `0.57s/16 steps` 的
  `python_overhead` 被拆出约 `0.445s/16 steps` 的
  `nonlinear_solve_overhead`，剩余 `python_overhead` 降到约
  `0.107s/16 steps`。最终 VTU 相对
  `/tmp/v04_unflatten_fastpath_bench/small-loop_spsolve_r01_run/step_000015_scan.vtu`
  的 point data `T/sol/u` 均为 `max_abs=0`、`mean_abs=0`。结论：下一步
  优先级应转向 jax-fem Newton 外层 overhead（BC/residual orchestration、
  apply_bc、Python/JAX dispatch）以及 setup 复用，而不是继续只看 v03
  path bookkeeping。

- 2026-07-07 继续给 `jax_fem.solver` 增加单变量 Dirichlet BC flat-index
  快路径：当 `Problem.num_vars == 1` 且只有一个 FE 时，
  `apply_bc_vec()`、`assign_bc()`、`assign_ones_bc()`、
  `assign_zeros_bc()` 和 `copy_bc()` 直接使用全局 flat DOF index，不再为
  标量热问题反复 `unflatten_fn_sol_list()` 再 `ravel_pytree()`。多变量耦合
  问题和非标准 BC 列表仍走原通用路径。该片针对上一段 profile 中的
  `nonlinear_solve_overhead`，目标是减少 Newton 外层 BC/residual glue
  开销，不改变 Dirichlet DOF 顺序或赋值语义。

  本轮先补了一个缓存失效保护：fast path 缓存带有
  `node_inds/vec_inds/vals` 的结构签名；如果 FE 的 Dirichlet 列表被
  `update_Dirichlet_boundary_conditions()` 一类逻辑替换，flat index/value
  会自动重建，不会复用旧 BC 值。新增回归测试先在旧实现上失败，修复后通过。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_flat_cache_rebuilds_when_bc_lists_change` | first failed on stale cached BC values, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_uses_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_assign_and_copy_bc_use_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_flat_cache_rebuilds_when_bc_lists_change` | 3 tests OK |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 113 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_bc_flat_cache_guard_cpu_check --json /tmp/v04_bc_flat_cache_guard_cpu_check/combined.json` | driver completed, 16 steps |
  | VTU compare against `/tmp/v04_nonlinear_profile_bench/small-loop_spsolve_r01_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile jax_fem/solver.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check` | OK |

  由于一个旧 first5 进程仍在运行并占用 CPU（旧脚本启动，带
  `--xla-jax-precond`，输出目录为
  `thermal_macro1mm_intersection_first5_h60_mech100_xla_v04_20260707_175954`），
  本轮 CPU small-loop 只作为流程/数值健康检查，不把 wall time 当作可信性能
  证据。待该进程停止后，应再运行同一 `small-loop` / `repeat=2` /
  `discard-first=1` 基准，用同机 warm 样本判断 BC fast path 是否带来
  `nonlinear_solve_overhead` 下降。

- 2026-07-07 在同一 Newton 外层路径继续加了 Dirichlet BC zero-seed 缓存：
  `newton_step()` 原先每次线性求解前都会重新执行
  `assign_bc(zeros(...), problem)` 来构造边界位置正确的 `x0_1`。对于单变量
  静态 Dirichlet BC，这个向量只依赖 BC index/value 和 DOF dtype/shape；
  v04 现在通过 `_assign_bc_zero_seed(dofs, problem)` 缓存该 seed，并在
  BC 列表变化时随 flat BC cache 一起失效重建。`copy_bc(dofs, problem)`
  仍逐步读取当前 DOF，因此 Newton 初值中随状态变化的边界部分没有被缓存。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_zero_seed_cache_rebuilds_when_bc_lists_change` | first failed before helper existed, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_uses_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_assign_and_copy_bc_use_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_flat_cache_rebuilds_when_bc_lists_change tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_zero_seed_cache_rebuilds_when_bc_lists_change` | 4 tests OK |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 114 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_bc_seed_cache_cpu_check --json /tmp/v04_bc_seed_cache_cpu_check/combined.json` | driver completed, 16 steps |
  | VTU compare against `/tmp/v04_bc_flat_cache_guard_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile jax_fem/solver.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check` | OK |

  该片仍只是固定开销 micro-opt。由于旧参数 first5 进程仍在占用 CPU，本轮
  不声明 wall-time speedup；待资源空闲后需要用 repeat/discard-first 的 warm
  small-loop 重新比较 `nonlinear_solve_overhead`。

- 2026-07-07 继续合并单变量 no-projection Newton 初值构造：
  在没有 `P_mat` 的常见热标量路径中，`newton_step()` 原来会先构造
  `x0_1 = BC value seed`，再构造 `x0_2 = copy_bc(dofs)`，最后执行
  `x0 = x0_1 - x0_2`。v04 现在通过
  `_single_var_bc_initial_guess(dofs, problem)` 直接在边界 flat indices 上写入
  `flat_vals - dofs[flat_inds]`，避免一次 `copy_bc()` 和一次整向量相减。
  带 `P_mat` 的投影路径保持原逻辑，多变量问题仍走通用路径。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_newton_step_builds_x0_without_copy_bc` | first failed while old code still called `copy_bc`, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_uses_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_assign_and_copy_bc_use_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_flat_cache_rebuilds_when_bc_lists_change tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_zero_seed_cache_rebuilds_when_bc_lists_change tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_newton_step_builds_x0_without_copy_bc` | 5 tests OK |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 115 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_bc_x0_fastpath_cpu_check --json /tmp/v04_bc_x0_fastpath_cpu_check/combined.json` | driver completed, 16 steps |
  | VTU compare against `/tmp/v04_bc_seed_cache_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile jax_fem/solver.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check` | OK |

  该片同样只声明为 correctness-preserving fixed-overhead cleanup。旧参数
  first5 进程仍在运行时，CPU small-loop wall time 不作为性能结论。

- 2026-07-07 继续把 Newton BC 初值构造拆成独立 profile stage：
  `jax_fem.solver.newton_step()` 现在通过 `_timing_record()` 写入
  `bc_initial_guess`，v04 wrapper 将其映射到新增 stage
  `bc_initial_guess`，并把它从 `nonlinear_solve_overhead` 的内部归因中扣除。
  这样后续可以区分 BC 初值构造、线性求解器、local/global assembly 和剩余
  Newton orchestration。该片只改变 profile 归因，不改变求解路径或输出。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_newton_step_builds_x0_without_copy_bc tests.test_v04_xla_wrapper.MacroMech100V04XlaWrapperTest.test_timing_patch_maps_jax_internal_breakdown_without_double_counting` | first failed before timing/stage existed, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 115 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_bc_stage_cpu_check --json /tmp/v04_bc_stage_cpu_check/combined.json` | driver completed, 16 steps; `bc_initial_guess=0.028s`, 16 calls |
  | VTU compare against `/tmp/v04_bc_x0_fastpath_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py jax_fem/solver.py tests/test_v04_xla_wrapper.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check` | OK |

  在当前 CPU 健康样本中，`bc_initial_guess` 只有约
  `1.75 ms/step`，不是后续最大项；但把它拆出后，剩余
  `nonlinear_solve_overhead` 仍约 `26.7 ms/step`。后续应继续定位这块剩余
  overhead，而不是继续在 BC 初值上做复杂优化。

- 2026-07-07 继续把 residual vector glue 拆成独立 profile stage：
  `newton_update_helper()` 中 `problem.newton_update()` 之后的 residual list
  flatten、Dirichlet residual BC 和可选 `P_mat.T` 投影现在计入
  `residual_vector`。该 stage 同样从 `nonlinear_solve_overhead` 内部归因中
  扣除，用来区分 local assembly kernel 和 residual 向量整理开销。该片只改变
  profile 归因，不改变 FEM residual/Jacobian 或输出。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper.MacroMech100V04XlaWrapperTest.test_timing_patch_maps_jax_internal_breakdown_without_double_counting` | first failed before stage existed, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 115 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_residual_vector_stage_cpu_check --json /tmp/v04_residual_vector_stage_cpu_check/combined.json` | driver completed, 16 steps; `residual_vector=0.160s`, 32 calls |
  | VTU compare against `/tmp/v04_bc_stage_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py jax_fem/solver.py tests/test_v04_xla_wrapper.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check` | OK |

  当前 CPU 健康样本显示 `residual_vector` 约 `10.0 ms/step`，比
  `bc_initial_guess` 更值得后续优化；拆出后剩余 `nonlinear_solve_overhead`
  降到约 `16.0 ms/step`。下一步优先考虑 residual vector 的单变量
  fast path / 避免通用 pytree flatten，而不是继续扩大线性求解器 wrapper。

- 2026-07-07 继续给 residual list flatten 增加单变量快路径：
  `_flatten_residual_list(res_list, problem)` 在 `num_vars == 1`、单 FE、
  单 residual array 时直接 `reshape(-1)`，避免通过通用 pytree flatten
  处理热标量 residual；多变量或多 residual 路径仍回退原
  `jax.flatten_util.ravel_pytree(res_list)[0]`。该片只替换
  `residual_vector` stage 内的 residual flatten，不改变 Dirichlet BC 处理、
  `P_mat` 投影、local assembly 或线性求解器。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_flatten_residual_list_uses_direct_reshape` | first failed before helper existed, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_flatten_residual_list_uses_direct_reshape tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_flatten_residual_list_falls_back_for_multivar_problem tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_newton_step_builds_x0_without_copy_bc tests.test_v04_xla_wrapper.MacroMech100V04XlaWrapperTest.test_timing_patch_maps_jax_internal_breakdown_without_double_counting` | targeted tests OK |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 117 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_residual_flatten_fastpath_cpu_check --json /tmp/v04_residual_flatten_fastpath_cpu_check/combined.json` | driver completed, 16 steps; `residual_vector=0.160s`, 32 calls; `nonlinear_solve_overhead=0.242s` |
  | VTU compare against `/tmp/v04_residual_vector_stage_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0` |
  | `python -m py_compile 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py jax_fem/solver.py tests/test_v04_xla_wrapper.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check && bash -n 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh` | OK |

  结论：该片把单变量 residual flatten 的通用 pytree 开销移除，并用多变量
  fallback 测试锁住边界；当前 CPU 健康样本中 `residual_vector` 仍约
  `10.0 ms/step`，说明剩余成本主要不在 pytree flatten，而更可能在
  `apply_bc_vec` 的 scatter/update、JAX dispatch 或计时窗口内的同步。由于旧
  first5 进程仍占用资源，本次不声明 wall-time speedup。

- 2026-07-07 继续把 residual vector 明细拆成诊断子阶段：
  `newton_update_helper()` 现在在 `residual_vector` 总窗口内分别记录
  `residual_flatten`、`residual_bc` 和可选 `residual_projection`。
  v04 wrapper 将三者映射到 profile stage，但 `ProfilingReport.finish()`
  会把这些明细从 accounting 中排除，避免与父级 `residual_vector` 双重计入
  `python_overhead` 扣减。该片只改变 profile 颗粒度，不改变 residual、
  Dirichlet BC、`P_mat` 投影或输出。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper.MacroMech100V04XlaWrapperTest.test_finish_does_not_double_count_residual_vector_detail_stages tests.test_v04_xla_wrapper.MacroMech100V04XlaWrapperTest.test_timing_patch_maps_jax_internal_breakdown_without_double_counting` | first failed before stage constants/mapping existed, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 118 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_residual_detail_stage_cpu_check --json /tmp/v04_residual_detail_stage_cpu_check/combined.json` | driver completed, 16 steps; `residual_vector=0.149s`, `residual_flatten=0.008s`, `residual_bc=0.141s`, `residual_projection=0.000s` |
  | VTU compare against `/tmp/v04_residual_flatten_fastpath_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py jax_fem/solver.py tests/test_v04_xla_wrapper.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check && bash -n 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh` | OK |

  结论：下一步 residual-vector 热点已经明确落在 `residual_bc`
  （约 `8.82 ms/step`，32 calls）而不是 flatten。优先方向应是把单变量
  Dirichlet residual BC 的 scatter/update 做成更少 dispatch 的热路径，或把
  BC 行处理推近 residual assembly / sparse row elimination 边界；继续优化
  residual flatten 收益很小。

- 2026-07-07 继续给单变量 residual BC 增加缓存 JIT fast path：
  `apply_bc_vec()` 在 `num_vars == 1`、单 FE、非空 Dirichlet BC 且
  `scale` 是 Python 数值时，通过 `_single_var_residual_bc_kernel(...)`
  缓存一个按 flat BC index/value 和 scale 绑定的 JIT kernel，将
  `dofs[flat_inds] - flat_vals * scale` 与 `res_vec.at[flat_inds].set(...)`
  放进同一个 compiled helper。多变量、空 BC 或非数值 scale 仍回退原通用
  路径；BC 列表或 scale 改变时 cache key 变化并重建 kernel。

  本轮验证：

  | check | result |
  |---|---|
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_caches_jit_kernel_by_bc_and_scale` | first failed before residual-BC JIT helper existed, then passed |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_caches_jit_kernel_by_bc_and_scale tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_jit_cache_rebuilds_when_bc_lists_change tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_apply_bc_vec_uses_flat_dof_indices tests.test_jax_solver_preconditioner.JaxSolvePreconditionerTest.test_single_var_bc_flat_cache_rebuilds_when_bc_lists_change` | 4 tests OK |
  | `JAX_PLATFORMS=cpu python -m unittest tests.test_v04_xla_wrapper tests.test_jax_problem_cell_cuts tests.test_jax_solver_preconditioner` | 120 tests OK |
  | `JAX_PLATFORMS=cpu python 159_local/v04/bench_mech100_xla.py --tier small-loop --solvers spsolve --out /tmp/v04_residual_bc_jit_fastpath_cpu_check --json /tmp/v04_residual_bc_jit_fastpath_cpu_check/combined.json` | driver completed, 16 steps; `residual_bc=0.079s`, `residual_vector=0.087s` |
  | VTU compare against `/tmp/v04_residual_detail_stage_cpu_check/small-loop_spsolve_run/step_000015_scan.vtu` | point data `T/sol/u` all `max_abs=0`, `mean_abs=0` |
  | `python -m py_compile 159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py 159_local/v04/bench_mech100_xla.py jax_fem/solver.py tests/test_v04_xla_wrapper.py tests/test_jax_solver_preconditioner.py` | OK |
  | `git diff --check && bash -n 159_local/v04/run_macro_intersection_h60_mech100_XLA_first5.sh` | OK |

  与上一版 `/tmp/v04_residual_detail_stage_cpu_check/combined.json` 相比，
  `residual_bc` 从 `0.141s` 降到 `0.079s`，`residual_vector` 从
  `0.149s` 降到 `0.087s`；但同一冷启动 CPU 样本中
  `bc_initial_guess` 和其它 host overhead 有波动，端到端 wall
  `3.53s -> 3.56s`，不能声明整体加速。该片的价值是把 residual-BC 热点
  收窄到一个可复用的 compiled helper，并保持物理输出不变。

- 2026-07-07 继续给 v04 wrapper 增加 `make_quad_scalar` 的单积分点快路径：
  当 `num_quads == 1` 时，直接把 cell 标量 reshape 成
  `(num_cells, 1, 1)`；多积分点仍回退 v03 原函数，整数/非浮点输入保留原
  broadcast 语义。浮点/复数快路径返回 copy 而不是 view，以保留 v03 原函数
  “返回独立数组”的别名语义。该 patch 安装在 loop profiling patch 之前，
  因此 `quad_state` stage 记录的是优化后的真实调用成本。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | setup(s) | quad_state(s) | local_assembly(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | single-var scatter fast path | 1.79 | 0.704 | 0.033 | 0.286 | 0.684 | 112 |
  | quad scalar fast path | 1.83 | 0.708 | 0.024 | 0.277 | 0.741 | 115 |

  profile 文件为
  `/tmp/v04_bench_small_loop_singlevar_scatter_fastpath/combined.json` 与
  `/tmp/v04_bench_small_loop_quadscalar_copy_fastpath/combined.json`。本次
  快路径记录 `quad_scalar_fast_path_calls=49`，`quad_state` 约下降 27%，
  但总 wall 在该 repeat 下约上升 2.3%，不能声明端到端加速。最终 VTU
  `T/sol/u` point data 相对上一版均为 `max_abs=0`、`mean_abs=0`。结论：
  这是低风险、可保留的 micro-opt 和 profiling 清理，但已经不是主瓶颈；
  后续优先级仍应放在 setup 拆分、主循环 host overhead 定位，以及更大
  代表性网格上的 device-resident/fixed-shape 路径。

- 2026-07-07 继续补 `FiniteElement.convert_from_dof_to_quad` 的计时归因：
  v04 wrapper 在 `jax_fem.fe.FiniteElement` 类上安装幂等 timing patch，
  将主循环中 `T_old -> T_old_quad` 和 `T_new -> T_quad` 的 DOF-to-quadrature
  投影单独计入 `dof_to_quad` stage。该 patch 只改变 profile 归因，
  不改变 FEM 计算和输出。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 对照：

  | setting | wall(s) | dof_to_quad(s) | python_overhead(s) | local_assembly(s) | ms/step |
  |---|---:|---:|---:|---:|---:|
  | before dof_to_quad stage | 1.83 | 0.000 | 0.741 | 0.277 | 115 |
  | dof_to_quad stage | 1.90 | 0.081 | 0.704 | 0.275 | 119 |

  profile 文件为
  `/tmp/v04_bench_small_loop_quadscalar_copy_fastpath/combined.json` 与
  `/tmp/v04_bench_small_loop_dof_to_quad_stage/combined.json`。本次记录
  `dof_to_quad` 为 48 次调用、约 0.081 s，说明此前仍有一块数组投影成本
  混在 `python_overhead` 中。总 wall 在该 repeat 下约上升 3.6%，属于计时
  wrapper 和运行噪声带来的诊断成本，不能声明加速。最终 VTU `T/sol/u`
  point data 相对上一版均为 `max_abs=0`、`mean_abs=0`。后续若推进
  device-resident 状态，应把这两次每步 DOF-to-quad 投影作为明确候选，而不是
  继续把它归为泛泛的 Python overhead。

- 2026-07-07 追加了 `FiniteElement.convert_from_dof_to_quad` 身份缓存：
  v04 wrapper 默认启用 `--xla-dof-to-quad-cache`，可用
  `--no-xla-dof-to-quad-cache` 关闭。缓存只覆盖同一个
  `FiniteElement` 对象、同一个真实 `jaxlib` array 解向量对象的重复
  DOF-to-quadrature 投影；不会跨热/力学两个 FE 对象复用，也不会缓存普通
  NumPy/可变数组或 JAX tracer 路径。该范围正好消除 v03 每步中
  `T_old_quad = thermal.fes[0].convert_from_dof_to_quad(T_old)` 与
  `TransientThermal.set_params()` 内部同一 `T_old` 的重复投影。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking 的 cache on/off
  单次对照：

  | setting | wall(s) | dof_to_quad calls | dof_hit | dof_miss | dof_to_quad(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | cache on | 6.44 | 32 | 16 | 32 | 0.261 | 402 |
  | cache off | 6.46 | 48 | 0 | 0 | 0.292 | 403 |

  profile 文件为 `/tmp/v04_bench_small_loop_dof_cache_on/combined.json` 与
  `/tmp/v04_bench_small_loop_dof_cache_off/combined.json`。最终 VTU 对照中
  `sol/T/u` point data 均为 `max_abs=0`；cell data 只有 `dT` 出现
  `3.64e-12` 的舍入级差异，其余字段为 0。该优化把每步 3 次投影降到
  2 次，但在 small-loop 上端到端 wall 几乎不变；它是低风险去重与 profile
  清理，不是 GPU speedup 证据。

- 2026-07-07 追加了 Phase 2 的 step predicate cache：
  v04 wrapper 默认启用 `--xla-step-predicate-cache`，可用
  `--no-xla-step-predicate-cache` 关闭。该 patch 在
  `generate_raster_step_states()` / `generate_path_file_step_states()` 返回后，
  为每个 `global_step` 预计算
  `should_activate_layer_for_state`、`should_run_mechanics` 和
  `should_save_step` 所需的标量 predicate；主循环仍保持 v03 逐 step 结构，
  只是把重复的字符串/模运算判断转为 cache lookup。不同 `args` 对象或缺失
  cache 时自动回退 v03 原函数。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 的 on/off 对照：

  | setting | wall(s) | step_hit | step_miss | setup(s) | python_overhead(s) | activation(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | cache on | 1.765 | 48 | 0 | 0.668 | 0.661 | 0.0006 | 110.3 |
  | cache off | 1.784 | 0 | 0 | 0.683 | 0.668 | 0.0006 | 111.5 |

	  profile 文件为 `/tmp/v04_bench_small_loop_step_pred_on/combined.json` 与
	  `/tmp/v04_bench_small_loop_step_pred_off/combined.json`。最终 VTU 对照中
	  `sol/T/u` point data 均为 `max_abs=0`；cell data 只有 `dT` 出现
	  `1.82e-12` 的舍入级差异，其余字段为 0。该片是低风险 host-loop predicate
	  预计算，收益在 small-loop 上接近噪声线；真正下一步仍应继续处理 setup /
	  python_overhead 的来源拆分和更大网格上的 device-resident step/state 路径。

- 2026-07-07 追加了 setup detail timing：
  `STAGE_SETUP` 仍以第一次 thermal/mechanical solve 调用为边界，不改变
  原有 stage accounting；v04 只在 profile `meta` 中额外记录
  `setup_detail_mesh_read_seconds`、`setup_detail_path_generation_seconds`、
  `setup_detail_mesh_construction_seconds`、`setup_detail_thermal_problem_seconds`、
  `setup_detail_mechanics_problem_seconds`、`setup_detail_total_seconds` 和
  `setup_unattributed_seconds`。该 patch 放在 warm-start patch 之后、step
  predicate/path wrapper 之前，并使用专用 `_v04_setup_detail_original_*`
  保存原函数，避免被其它 `_v04_original_*` patch 绕过。

  同机 `small-loop`、spsolve、`repeat=2, discard-first=1` warm 样本来自
  `/tmp/v04_bench_small_loop_setup_detail_v2/combined.json`：

  | wall(s) | setup(s) | detail_total(s) | mesh_read(s) | path_generation(s) | thermal_problem(s) | mechanics_problem(s) | mesh_construction(s) | unattributed(s) |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 1.830 | 0.708 | 0.699 | 0.604 | 0.00015 | 0.0555 | 0.0380 | 0.00078 | 0.0090 |

  这说明当前 small-loop 档的 setup 主要是 `.inp` 读取/筛选
  (`read_tet4_inp`) 而不是 path 生成或 Problem 构造。该信息主要用于解释
  smoke/小档 benchmark 的固定成本，不能作为 GPU speedup 证据；真正 XLA
  升级热路径仍是 `cell_jacobian`/local assembly、JAX solver kernel 和
  device-resident step/state 数据通路。

- 2026-07-07 追加了 JAX sparse direct solver 候选：
  `--xla-linear-solver jax --xla-jax-method spsolve` 直接使用
  `jax.experimental.sparse.linalg.spsolve(data, indices, indptr, b)` 的 CSR
  路径，不再为直接求解先构造 BCOO；profile meta 记录
  `jax_spsolve_calls`。bench 同步提供 `jax-spsolve`，默认带
  `--xla-jax-skip-residual-check`，用于与 CPU `spsolve` 做热路径对照。该路径
  目前是 opt-in 候选，默认求解器仍保持 CPU `spsolve`。

  同机 `tiny`、`repeat=2, discard-first=1` 样本来自
  `/tmp/v04_bench_tiny_jax_spsolve/combined.json`：

  | solver | wall(s) | solver(s) | conversion(s) | calls |
  |---|---:|---:|---:|---:|
  | spsolve | 0.914 | 0.0032 | 0.0000 | - |
  | jax-cg-no-check | 0.982 | 0.0772 | 0.0007 | - |
  | jax-spsolve | 0.887 | 0.0008 | 0.0005 | 1 |

  `tiny` 上 `jax-spsolve` wall 略低，但样本只有 1 step，受 setup/cache/运行顺序
  影响大，不能作为泛化 GPU speedup 证据。

  同机 `small-loop`、`repeat=2, discard-first=1` 样本来自
  `/tmp/v04_bench_small_loop_jax_spsolve/combined.json`：

  | solver | wall(s) | solver(s) | conversion(s) | local_assembly(s) | dof_to_quad(s) | ms/step | calls |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | spsolve | 1.885 | 0.0386 | 0.0000 | 0.2809 | 0.0549 | 117.8 | - |
  | jax-spsolve | 1.893 | 0.0222 | 0.0095 | 0.2844 | 0.0574 | 118.3 | 16 |

  该档中 JAX sparse direct solver 阶段更短，但加上 CSR device conversion 后
  总 wall 与 CPU `spsolve` 基本打平且略慢。最终 VTU 对比显示 point data
  `T/sol/u` 和网格点完全一致，cell data 只有 `dT` 为 `1.63e-08` 量级差异，
  其余字段为 0。结论：`jax-spsolve` 是比 `jax-cg-no-check` 更可信的
  Phase 4 线性求解候选，但还需要 medium/representative 档验证，不能替代
  默认 `spsolve`。

- 2026-07-07 追加了热场 Newton warm-start 的 opt-in 实验路径：
  `--xla-thermal-warm-start` 会在 v04 wrapper 中 patch
  `TransientThermal.set_params()`，把参数 0 的上一时刻 `T_old` 暂存为
  DOF 初值；solver patch 只在调用方没有显式 `initial_guess` 时注入。
  默认保持关闭，确保 v04 默认路径仍复用 v03 的求解行为。tiny 对照显示
  默认 benchmark 的 profile 为 `thermal_warm_start_enabled=False`
  （`/tmp/v04_default_tiny_no_warm/combined.json`），显式
  `--thermal-warm-start` 时记录 `thermal_warm_start_injections=1`
  （`/tmp/v04_optin_tiny_warm/combined.json`）。small-loop opt-in
  样本记录 spsolve / JAX 均为 16 次注入
  （`/tmp/v04_bench_small_loop_warmstart_repeat2/combined.json`），但与
  no-warm 运行的最终温度存在约 `1e-5 K` 量级差异，因此该路径目前只能作为
  容差敏感的实验优化，不能作为默认兼容路径或 GPU 加速证据。

- 2026-07-07 追加了 `jax_fem` 求解器日志安静模式：
  v04 wrapper 默认启用 `--xla-quiet-jax-fem-logs`，在求解期间把
  `jax_fem` 包级 logger 提升到 `WARNING`，避免每个 Newton solve 的
  INFO/DEBUG 日志污染 host-loop profiling 和 benchmark 输出；需要调试
  solver 细节时可用 `--no-xla-quiet-jax-fem-logs` 恢复原 verbosity。
  bench 同步支持 `--no-quiet-jax-fem-logs`。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 的 on/off 对照：

  | setting | wall(s) | setup(s) | solver(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|
  | quiet logs on | 1.672 | 0.675 | 0.013 | 0.601 | 104.5 |
  | quiet logs off | 1.690 | 0.651 | 0.043 | 0.633 | 105.6 |

  profile 文件为
  `/tmp/v04_bench_small_loop_quiet_logs_on/combined.json` 与
  `/tmp/v04_bench_small_loop_quiet_logs_off/combined.json`。最终 VTU 对照中
  point data `T/sol/u` 均为 `max_abs=0`；cell data 只有 `dT` 为
  `3.64e-12` 的舍入级差异，其余字段为 0。该片主要是 benchmark/host
  overhead 边界清理，收益接近噪声线，不能作为 GPU speedup 证据；但它让
  后续 profiling 更少受日志 I/O 干扰。

- 2026-07-07 追加了 lazy output postprocess 的 opt-in 实验路径：
  `--xla-lazy-output-postprocess` 会利用 step predicate cache 判断当前 step
  是否会写 VTU；非保存步跳过 output-only 的
  `phase_cell_from_quad(...) -> material_cell_state(...)`，并复用上一份
  `last_material_state`。没有 step predicate cache 或缺少当前 step 上下文时
  自动回退原计算。该 patch 不改变 `compute_cell_temperature`、
  activation temperature、max temperature、solidification history、summary
  和最终保存步输出。由于 small-loop 当前 output-only material-state reduction
  本身很小，而 lazy proxy/context 判断也有成本，该路径保持 opt-in，默认关闭。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 的 on/off 对照：

  | setting | wall(s) | post_skip | post_compute | postprocess(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|
  | lazy output postprocess on | 1.626 | 15 | 2 | 0.017 | 0.581 | 101.6 |
  | lazy output postprocess off | 1.583 | 0 | 0 | 0.011 | 0.545 | 99.0 |

  profile 文件为
  `/tmp/v04_bench_small_loop_lazy_post_on/combined.json` 与
  `/tmp/v04_bench_small_loop_lazy_post_off/combined.json`。最终 VTU 对照中
  point data `T/sol/u` 均为 `max_abs=0`；cell data 只有 `dT` 为
  `3.64e-12` 的舍入级差异，其余字段为 0。结论：该片证明了
  output-only postprocess 可以被安全延后到保存步，但当前实现不是默认加速；
  更适合作为后续 full device-resident 输出边界设计的验证件。

- 2026-07-07 追加了 thermal-only mechanics Problem surrogate 默认路径：
  当 `mechanics_every == 0` 且 `release_after_cooling == False` 时，v03
  主循环不会进入 mechanics solve，但原逻辑仍会构造完整
  `ThermoMechanical` Problem，并且后续只用 `mechanics.fes[0]` 做
  `T_new -> T_quad` 投影。v04 现在默认启用
  `--xla-thermal-only-mechanics-surrogate`：在 thermal Problem 创建后缓存
  thermal FE，随后用轻量 surrogate 替代完整 mechanics Problem；该 surrogate
  只暴露 `fes[0].convert_from_dof_to_quad`、`node_inds_list` 和
  `num_total_dofs_all_vars`，真实 mechanics 或 release 路径自动回退原
  `ThermoMechanical`。可用 `--no-xla-thermal-only-mechanics-surrogate` 关闭，
  bench 同步支持 `--no-thermal-only-mechanics-surrogate`。

  同机 `small-loop`、spsolve、默认 loop JIT、auto chunking、
  `repeat=3, discard-first=1` 的 on/off 对照（第二版避免了 surrogate
  为启动日志遍历 `thermal_fe.points` 的额外 host/device 成本）：

  | setting | wall(s) | setup(s) | mech_surr | dof_hit | dof_miss | dof_to_quad(s) | python_overhead(s) | ms/step |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | surrogate on | 1.544 | 0.612 | 1 | 31 | 17 | 0.027 | 0.578 | 96.5 |
  | surrogate off | 1.550 | 0.634 | 0 | 16 | 32 | 0.042 | 0.558 | 96.9 |

  profile 文件为
  `/tmp/v04_bench_small_loop_mech_surrogate_on_v2/combined.json` 与
  `/tmp/v04_bench_small_loop_mech_surrogate_off_v2/combined.json`。最终 VTU
  对照中 point data `T/sol/u` 均为 `max_abs=0`；cell data 只有 `dT`
  为 `7.28e-12` 的舍入级差异，其余字段为 0。结论：该片不是 GPU speedup，
  但它把 thermal-only 路径中一份未使用的 mechanics Problem 初始化和一次
  重复 FE 投影边界收窄了；对 small-loop 端到端 wall 是噪声级小收益，默认
  开启的前提是仅限 no-mechanics/no-release，且有显式关闭开关。

- 2026-07-08 追加了 `real-slice` benchmark 档并修正了各档的代表性结论：
  真实 h60 网格是 197,266 TET4 单元 / 52,739 节点，而 medium 档只有 500
  单元（差约 400 倍）。`real-slice` 使用完整网格（`--max-cells 0`）、2 层 x
  每层 4 scan step、mechanics 关闭，用可控时长复现真实装配/求解成本配比。
  该档揭示：小档位上占 50%+ 的 host 固定开销在真实网格上只占约 3%，
  而 `cell_jacobian` 占 71%、CPU spsolve 占 21%。因此 tiny/small/medium
  上得到的 GPU-vs-CPU 结论不能外推到真实案例，必须以 real-slice 及以上
  规模为准。

- 2026-07-08 追加了 thermal Newton residual-only 收敛检查
  （`--xla-residual-only-check`，默认开，`--no-` 可关）：
  `jax_fem.solver.solver()` 新增 newton 选项 `residual_only_check`，收敛
  检查只调用 `problem.compute_residual`（residual kernel），不再为被丢弃的
  收敛检查装配元素雅可比并做 device→host `V` 拷贝；探测不收敛时自动回退
  完整 `newton_update` 重建 tangent 再继续迭代。v04 wrapper 只对
  `TransientThermal` 注入该选项（类名匹配，兼容其它 patch 包装类对象的
  情况）；力学求解保持 v03 残差+雅可比联合检查语义，因为 medium 档实测
  力学解在容差边界会差一次 Newton 迭代（应力差 ~3e-5 Pa 量级，物理无意义
  但不满足字节级验收）。同轮还把 `solver()` 每次调用无条件执行的
  `print()` / `np.max(dofs)` / `np.min(dofs)` 放到
  `logger.isEnabledFor(INFO)` 后面，quiet 模式下消除每 solve 两次 device
  归约和多次 stdout 写。

  同机 `medium`、spsolve、`repeat=2, discard-first=1` 对照（logging 防护
  两侧生效）：ROC off `9.68s`，ROC on `9.84s`（500 cell 档探测开销与收益
  抵消，打平）；相对本轮之前的 medium 基线 `11.22s`，logging 防护带来约
  `-14%`。`real-slice`（197k cells）上 ROC off `18.12s` → ROC on
  `16.68s`（`-8%`），`cell_jacobian` 从 `12.92s` 减半到 `6.39s`。
  VTU 对照：热场 `T/sol/u` 均 `max_abs=0`，`dT` 舍入级 `~3e-12`。
  新增单测覆盖 solver 层装配调用次数模式（converged/not-converged/默认
  三种）、flat option 透传、wrapper 注入与 thermal 类包装兼容，合计
  127 tests OK。profile 文件：
  `/tmp/v04_bench_medium_roc_{on,off}/combined.json`、
  `/tmp/v04_bench_realslice_roc_{on3,off}/combined.json`。

- 2026-07-08 把 `DEFAULT_CELL_TARGET_BATCH_SIZE` 从 `2048` 提升到
  `262144`：2048 是在小网格上调出的值，真实 197k 网格被切成 96 个
  chunk，每个 chunk 的 residual/jacobian 都经 `onp.vstack` 隐式做一次
  device→host 拷贝——这（而不是 FLOPs）主导了 `cell_jacobian`。
  同机 `real-slice`、spsolve、ROC on、`repeat=2, discard-first=1` 扫描：

  | target batch | chunks | wall(s) | cell_jacobian(s) | cell_residual(s) | ms/step |
  |---:|---:|---:|---:|---:|---:|
  | 2048（旧默认） | 96 | 16.68 | 6.39 | 5.06 | 2085 |
  | 32768 | 7 | 6.33 | 0.74 | 0.45 | 791 |
  | 262144（新默认，单 chunk） | 1 | 5.34 | 0.09 | 0.07 | 667 |

  相对本轮起点（2048 + 旧收敛检查，`18.12s`），real-slice 端到端约
  `3.4x`。VTU 对照：跨 batch、跨 ROC on/off 的 `T/sol/u` 均
  `max_abs=0`，仅 `dT` 有 `~3e-12` 舍入差。262144 单 chunk 的力学雅可比
  约 300MB device 侧，16GB 显卡余量充足；更小显存需显式降低
  `--xla-cell-target-batch-size`。profile 文件：
  `/tmp/v04_bench_realslice_bs_{32768,262144}/combined.json`。

- 2026-07-08 在真实网格规模上首次验证了 GPU 直接法：`jax-spsolve` 在
  52,739 DOF 热系统上 solver 阶段 `16.34s` vs CPU spsolve `3.85s`
  （约 4x 慢）。原因判断为 GeForce 卡的 FP64 吞吐限制（约 FP32 的
  1/64）叠加 experimental cusolver 路径；本机（RTX 5080）上双精度直接法
  留在 32 线程 CPU 是正确分工，GPU 的价值在大批量装配 kernel（见上一条）。
  装配提速后 CPU spsolve 已占 real-slice 总 wall 的约 68%，Phase 4 的
  下一步应评估固定 sparsity 的符号分解复用（CHOLMOD/SuperLU analyze 一次）
  或 PETSc 迭代法 + warm start，而不是继续尝试 GPU 直接法。

## Phase 1 — Profiling harness（已接线）

- `ProfilingReport`：按 stage 累计 wall time，stage 集合固定为
  `setup / activation / quad_state / material / history / postprocess /
  dof_to_quad / nonlinear_solve / nonlinear_solve_overhead /
  bc_initial_guess / residual_vector / residual_flatten / residual_bc /
  residual_projection / solver / conversion / transfer / local_assembly /
  global_matrix / assembly / cell_jacobian / cell_residual / face_jacobian /
  face_residual / residual_scatter / io / python_overhead`。
  `setup` 在首个 solver 调用前关闭，避免把 FE/problem 初始化和启动写文件
  误算成逐步 Python 循环成本。未显式归因且不属于 setup 的时间自动落入
  `python_overhead` —— 这才是 826k 扫描步下的每步固定开销（path step state、
  layer activation 判断、output 判断、dict 拷贝）。
- v03 主循环中的 activation mask、quad broadcast、DOF-to-quadrature
  projection、material property、phase/reference history、cell/output
  postprocess 已有独立 stage。DOF-to-quadrature stage 同时支持同一
  FE/同一 JAX 解数组身份缓存，profile meta 记录
  `dof_to_quad_cache_hits` / `dof_to_quad_cache_misses` /
  `dof_to_quad_cache_entries`。
- `jax_fem.solver.jax_solve()` 内部记录 `sparse_conversion`、
  `linear_kernel`、`linear_residual_check`；v04 wrapper 将它们归并到
  profile 的 `conversion` / `solver` stage，并避免外层 `linear`
  总时间双重计数。`local_assembly` / `global_matrix` 同时保留子阶段和
  `assembly` 汇总；`cell_jacobian` / `face_jacobian` / `residual_scatter`
  由 `jax_fem.problem.Problem` 方法 wrapper 记录。`residual_vector`
  同时拆出 `residual_flatten` / `residual_bc` / `residual_projection`
  明细。`assembly`、局部装配子项和 residual-vector 明细均作为诊断派生量，
  不参与 `python_overhead` 二次扣减。Newton 迭代数由
  `jax_fem.solver._log_timing_table` 记录到
  `linear_iterations` 和 profile meta。JAX BCOO structure cache 的 hit/miss 以
  `jax_bcoo_cache_hits` / `jax_bcoo_cache_misses` 写入 profile meta。
- `159_local/v04/bench_mech100_xla.py`：tiny / small-loop / medium /
  representative 四档，对 spsolve / jax / jax-precond / jax-cg /
  jax-cg-no-check / jax-gmres / jax-spsolve / petsc / petsc-gpu / amgx
  逐一实跑并出表。
  支持 `--repeat` / `--discard-first`，combined JSON 中保留 raw runs，
  防止把 JAX 编译缓存或运行顺序误判成 solver 加速。`--thermal-warm-start`
  会显式传递 wrapper 的 `--xla-thermal-warm-start`，用于单独评估热场
  Newton 初值实验；`--cell-num-cuts` / `--cell-target-batch-size` 会显式
  传递 wrapper 的 `--xla-cell-num-cuts` / `--xla-cell-target-batch-size`，
  `--no-dof-to-quad-cache` 可显式关闭 wrapper 的
  `--xla-dof-to-quad-cache` 以做缓存 on/off 对照。
  `--no-step-predicate-cache` 可关闭 wrapper 的
  `--xla-step-predicate-cache`，用于评估逐步 predicate 预计算。
  用于评估 `split_and_compute_cell` chunk 策略。`--no-loop-jit` 会显式传递
  wrapper 的 `--no-xla-jit-loop-kernels`，用于对比 Phase 3 loop-side
  material/history JIT 的收益。默认 benchmark 启用 v04 wrapper 的 auto
  cell chunking 和 loop-side JIT，但不启用 thermal warm-start。
- GPU utilization：representative 档跑期间用
  `nvidia-smi dmon -s um -d 1 -o T` 旁路采样，采样文件与 profile JSON 一起归档。
- 基线要求：在任何优化 PR 之前，先在目标机器上固化各档 × spsolve 的
  baseline JSON，之后所有 PR 用同机同档对比。

接线点：当前不改 v03 主体接口。v04 wrapper 负责加载 v03 base solver、
追加 XLA/profiling CLI、安装 solver/profiling patches，再调用 v03 `main()`。
这样可以在不破坏 v03 产物兼容性的前提下启动升级。

下一步必须优先降低 `cell_jacobian` 固定成本、JAX 编译/迭代开销和
linear solve hot path；material/history/quad_state 可作为 Phase 3 kernel 化候选，
但 activation cache 已经把 layer mask 重算降到非主瓶颈级别。完成这些前
不讨论 GPU speedup。

## Phase 2 — 主循环固定开销

优先级排序（按每步成本 × 步数）：

1. **path step state**：把逐步读取/解析改为一次性预处理成
   结构化数组（time, x, y, z, power, layer_id, is_output_step），
   主循环只做整型索引。output 判断预先算成布尔数组，循环内零分支字符串比较。
2. **layer activation**：预计算 layer→element 的激活映射
   （每层一个 element index 数组），激活变成一次数组切片赋值，
   而不是逐元素几何判断。当前 v04 已先用 wrapper-level cache
   消除同一层内重复的全 cell 掩码计算；下一步再把 cache 的值从
   bool mask 推进为 layer→element index / device-resident bool kernel。
3. **solver options / dict churn**：`rewrite_solver_options` 只在启动时
   调用一次；主循环内禁止 deep copy 任何 options dict。
4. **sparse structure cache**：fixed mesh ⇒ Jacobian 的 sparsity pattern
   不变。JAX 路线已缓存 BCOO indices / diagonal positions，先按当前
   PETSc Mat 对象命中；对象变了但 CSR pattern 相同时，再走最近 pattern
   cache，后续只更新 values。spsolve 路线仍可切 `splu` 复用符号分解
   （若矩阵值也不变的线弹性子段，直接复用数值分解）。

## Phase 3 — 纯数组 kernel JIT 化

当前已完成第一片：无 property table 的 thermal material update、线弹性
mechanics material update、phase/T_ref/eqp history update 已在 v04 wrapper
层 JIT 化，并保留 v03 fallback 与关闭开关。该片只消除 loop-side
material/history 的 Python/JAX op 派发成本，不代表 full-loop XLA。
核心 `Problem.split_and_compute_cell` 也已支持 `num_cuts=1` 的直接 cell
kernel 调用，避免 auto chunking 小/中规模路径仍走单元素 `vstack`。
空 boundary face set 已在 `compute_face` / residual scatter 中直接跳过，
避免对零 face 的 surface kernel 做无效 JIT 调用。

迁移顺序（fixed shape、无控制流依赖 host 的先走）：

1. activation mask 更新（printed/active/cooling_only 布尔数组）
2. phase update（相态状态机，可写成 `jnp.where` 链；无表格路径已完成）
3. T_ref update（无表格路径已完成）
4. material quadrature update（无表格 thermal/linear_elastic 路径已完成）
5. stress / von Mises postprocess（quadrature 级张量运算）

规则：

- 每个 kernel 独立 `jax.jit`，固定 shape（active 用 mask 而不是变长数组，
  避免 recompile）。
- kernel 之间的 state 保持 device-resident；**非输出步禁止任何
  device→host copy**。用 `jax.block_until_ready` 只在计时和输出步调用。
- `linear_elastic` 模式：eqp 等塑性历史变量完全不更新、不分配。

## Phase 4 — 稀疏线性求解数据通路

当前状态：每步 PETSc → SciPy CSR → JAX BCOO 的三段转换已经被
PETSc CSR values → cached BCOO structure 替换。

- fixed sparsity ⇒ 首次建立 CSR→BCOO 的 index mapping 和 diagonal
  positions，之后每步只用新 values 构造 BCOO，pattern 变化时自动重建。
  cache 覆盖同对象和跨对象相同 CSR pattern 两类情况。tiny 档 conversion
  已从约 `0.334s` 降到 cold 首轮约 `0.016s`，repeat warm 样本约
  `0.001s`；small-loop 真实单次 run 记录 `miss=1/hit=15`。
- JAX residual check 默认保留，用于开发期安全；`jax-cg-no-check`
  仅作为显式 benchmark / production-like 热路径开关，不能单独证明收敛。
- JAX sparse direct 路线：`--xla-jax-method spsolve` 直接消费 PETSc CSR
  `(data, indices, indptr)` 并调用 `jax.experimental.sparse.linalg.spsolve`。
  当前 CUDA 后端可走该 experimental direct solver，CPU 后端会退回 SciPy；
  因此它只作为 opt-in benchmark 候选，默认仍保持 CPU `spsolve`。
- PETSc 路线：`--xla-petsc-gpu` 已预留 `aijcusparse`/`cuda` 类型；
  组装若仍在 CPU，用 `Mat.setValuesCSR` 批量灌值，避免逐元素 setValue。
- AMGX：resources / config / solver handle 建一次，`persistent_resources`
  已在 options 中预留；每步只 `solver.setup(A)`（值变）+ `solve`。
  若 sparsity 不变可用 `replace_coefficients` 跳过重新 setup。

## Phase 5 — full-loop XLA / lax.scan（有前置条件）

只有当以下条件全部满足才评估：

- [ ] Phase 3 kernels 全部 device-resident，非输出步零 host copy
- [ ] 线性求解在 device 上闭环（JAX native 或 FFI 调 AMGX）
- [ ] host I/O 边界清理完成：输出步通过 `jax.experimental.io_callback`
      或分段 scan（在输出步之间 scan）实现
- [ ] representative 档 profiling 显示 python_overhead 仍是主要瓶颈

## State 归属表

| 字段 | 归属 | 说明 |
|---|---|---|
| T, u | GPU resident | 每步更新，仅输出步回传 |
| phase, printed, active, cooling_only | GPU resident (bool/int8) | Phase 3 kernel 输入输出 |
| T_ref, eqp | GPU resident | linear_elastic 时 eqp 不分配 |
| stress_quad, vm_quad | GPU resident | 仅输出步 reduce 到节点并回传 |
| mesh 坐标 / 连接 / quadrature 常量 | GPU constant | 启动上传一次 |
| sparsity pattern / CSR→BCOO 映射 | cached (device) | fixed mesh 建一次 |
| path 数组、layer→element 映射 | CPU 预处理 → GPU constant | Phase 2 产物 |
| output 文件名 / VTU writer / 日志 | CPU only | I/O stage 计时 |
| benchmark / fallback 控制 | CPU only | GPU 失败自动回退 spsolve 并记录原因 |

## 验收标准（每个 PR）

1. 附 `bench_mech100_xla.py` 前后对比 JSON + 表格（同机同档）。
2. 禁止用 dry-run 或 wrapper unit test 声称 GPU 加速。
3. 报告必须拆分 setup / activation / quad_state / material / history /
   postprocess / dof_to_quad / nonlinear_solve / nonlinear_solve_overhead /
   bc_initial_guess / residual_vector / residual_flatten / residual_bc /
   residual_projection / solver / conversion / transfer / local_assembly /
   global_matrix / assembly / cell_jacobian / face_jacobian /
   residual_scatter / io / python_overhead。
4. 输出 VTU 字段与 baseline 数值 diff 在容差内（另建 golden-file 测试）。
5. jax_solver 慢于 spsolve 时，用 `explain_gpu_vs_cpu` 的归因结论说明
   原因（conversion-dominated / solve-dominated / transfer-dominated），
   默认解算器保持 spsolve。
6. 不允许在 Phase 3/4 未完成前提交 lax.scan 全循环重写。
