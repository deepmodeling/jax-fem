# mech100 XLA 升级路线图（v03）

目标：把 `am_thermal_stress_macro_intersection_mech100_XLA.py` 从"JAX/GPU 线性求解器尝试层"演进为"面向 GPU 高效执行的热-力求解流程"，同时保持物理流程和输出字段的字节级兼容。

## Phase 1 — Profiling harness（已落地骨架）

- `ProfilingReport`：按 stage 累计 wall time，stage 集合固定为
  `solver / assembly / conversion / transfer / io / python_overhead`。
  未显式归因的时间自动落入 `python_overhead` —— 这正是 826k 扫描步下的
  每步固定开销（path step state、layer activation 判断、output 判断、dict 拷贝）。
- `benchmarks/bench_mech100_xla.py`：tiny / medium / representative 三档，
  对 spsolve / jax / jax-precond / petsc / petsc-gpu / amgx 逐一实跑并出表。
- GPU utilization：representative 档跑期间用
  `nvidia-smi dmon -s um -d 1 -o T` 旁路采样，采样文件与 profile JSON 一起归档。
- 基线要求：在任何优化 PR 之前，先在目标机器上固化三档 × spsolve 的
  baseline JSON，之后所有 PR 用同机同档对比。

接线点：base solver 需要暴露 `default_solver_options()` 和
`run(solver_options=..., profiler=..., argv=...)`，主循环内用
`with profiler.stage("solver"): ...` 等打点。打点侵入极小，且
profiler=None 时可用 no-op 实现零开销退化。

## Phase 2 — 主循环固定开销

优先级排序（按每步成本 × 步数）：

1. **path step state**：把逐步读取/解析改为一次性预处理成
   结构化数组（time, x, y, z, power, layer_id, is_output_step），
   主循环只做整型索引。output 判断预先算成布尔数组，循环内零分支字符串比较。
2. **layer activation**：预计算 layer→element 的激活映射
   （每层一个 element index 数组），激活变成一次数组切片赋值，
   而不是逐元素几何判断。
3. **solver options / dict churn**：`rewrite_solver_options` 只在启动时
   调用一次；主循环内禁止 deep copy 任何 options dict。
4. **sparse structure cache**：fixed mesh ⇒ Jacobian 的 sparsity pattern
   不变。首步 symbolic 分析后缓存 (indices, indptr, permutation)，
   后续只更新 values。spsolve 路线可切 `splu` 复用符号分解
   （若矩阵值也不变的线弹性子段，直接复用数值分解）。

## Phase 3 — 纯数组 kernel JIT 化

迁移顺序（fixed shape、无控制流依赖 host 的先走）：

1. activation mask 更新（printed/active/cooling_only 布尔数组）
2. phase update（相态状态机，可写成 `jnp.where` 链）
3. T_ref update
4. material quadrature update（温度→材料参数插值，纯 gather + 多项式）
5. stress / von Mises postprocess（quadrature 级张量运算）

规则：

- 每个 kernel 独立 `jax.jit`，固定 shape（active 用 mask 而不是变长数组，
  避免 recompile）。
- kernel 之间的 state 保持 device-resident；**非输出步禁止任何
  device→host copy**。用 `jax.block_until_ready` 只在计时和输出步调用。
- `linear_elastic` 模式：eqp 等塑性历史变量完全不更新、不分配。

## Phase 4 — 稀疏线性求解数据通路

当前问题：每步 PETSc → SciPy CSR → JAX BCOO 的三段转换。

- fixed sparsity ⇒ 首步建立 CSR→BCOO 的 index mapping，之后每步只
  `bcoo.data = values[perm]`（一次 device 上 gather），转换成本从
  O(nnz × python) 降到 O(nnz) 纯 GPU。
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
3. 报告必须拆分 solver / conversion / transfer / io / python_overhead。
4. 输出 VTU 字段与 baseline 数值 diff 在容差内（另建 golden-file 测试）。
5. jax_solver 慢于 spsolve 时，用 `explain_gpu_vs_cpu` 的归因结论说明
   原因（conversion-dominated / solve-dominated / transfer-dominated），
   默认解算器保持 spsolve。
6. 不允许在 Phase 3/4 未完成前提交 lax.scan 全循环重写。
