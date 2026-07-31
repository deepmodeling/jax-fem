# 求解器行为备忘(V 会话实测,供主线参考)

这里记录 V 轨在使用共享求解器时踩到、且**文档里没有写明**的行为。
均为实测,附证据运行;不是猜测。不修改任何决策,只陈述事实。

---

## 1. 扫描步的 dt 由路径行间距决定,但**第一步用 `--dt`**

**机制**:带 `--path-file` 的运行中,扫描步的时间步长等于相邻路径行的时间差;
但**第 0 步例外,它用 `--dt` 的值**。

**证据**(两次运行,路径行间距完全相同,仅 `--dt` 不同):

| 运行 | `--dt` | 路径行间距 | 实际首步 dt | 首步沉积 | 首步 T_max |
|---|---|---|---|---|---|
| v2_meshconv_r40 | 20 µs | 19.23 µs | **20.0 µs** | 2.73 mJ | 2882 K |
| v2_dtprobe_ss12.5 | 50 µs | 19.23 µs | **50.0 µs** | 6.95 mJ | 11377 K |

同一网格族、同一功率、同一路径采样,仅首步 dt 差 2.5 倍,首步峰温差 3.9 倍。

**为什么危险**:第 0 步把能量灌进一个刚激活、还没有建立温度场的冷域,
所以它对 dt 特别敏感。若 `--dt` **大于**稳态路径间距,首步就成为全程最热的一步,
而且这个峰值会污染任何"最高温度"统计。

**V1 为何没踩到**:V1 传 `--dt 1.0e-5`(10 µs),路径间距 15.6 µs ——
首步小于稳态步,反而更保守,所以问题从未显现。

**建议**:凡使用 `--path-file` 的运行,把 `--dt` 设成**不大于**最小路径行间距。
V 轨现在的做法:12.5 µm 采样 @ 650 mm/s = 19.23 µs → `--dt 1.9e-5`。

**对主线的相关性**:目前 L0 用的是 `--scan-steps-per-layer` 参数化扫描
(非 path-file),`--dt 20.0` s 属于件级时间尺度,不在这个陷阱里。
但一旦主线改用 XY 路径文件驱动(D-10 之后的自然演进),这条就适用。

---

## 2. 诊断输出频率会误导"峰值"类统计

`--summary-every N` 控制日志里 `T_max=` 这类行的频率。若设成大于总步数
(例如 1000),日志只留下**首步与末步**两条,任何对日志做 max 的统计
实际上得到的是"首步值"而非全程峰值。

V 会话曾因此把首步的 11377 K 当成稳态峰温,并据此得出了错误结论
(详见 v2-cube-rs 的偏差册与本目录 regression 报告)。

**建议**:凡是要报告峰值温度/应力的运行,`--summary-every` 取 ≤ 5。

---

## 3. `--mechanics-every 0` 与 thermal-only 代用对象(已由主线修复)

历史问题:`--mechanics-every 0` 会在 `stepper.py:966` 崩溃,因为
`_ThermalOnlyMechanicsProblem` 代用对象缺 `set_flow_curve_active_mask`。
V1 期间用 `--no-xla-thermal-only-mechanics-surrogate` 绕行。
**主线已在 e7acada 修复**;此条仅作历史记录。

---

## 4. `run_audit` 无法审计纯热运行

`jax_fem_am.verification.run_audit` 要求运行目录里同时存在 `step_*.vtu`
与 `release.vtu`;纯热运行(无力学、无释放)不产生 `release.vtu`,
审计直接抛 `ValueError: run must contain step_*.vtu and release.vtu`。

V1 的五个纯热运行因此都没有 run_audit 工件,只能靠热能台账
(`thermal_energy_ledger_summary.json`)作为验收证据。

**建议**:若纯热运行会成为常规验证手段(V1 已经是),run_audit 需要一个
thermal-only 模式,或明确声明纯热运行的验收工件就是能量台账。

---

## 5. 环境记录已过期:jax **默认走 GPU**

台账 F 节记载 "jaxlib is CPU-only; the RTX 5080 is unused until jax[cuda12]
is installed"。实测(2026-07-31):

```
jax 0.10.2 / jaxlib 0.10.2
默认后端: gpu
所有设备: [CudaDevice(id=0)]
已装: jax-cuda13-pjrt, jax-cuda13-plugin, nvidia-cu* 全套 (CUDA 13)
硬件: RTX 5080, 16 GB
```

**影响**:任何**没有**显式设置 `JAX_PLATFORM_NAME=cpu` 的运行现在都在 GPU 上执行。
这会改变性能数字的解释(某条运行到底是 CPU 还是 GPU 的耗时),也解释了
未设该变量时出现的显存 OOM 报错(16 GB 卡上试图预分配 11.9 GB)。

V 轨所有运行均显式钉 `JAX_PLATFORM_NAME=cpu`,不受影响,且金标基线
(c3d4 / c3d8 两对)都是在 CPU + `MKL_NUM_THREADS=1` 下建立的确定性基线。

---

## 6. 已建立的确定性金标基线(供 GPU 移植验收使用)

| 基线 | 网格 | 单元 | 目录 |
|---|---|---|---|
| c3d4(遗留对照臂) | kaess_cantilever_c3d4_powder | TET4 | `kaess_golden_regression_20260730` / `_b` |
| c3d8(生产路径) | kaess_cantilever_c3d8_powder_margin | HEX8+B-bar | `kaess_c3d8_golden_20260730_a` / `_b` |

两对均为**同配置两次独立运行、逐位一致**(T、u、max_temperature_history
在 step 0 / 200 / release 全部 `max_abs = 0.0`),协议为
`MKL_NUM_THREADS=1` + `JAX_PLATFORM_NAME=cpu`。

**用于 GPU 移植验收的建议判据**:GPU 版不可能逐位一致(浮点归约顺序不同),
但可用 `regression-20260730/compare_golden.py` 给出量化上界——
热场相对偏差若在 1e-10 量级即为纯归约顺序差异;若达到 1e-6 以上,
说明改动触及算法而不仅是后端。这样"移植未改变物理"成为可证命题。
