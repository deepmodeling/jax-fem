# Kaess 中等尺度完整高度测试（约 2–3 小时）

## 定位

`run_kaess_medium_fullheight.sh` 是求解器和完整后处理 pipeline 的中等尺度回归，不是 Kaess 参考分辨率结果，也不是实验验证。

它保留完整的 C3D8 powder-margin 网格（29,568 单元）和 0.3 mm 部件构建高度，但将参考构建的 `10 x 30 um` 合并为 `3 x 100 um` 宏观沉积层。这样不会只计算底部 3/10 高度；第三层路径顶面仍到达 `z = 0.6 mm`（支撑顶面为 `z = 0.3 mm`）。代价是层间热循环和逐层塑性历史被粗化；三次扫描的激光总作用时间/输入能量也低于十次参考扫描，未通过提高功率做不稳定的等能量补偿。因此结果只用于全流程、稳定性和定性趋势检查，温度峰值、塑性应变和残余应力不能当作参考构建预测。

## 默认求解设定

| 项目 | 中等尺度默认值 | 参考设置/说明 |
|---|---:|---|
| 网格 | 29,568 C3D8，B-bar | 完整 powder-margin 网格，不降空间网格 |
| 构建层 | 3 x 100 um | 覆盖与 10 x 30 um 相同的 0.3 mm 高度 |
| 路径采样 | 50 um | 一个 beam radius；参考为 25 um |
| hatch / beam radius | 100 / 50 um | 不变 |
| 功率 / 扫描速度 | 250 W / 0.85 m/s | 不变 |
| 路径状态 | 334 | 308 个 laser-on，26 个 jump |
| recoat | 2 x 45 s，各 10 步 | 总等待 90 s，与参考 9 x 10 s 相同；时间离散更粗 |
| cooling | 30 x 1 s | 不变 |
| 总热步 | 384 | `334 + 20 + 30` |
| 力学 | J2，Abaqus/strict-residual 兼容验收 | `rel_tol=5e-5`、line search、最多 3 级 cutback |
| 线性求解 | CPU PARDISO Phase23 | 保留确定性和符号分析复用 |
| 输出链 | cooling、release、XRD、audit、response gate、manifest | 全部保留 |

注意：熔化、凝固和参考态重置会强制力学更新，`MECH_EVERY=20` 只是补充周期，不应把预计力学次数简单算成 `384/20`。

## 启动

在 WSL 的 `jax-fem-env` 环境中运行：

```bash
cd "/mnt/c/Users/user/Documents/New project/jax-fem-r3-opt"
bash cases/kaess_2023/run_kaess_medium_fullheight.sh --print-plan

mkdir -p /home/user/work/159/output/logs
RUN_ID="medium3h_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID
bash cases/kaess_2023/run_kaess_medium_fullheight.sh \
  2>&1 | tee "/home/user/work/159/output/logs/${RUN_ID}.log"
```

启动器默认创建带 UTC 时间戳的新输出目录，不会覆盖已有结果。若要固定目录，可显式设置 `OUT_ROOT`。

为避免继承 R3/R4 运行残留的环境变量，这个专用启动器会锁定网格、材料、宏层、路径、热学、力学和线性求解设置；只应覆盖 `WORK_ROOT`、`RUN_ID`、`OUT_ROOT`、`PYTHONUNBUFFERED` 这类非物理运行参数。需要其他物理配置时应直接使用 `run_kaess_phase2.sh` 创建另一个明确命名的算例，不要复用本启动器。

## 2–3 小时预算与早停门槛

预算基于同一 C3D8 网格的 R4 线性求解成本，以及正常收敛算例约 5–8 次 Newton 线性解的范围。新收敛修复后还没有一条完整 C3D8 生产运行，因此 `1.9–3.1 h` 是首轮标定区间，不是硬性保证。

- step 50：不超过 22 min 为正常；22–25 min 为观察区；超过 25 min 建议停止。
- step 100：不超过 42 min 为正常；42–47 min 为观察区；超过 47 min 建议停止。
- 若日志仍出现“`relative < 5e-5` 但触发 cutback”，说明新验收修复未生效，应停止。
- 偶发一次 2-substep 可继续观察；反复出现 4/8-substep refinement 时，三小时预算大概率失守。

逐步进度可用 ledger 行数直接读取（将 `<输出目录>` 替换成启动时打印的路径）：

```bash
wc -l "<输出目录>/thermal_energy_ledger.jsonl"
```

## 完成判据

不能只以进程退出作为完成。至少检查：

- 日志到达第 3 个宏层和最终 cooling；
- `release.vtu` 存在；
- `v06_run_audit.json`、`v06_response_gate.json`、`xrd_operator_smoke.json` 存在；
- `v06_manifest.json` 中 `completeness.complete` 为 `true`；
- `profile.json` 标记 `pardiso_v07(phase23)`，日志末尾的 PARDISO 统计为 `pattern_rebuilds: 0`；
- wall time 落在预期区间，或记录偏离原因。

## 与完整 10 层构建的边界

完整 `10 x 30 um` 构建即使把路径采样放宽到 100 um，仍有 773 个热步，按当前事件驱动力学机制约需 3.5–6 h，无法诚实保证在 2–3 h 内完成。需要参考级 10 层结果时，应另开长算例；不要把本测试的宏层结果当作定量残余应力或 XRD 对比结论。
