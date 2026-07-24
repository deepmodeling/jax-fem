# Implementation Plan: Kaess 2023 论文级数值复现

**Feature Directory**: `specs/001-kaess-paper-reproduction`

**Branch**: `codex/r3-optimization`

**Date**: 2026-07-23

**Spec**: [spec.md](spec.md)

**Status**: Preview draft — CPU-small/GPU-formal sequencing is approved;
remaining scientific inputs, thresholds and P0 scope still block solver
implementation until the Review Gate is completed

## Summary

在不改变现有 medium 回归用途的前提下，新建一条论文一致性开发线：

1. 冻结公开论文证据、数字化曲线、未知输入和科学声明；
2. 逐项修正并验证底面约束、三维热源、精确激活、动态热表面、
   冷却环境、材料历史和 release；
3. 建立可重复的 CPU float64 小尺度 reference，覆盖真实自由度短前缀及
   build/cooling/release；
4. 以完全相同的物理和验收模型资格认证 GPU/hybrid，明确实际设备放置；
5. 通过后用加速后端完成 3×30、5×60 和 10×30 µm 正式梯度；
6. 使用预注册指标对比论文 Figure 8/9，并完成参数矩阵；
7. 生成第三方可复跑的证据包、性能报告和技术报告。

## Technical Context

**Language/Version**: Python `>=3.10`; 当前 WSL 环境 Python `3.13.13`

**Primary Dependencies**: JAX/JAXlib、JAX-FEM、NumPy/SciPy、PARDISO/MKL；
精确版本必须由正式 run manifest 捕获

**Storage**: 仓库内 YAML/JSON/CSV/VTU/日志文件；无数据库

**Testing**: pytest `9.1.1`，unit/integration/regression 分层

**Target Platform**: Windows 主机 + WSL2 Ubuntu；单线程 CPU/PARDISO
小样本为数值 reference；NVIDIA GPU/JAX + CPU PARDISO hybrid 为当前
加速候选；真正 `full_gpu` 为独立实验路线

**Project Type**: Python 科研求解器、CLI case launcher 和文件式证据流水线

**Performance Goals**: 先满足科学门禁；之后减少无效 Newton 分解和热学装配
时间。性能改善不得改变正式 QoI

**Constraints**: float64、固定线程、29,568 个参考 HEX8 单元、弱耦合热—力、
未来层必须具备正确未激活语义

**Scale/Scope**: 10×30 µm 标准例、5×60 µm 预正式例，以及后续约
20–25 个论文参数点

## Constitution Check

| Principle | Design status | Gate before implementation |
|---|---|---|
| Evidence traceability | Defined | G0 来源矩阵和假设登记必须审核 |
| Physics before performance | Defined | 所有 P0 差异必须有失败测试和修复任务 |
| Verification before long runs | Defined | 1层、3层、5×60 逐级通过 |
| CPU anchors before accelerated formal runs | Defined | 小尺度 CPU 覆盖全部物理阶段；无需先完成十层 CPU |
| Claim discipline | Pass | 始终声明 code-to-code，实验验证单列 |

**Constitution verdict**: 规划可继续；求解器实现和正式长算仍被
[spec.md](spec.md) 的 Review Gate 阻塞。

This `plan.md` is a review preview requested as part of the development
directory. It is not evidence that the Spec Kit `clarify` or `plan` phase has
passed. The local Spec Kit CLI/runtime scaffold is also not installed; see
`.specify/README.md`.

## Architecture Decisions

### AD-001 — 通用能力与论文算例分离

通用边界、热源、活动域、材料和验证能力放在 `jax_fem_am/`；Kaess
参数、冻结输入、启动器和比较脚本放在 `cases/kaess_2023/`。禁止在
通用模块中加入仅由 Kaess case id 触发的特判。

### AD-002 — Reference、资格、正式和回归启动器分离

保留 `run_kaess_medium_fullheight.sh` 的 `3×100 µm` 回归定位。新增
`run_kaess_cpu_reference.sh`、`run_kaess_gpu_qualification.sh` 和
`run_kaess_accelerated_formal.sh`；不得通过改变 medium 默认值把它
伪装成论文正式入口。

### AD-003 — 活动域而非有质量的“虚空”

未激活单元在热学和力学装配中严格不贡献方程。实现必须与小网格中
实际删除单元的参考解比较，不能依赖会改变 QoI 的质量或刚度占位因子。

### AD-004 — 构建、冷却和 release 分阶段检查点

弱耦合工作流保存三个不可变检查点：

1. `build_complete`;
2. `cooling_complete`;
3. `release_complete`.

release 只能读取通过构建与冷却门禁的状态，不能反向修改上游物理参数。

### AD-005 — CPU 小尺度数值 reference 与正式加速后端分离

CPU float64 单线程 reference 覆盖核、小网格、真实全网格短前缀、
1层完整 build/cooling/release 和缩减空间域三层 history/release
mini-cycle；每个样本至少重复两次。加速模式用完全相同输入逐级认证。
通过后可直接承担全网格 3×30、5×60、10×30 和参数矩阵，不要求完整
三层或十层 CPU 双跑。多线程 CPU 只作为相同资源预算下的性能 control。

当前候选命名为 `hybrid_gpu_assembly_cpu_pardiso`：JAX 局部装配在 GPU，
global sparse matrix 和 PARDISO 在线性求解 CPU。只有线性解、主要状态和
三阶段 global sparse 运算都满足设备驻留门时才允许正式标记 `full_gpu`。
`full_gpu` 可由 `xla_loop` 实现，也可由 host 控制的 PETSc CUDA/AMGX
实现；编排位置不是计算后端资格的替代指标。

### AD-008 — 科学等价与性能资格分离

科学等价使用 MKL/OMP 单线程 CPU direct solve 和原生 float64 checkpoint，
消除病态矩阵上的多线程排序噪声；性能资格另用相同 CPU 线程预算顺序比较
CPU control 与加速候选，分别报告冷启动、稳态、装配、线性求解和迭代数。
VTU 当前用于可视化，不作为唯一 float64 等价证据。

### AD-006 — 文件式、模式约束的证据包

正式 run manifest 和 paper-comparison report 必须通过
`contracts/` 中的 JSON Schema。GPU promotion 还必须通过跨文件语义
验证器，实际复算 run/qualification/profiler/mask/checkpoint hashes、
dtype、run-id membership 和 placement；Schema 只负责对象形状。原始场
数据、CSV 和日志继续使用现有文件流水线，避免引入数据库或新服务。

### AD-007 — 外部材料输入必须冻结

当前正式材料文件位于仓库外
`/home/user/work/159/materials/316L/ss316l_material_config_kaess.json`。
实现前必须选择以下一种并记录决定：

- 将只读副本和来源信息放入 `cases/kaess_2023/inputs/materials/`；或
- 保持外部文件，但在每次运行前验证固定 SHA-256，并在复现包发布。

首选前者；移动或复制现有材料资产需要用户审核。

## Project Structure

### Documentation for This Feature

```text
specs/001-kaess-paper-reproduction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── run-manifest.schema.json
│   ├── paper-comparison.schema.json
│   ├── backend-qualification.schema.json
│   └── backend-qualification-validation.schema.json
├── checklists/
│   ├── requirements.md
│   └── paper-parity.md
└── tasks.md
```

### Source Code

```text
jax_fem_am/
├── config/                 # 新边界、表面、活动域和运行配置
├── domain/                 # 相态、生命周期和历史语义
├── materials/              # 温变表、相变和 J2
├── physics/                # 论文热源、热边界、力学和 release
├── process/                # 激活、扫描路径和时间表
├── simulation/             # 分阶段编排和检查点
├── solvers/                # Newton/PARDISO；不承载论文特判
└── verification/           # 守恒、QoI、审计、manifest 和比较门

cases/kaess_2023/
├── references/             # 论文、数字化参考和来源元数据
├── inputs/                 # 计划新增：冻结材料、cell set、路径及哈希
├── analysis/               # 计划新增：QoI 提取和论文图比较
├── run_kaess_cpu_reference.sh
├── run_kaess_gpu_qualification.sh
├── run_kaess_accelerated_formal.sh
└── run_kaess_medium_fullheight.sh  # 保留为 regression

tests/
├── unit/                   # 公式、材料点、切线、边界和守恒
├── integration/            # 活动域、动态表面、release 和证据链
└── regression/             # 计划新增：CPU小样本、release 与后端等价
```

**Structure Decision**: 在现有单项目包结构中增量实现，不重构
`jax_fem_am`，不改动上游 `jax_fem` 公共接口，case 特定数据保留在
`cases/kaess_2023`。

## Implementation Phases

### Phase 0 — Clarify and Freeze

- 审核声明等级、阈值、标定/留出划分和作者资料获取计划。
- 修订 `kaess_2023.json` 中过时或自相矛盾的状态字段。
- 生成来源矩阵、假设登记、数字化 CSV 和输入哈希。

**Checkpoint G0**: 没有未登记的高影响输入；正式配置仍不可执行。

### Phase 1 — P0 Physics Parity

依赖顺序：

```text
底面 BC ─┐
论文热源 ├─> 精确活动域 ─> 动态顶面/冷却 ─> 材料历史 ─> release
材料点 ──┘
```

热源、底面 BC 和材料点测试可以并行开发；活动域必须先于动态顶面；
构建前状态必须先于 release。

**Checkpoint G1/G2**: 解析、unit、patch、活动域和 release 测试全部通过。

### Phase 2 — CPU Small-Scale Verification Baseline

依次执行材料点/element、small domain、single track、真实全网格
12–20 步短前缀和 `1×30 µm` 最小完整 build/cooling/release。CPU
reference 使用 float64、MKL/OMP=1 并重复两次；完成时间、路径、网格、
求解器容差、能量和 QoI 收敛。另建缩减空间域三层 mini-cycle，强制
覆盖多层扫描角、recoat、再熔化、历史累积、冷却和 release。

**Checkpoint G3**: 全部物理阶段已覆盖；kernel、small-domain、
real-DOF prefix、1层完整流程和缩减三层 mini-cycle 均以 MKL/OMP=1
重复两次并冻结 run id/hash；两档最细离散 QoI 差≤2%，solver-tolerance
门通过。

### Phase 3 — Accelerated Backend Qualification

先验证 `hybrid_gpu_assembly_cpu_pardiso`，再独立研究
`gpu_dominant_experimental`/`full_gpu`。每个候选用与 G3 完全相同的
case、mask、float64 checkpoint 和 acceptance model 比较字段、事件、
能量、收敛、build/cooling/release 及重复性。随后用相同线程预算做
性能 control。

**Checkpoint G4**: 候选模式只有同时满足 `numerically_qualified=true`、
`performance_qualified=true` 和 `promotion_eligible=true` 才能作为正式
accelerated backend；科学等价但性能未过门只保留 diagnostic verdict。
未通过只阻断该模式。当前没有成功 GPU sparse solve 证据时 `full_gpu`
保持 `unsupported`。

### Phase 4 — Accelerated Scale Bridge and Standard Case

使用 G4 已通过的模式依次执行 `3×30 → 5×60 → 10×30 µm`。每一级先
通过能量、收敛、活动域和阶段检查点再进入下一级。完整十层 CPU 双跑
不再是前置；发生超门、线性解次数膨胀或新物理代码变更时，回到 G3
CPU 小尺度 reference 重新资格。

**Checkpoint G5**: 10×30 标准例 build/cooling/release 完成并生成
Figure 8/9 verdict。`pass` 才产生 `promotion_passed` 并进入正式参数矩阵；
`partial`/`fail` 关闭本轮证据链但只允许后续 diagnostic matrix。

### Phase 5 — Accelerated Parameter Matrix and Performance

标准例通过后，以同一合格加速模式运行 30/60 µm、预热、功率、速度和
恒定线能量组。标定和留出工况按 G0 登记，不得事后互换；性能报告同时
保留 CPU performance control、冷/热启动和阶段分解。

**Checkpoint G6**: 参数趋势、性能和资源证据完整；没有把 hybrid
误称为 full GPU。

### Phase 6 — Reproduction Package and Report

从 JSON/CSV 自动生成图表、误差表、claim matrix 和技术报告。使用新
输出目录执行独立 quickstart 复跑。

**Checkpoint G7**: 第三方能够生成同样的输入哈希、QoI 和比较图。

## Verification Checkpoints

| Checkpoint | Required evidence |
|---|---|
| G0 | 来源矩阵、假设登记、数字化误差、冻结阈值 |
| G1 | P0 physics tests and implementation review |
| G2 | unit/integration suite, analytical error report |
| G3 | CPU small-reference manifests, native-float64 checkpoints, time/path/mesh/solver and repeatability report |
| G4 | CPU/GPU active-domain field, event, cooling/release and performance qualification |
| G5 | accelerated 3×30/5×60/10×30 manifests and Figure 8/9 comparison |
| G6 | accelerated parameter matrix, backend-placement and speedup report |
| G7 | clean-directory rerun and final claim matrix |

## Parallel Opportunities

- **Can run in parallel**: 来源提取、底面 BC tests、热源 tests、材料点 tests、
  JSON contracts、报告模板。
- **Must be sequential**: 精确激活 → 动态活动顶面；构建/冷却通过 →
  release；CPU small reference → GPU qualification → 3层 → 5×60 →
  10×30 accelerated formal.
- **Needs coordination**: `stepper.py`、`schema.py`、正式 launcher 和
  run-manifest contract 的接口必须先冻结再并行。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 作者输入缺失 | 无法声称 input-equivalent | 公开信息复现 + 假设敏感性；请求作者资产 |
| 当前活动域设计影响矩阵规模 | 高开发风险 | 先做小网格删除参考和最小 API |
| J2 切线/残差失配 | Newton 停滞、长算失败 | V 形谷/有限差分切线验证先于正式例 |
| B-bar 与近理想塑性粉末病态 | 非确定 release 位移 | 材料/模态测试、CPU 固定线程、锚点敏感性 |
| 外部材料文件漂移 | 无法复跑 | 仓库内冻结或强制 SHA-256 |
| GPU 双精度/并行归约差异 | 位移超门 | 单线程 CPU reference、原生 float64 checkpoint、活动域 mask 和分级资格 |
| 多线程 PARDISO 排序放大病态模态 | CPU reference 自身分叉 | 科学基线 MKL/OMP=1；多线程只进入性能 control |
| Hybrid 被误报为 full GPU | 性能/方法声明失真 | 阶段级设备 placement contract；full-GPU 条件门 |
| liquid/mushy 消融被当作 GPU 修复 | 物理模型被平台调参污染 | 消融仅 diagnostic；正式输入需来源和独立物理审批 |
| 长算在上游失败后继续 | 浪费数十小时 | launcher 强制读取 gate 状态 |
| 事后调阈值 | 复现可信度丧失 | 阈值变更需规格修订和用户审批 |

## Complexity Tracking

| Potential complexity | Why it may be needed | Simpler alternative |
|---|---|---|
| 活动域自由度压缩 | 对齐 MODEL CHANGE 的零贡献语义 | 固定全局 DOF + mask；只有证明数学等价时采用 |
| CPU/hybrid/full-GPU 多模式 | 资格和性能研究 | 先让当前 hybrid 通过；full-GPU 不阻塞论文复现 |
| 多种证据 schema | 科研审计和自动报告 | 保持两个核心 schema，避免数据库 |

## Open Questions

所有阻塞问题见 [spec.md](spec.md#open-questions)。其中声明等级、验收阈值、
标定/留出划分和外部材料冻结方式必须在 Phase 0 结束前批准。
