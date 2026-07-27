---
description: "Dependency-ordered task list for Kaess 2023 paper-level reproduction"
---

# Tasks: Kaess 2023 论文级数值复现

**Input**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`

**Status**: G0 approved; T014-T017 and T019-T021 are complete. T018 material
approval, anchor sensitivity, and the remaining unchecked parity items still
block G1/G2.

**Organization**: 任务按 user story 和科学 gate 组织。`[P]` 表示不同文件、
无前置依赖时可并行；测试任务必须先失败，再实现对应能力。

## Format

`[ID] [P?] [Story] Description`

- **[P]**: 可与同阶段其他 `[P]` 任务并行。
- **[US#]**: 对应 `spec.md` user story。
- 每项包含 Acceptance、Verify 和 Files。
- 单项通常不超过五个文件；正式长算任务除外，但不得修改代码。

## Phase 1 — Source Freeze and Shared Contracts

- [x] T001 [P] [US1] 建立论文来源清单 — 在 `cases/kaess_2023/inputs/source-manifest.yaml` 记录论文、全文、图件、材料、网格和路径的来源类别及 SHA-256。
  - **Acceptance**: 每个 critical/high-impact 输入有 evidence id 和 locator。
  - **Verify**: 新增 contract test 能拒绝缺哈希或未知来源类别。
  - **Files**: `cases/kaess_2023/inputs/source-manifest.yaml`,
    `tests/contract/test_kaess_source_manifest.py`

- [ ] T002 [P] [US1] 冻结 Figure 8/9 数字化数据 — 将现有 JSON 数值导出为带单位和读图误差的 CSV。
  - **Progress**: Figure 8b 的四个锚点和 Figure 9a/9b 的现有数值已冻结；
    完整 Figure 8 曲线的像素级数字化仍未完成。
  - **Acceptance**: Figure 8 曲线、Figure 9a/9b 标量和不确定度可独立读取。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_reference_data.py`
  - **Files**: `cases/kaess_2023/references/digitized/fig8_sigma_x.csv`,
    `cases/kaess_2023/references/digitized/fig9_bending.csv`,
    `tests/unit/test_kaess_reference_data.py`

- [x] T003 [P] [US1] 建立 assumptions 和 deviations register — 登记锚点、时间表、路径次序、材料历史和 release cell set 未知项。
  - [x] 建立作者输入请求/响应日志，记录请求日期、资产、状态和取得文件哈希。
  - **Acceptance**: 每个假设有影响级别、范围、affected QoI 和处理状态；
    每个 critical 未知项都映射到作者请求或敏感性任务。
  - **Verify**: schema/contract test 拒绝未分类的 critical 假设。
  - **Files**: `cases/kaess_2023/inputs/assumptions.yaml`,
    `cases/kaess_2023/inputs/deviations.yaml`,
    `cases/kaess_2023/inputs/author-input-requests.md`,
    `tests/contract/test_kaess_assumptions.py`

- [x] T004 [US1] 清理 benchmark metadata — 修正 `verification_status` 与已存在数字化数据的矛盾，保留历史说明。
  - **Acceptance**: metadata、CSV 和 claim boundary 一致。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_reference_data.py`
  - **Files**: `cases/kaess_2023/references/cases/kaess_2023.json`

- [x] T005 [P] [US5] 为五个 JSON contracts 和跨文件语义验证器增加失败测试 — 验证最小有效对象、缺字段、错误 hash/claim/精度、kernel-only 或 experimental promotion、未类型化 metric、hybrid 冒充 full-GPU、Figure 8/9 漏项、阈值错绑、空不确定度和 all-fail partial。
  - **Acceptance**: schema 形状与跨文件身份/哈希/dtype/run-id/placement/
    threshold/status 反例均被拒绝；阈值 artifact/hash/approver 必须绑定
    G0 配置和当前 run manifest；validation verdict/promotion 双条件、
    dirty diff、float64、线程预算、环境/硬件与顺序执行身份均有失败测试；
    数值指标由原生 checkpoint 重算，性能由独立 raw evidence 重算；
    `mesh`、`scan_path`、material、最终 used config 与实际 command 均须
    作为内容寻址 input，used config 内 `config/inp/path_file` 必须回指对应
    artifact；energy gate 必须逐 run 绑定 typed ledger/summary/run-audit
    wrapper，并按冻结的 `thermal_energy_closure` 阈值重算；
    `performance_pair` 必须唯一绑定有序 CPU/candidate run ids，wall-time
    样本由 rehashed execution-order 区间重算，并与对应 manifest 的
    `completed_utc/resource_usage.wall_seconds` 交叉绑定；线性求解次数由
    checkpoint `linear_solve_count` 重算；T032/T035 产生并接入 typed QoI evidence
    前，paper comparison 必须保持
    `blocked`；新增依赖前必须询问用户。
  - **Verify**: `python -m pytest -q tests/contract/test_kaess_contracts.py`
  - **Files**: `tests/contract/test_kaess_contracts.py`,
    `jax_fem_am/verification/backend_qualification.py`, `pyproject.toml`,
    `specs/001-kaess-paper-reproduction/contracts/run-manifest.schema.json`,
    `specs/001-kaess-paper-reproduction/contracts/paper-comparison.schema.json`,
    `specs/001-kaess-paper-reproduction/contracts/backend-qualification.schema.json`,
    `specs/001-kaess-paper-reproduction/contracts/energy-audit-evidence.schema.json`,
    `specs/001-kaess-paper-reproduction/contracts/backend-qualification-validation.schema.json`

- [x] T006 [US1] 冻结 paper-parity protocol — 创建标准工况、QoI、阈值、标定/留出和 stop rules 的机器可读配置。
  - **Acceptance**: Review Gate 的人工决定全部可表示并有批准字段。
  - **Verify**: contract test 和人工 diff review。
  - **Files**: `cases/kaess_2023/inputs/paper-parity-config.yaml`,
    `tests/contract/test_kaess_parity_config.py`

### Checkpoint G0 — Human Source Review

- [x] CHK023–CHK030 全部批准。
- [x] `paper-parity-config.yaml` 为 immutable approved protocol。
- [x] 外部材料输入冻结为 `KAESS_MATERIAL_CONFIG` 指向的外部文件及
  SHA-256。
- [x] 审批证据冻结在 `cases/kaess_2023/inputs/g0-approval.json`，阈值
  artifact 冻结在 `cases/kaess_2023/inputs/threshold-set.json`。
- 未达到本 checkpoint 时，Phase 2 只允许编写失败测试，不允许正式实现。

## Phase 2 — Foundational Failing Tests for P0 Physics

- [x] T007 [P] [US2] 编写论文式底面 BC 失败测试 — 测试全底面 `uz=0`、最小 `x/y` 锚点、满秩和自由冷缩。
  - **Acceptance**: 当前 full clamp 因面内过约束而按预期失败。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_paper_bottom_bc.py`
  - **Files**: `tests/unit/test_kaess_paper_bottom_bc.py`

- [x] T008 [P] [US2] 编写半球三维高斯热源失败测试 — 测试中心值、径向/深度衰减、平移不变性和吸收功率积分。
  - **Acceptance**: 当前 plane×depth source 因公式/归一化不符而失败。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_hemispherical_source.py`
  - **Files**: `tests/unit/test_kaess_hemispherical_source.py`

- [x] T009 [P] [US2] 编写活动域删除等价失败测试 — 对比 masked 活动域和物理删除未激活单元的小网格。
  - **Acceptance**: 当前 `inactive_mass_factor=1.0` 因热质量差异而失败。
  - **Verify**: `python -m pytest -q tests/integration/test_active_domain_equivalence.py`
  - **Files**: `tests/integration/test_active_domain_equivalence.py`

- [x] T010 [P] [US2] 编写动态暴露顶面和冷却环境失败测试 — 测试未来层不遮挡活动顶面、表面积分和冷却 ambient 切换。
  - **Acceptance**: 当前静态 exterior face 和固定 ambient 按预期失败。
  - **Verify**: `python -m pytest -q tests/integration/test_active_surface_boundary.py`
  - **Files**: `tests/integration/test_active_surface_boundary.py`

- [x] T011 [P] [US2] 编写材料/相态历史失败测试 — 测试粉末 `k(T)`、潜热、不可逆 powder→solid 和循环熔化历史。
  - **Acceptance**: 无来源二次缩放或 history reset 会导致测试失败。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_material_history.py`
  - **Files**: `tests/unit/test_kaess_material_history.py`

- [x] T012 [P] [US2] 编写 J2 曲线和一致切线失败测试 — 测试单轴加载—卸载—再加载、温变曲线和有限差分/V 形谷。
  - **Acceptance**: 残差与切线异源时给出可定位失败。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_j2_tangent.py`
  - **Files**: `tests/unit/test_kaess_j2_tangent.py`

- [x] T013 [P] [US2] 编写精确 release cell-set 失败测试 — 测试 cell-set 身份、范围、保留根墙、刚体约束和解析悬臂方向。
  - **Acceptance**: 未验证 box 选择不能通过 paper release gate。
  - **Verify**: `python -m pytest -q tests/integration/test_kaess_release_cellset.py`
  - **Files**: `tests/integration/test_kaess_release_cellset.py`

### Checkpoint — Red Tests Confirmed

- 每个实现任务 `T014–T020` 仅依赖其对应红测 `T007–T013` 已因预期
  物理差异失败，而非导入/路径错误；允许按单项 red → green 闭环推进，
  不要求等待全部红测完成。
- 在关闭 G1/G2 前，`T007–T013` 的全部 red evidence 与对应 green
  regression 必须齐备并保存 failure summary。

## Phase 3 — User Story 2: P0 Physics Implementation

- [x] T014 [US2] 实现论文式最小刚体底面 BC — 增加 `paper_minimal` BC 配置和节点选择，不改变 legacy `fixed`。
  - **Acceptance**: 满秩、自由冷缩和四个等价角点锚定变体测试通过；
    used config 记录实际 bottom-node count、锚点 node ids/coordinates 和
    rotation component。论文未公开的精确锚点仍保留为 G2 物理敏感性义务。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_paper_bottom_bc.py tests/integration/test_v06_release_anchor_box.py`
  - **Files**: `jax_fem_am/config/schema.py`, `jax_fem_am/physics/release.py`,
    `jax_fem_am/simulation/stepper.py`,
    `cases/kaess_2023/run_kaess_phase1.sh`,
    `cases/kaess_2023/run_kaess_phase2.sh`,
    `tests/unit/test_kaess_paper_bottom_bc.py`

- [x] T015 [P] [US2] 实现论文半球三维高斯热源 — 新增显式 source model，保持 legacy source 可选。
  - **Acceptance**: 公式采样和功率积分误差≤0.5%。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_hemispherical_source.py`
  - **Files**: `jax_fem_am/config/schema.py`, `jax_fem_am/physics/thermal.py`,
    `jax_fem_am/simulation/stepper.py`,
    `jax_fem_am/verification/thermal_ledger.py`, `jax_fem_am/io/vtu.py`,
    `tests/unit/test_kaess_hemispherical_source.py`

- [x] T016 [US2] 实现未激活单元零贡献活动域 — 热容、导热、刚度、残差和求解 DOF 均遵循冻结活动域语义。
  - **Acceptance**: 小网格删除参考相对差≤`1e-8`，占位因子不影响 QoI。
  - **Verify**: `python -m pytest -q tests/integration/test_active_domain_equivalence.py tests/unit/test_v06_lifecycle.py`
  - **Files**: `jax_fem_am/process/activation.py`, `jax_fem_am/materials/phases.py`,
    `jax_fem_am/simulation/stepper.py`, `jax_fem_am/simulation/acceleration.py`,
    `jax_fem_am/verification/thermal_ledger.py`, `jax_fem/fe.py`,
    `jax_fem/solver.py`, `tests/integration/test_active_domain_equivalence.py`,
    `tests/contract/test_jax_solver_preconditioner.py`,
    `tests/unit/test_v06_thermal_ledger.py`,
    `specs/001-kaess-paper-reproduction/evidence/t009-t016-active-domain.md`

- [x] T017 [US2] 实现动态暴露表面和阶段 ambient — 从活动/未激活界面构造真实暴露面；冷却切换环境时间表。
  - **Acceptance**: 面积及热流积分误差≤0.5%，未来层不遮挡。
  - **Verify**: `python -m pytest -q tests/integration/test_active_surface_boundary.py tests/unit/test_v06_thermal_balance.py`
  - **Files**: `jax_fem_am/physics/thermal.py`, `jax_fem_am/process/schedule.py`,
    `jax_fem_am/simulation/stepper.py`,
    `tests/integration/test_active_surface_boundary.py`,
    `specs/001-kaess-paper-reproduction/evidence/t010-t017-active-surface.md`

- [ ] T018 [P] [US2] 对齐粉末、潜热和相态历史 — 冻结 powder `k(T)` 和潜热语义；移除未获批准的二次缩放/重置。
  - **Acceptance**: 潜热积分≤0.5%，循环材料点和自由膨胀测试通过。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_material_history.py tests/unit/test_v06_material_validation.py`
  - **Files**: `jax_fem_am/materials/tables.py`, `jax_fem_am/materials/phases.py`,
    `jax_fem_am/domain/events.py`, `tests/unit/test_kaess_material_history.py`

- [x] T019 [US2] 对齐温变塑性曲线和 J2 一致切线 — 支持冻结多点温变塑性输入并让残差/切线同源。
  - **Acceptance**: 加载循环与 V 形谷测试通过，旧 J2 回归不退化。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_j2_tangent.py tests/unit/test_v06_j2_kernel.py tests/unit/test_v03_bbar_hex8.py`
  - **Files**: `jax_fem_am/materials/j2.py`, `jax_fem_am/physics/mechanics.py`,
    `jax_fem_am/materials/material_validation.py`,
    `tests/unit/test_kaess_j2_tangent.py`
  - **Evidence**: `specs/001-kaess-paper-reproduction/evidence/t012-t019-j2-flow-curve.md`.
    The solver capability is closed; the pending Figure 4(b) material candidate
    remains outside formal G0 and does not close `P0-J2`.

- [x] T020 [US2] 实现显式 release cell set — 从冻结输入读取、可视化和哈希切割/锚定 cell set。
  - **Acceptance**: 空集、越界、零件本体误切和刚体漂移被拒绝。
  - **Verify**: `python -m pytest -q tests/integration/test_kaess_release_cellset.py tests/integration/test_v06_release_anchor_box.py`
  - **Files**: `jax_fem_am/physics/release.py`, `jax_fem_am/simulation/stepper.py`,
    `cases/kaess_2023/inputs/release-cellset.json`,
    `tests/integration/test_kaess_release_cellset.py`

- [x] T021 [US2] 运行 P0 完整回归并更新 parity checklist — 运行所有现有 physics/solver tests 与 T007–T013 新测试。
  - **Acceptance**: 无预期外失败；P0 gates 有机器可读证据。
  - **Verify**: 使用 `quickstart.md` 的 physics/solver command。
  - **Files**: `specs/001-kaess-paper-reproduction/checklists/paper-parity.md`,
    `cases/kaess_2023/inputs/deviations.yaml`
  - **Evidence**: clean commit `8f1603f` passed `588` tests with `2`
    conditional skips and `16` passing subtests; see
    `specs/001-kaess-paper-reproduction/evidence/t021-p0-regression.json`.

### Checkpoint G1/G2 — Physics and Code Verification

- PAR010–PAR027 全部通过。
- 代码审查确认无 Kaess 特判进入通用包。
- 未通过时禁止进入计算验证梯度。

## Phase 4 — User Story 3: CPU Small-Scale Verification Baseline

- [ ] T022 [P] [US3] 建立 single-track 能量验证 case — 提供热源输入、焓增、边界损失和网格捕获率输出。
  - **Acceptance**: 能量闭合≤1%，三档路径/时间/网格加密呈收敛。
  - **Verify**: `python -m pytest -q tests/regression/test_kaess_single_track.py`
  - **Files**: `cases/kaess_2023/run_kaess_verification_ladder.sh`,
    `tests/regression/test_kaess_single_track.py`

- [ ] T023 [US3] 建立真实全网格 12–20 步 CPU reference — 使用参考自由度、active/printed mask 和原生 float64 checkpoint，以 MKL/OMP=1 重复两次。
  - **Acceptance**: 输入/hash 一致；重复场和 QoI 通过门；全域未激活 `u_max` 不进入资格指标。
  - **Verify**: `python -m pytest -q tests/regression/test_kaess_real_dof_prefix.py`
  - **Files**: `cases/kaess_2023/run_kaess_verification_ladder.sh`,
    `tests/regression/test_kaess_real_dof_prefix.py`,
    `jax_fem_am/verification/checkpoints.py`,
    `cases/kaess_2023/analysis/convergence.py`

- [ ] T024 [US3] 建立 `1×30 µm` 最小完整 CPU reference — 覆盖全层扫描、recoat、cooling、release 和三阶段检查点，以 MKL/OMP=1 重复两次。
  - **Acceptance**: 无未来层贡献；两次 active-domain 场、事件、能量和 release QoI 通过批准重复门。
  - **Verify**: `python -m pytest -q tests/regression/test_kaess_1layer_release.py`
  - **Files**: `cases/kaess_2023/run_kaess_verification_ladder.sh`,
    `tests/regression/test_kaess_1layer_release.py`,
    `cases/kaess_2023/analysis/convergence.py`

- [ ] T025 [US3] 完成并冻结 CPU calculation-verification suite — 对 1层执行时间/路径/网格和 solver-tolerance 敏感性；以缩减空间域三层 case 覆盖旋转、recoat、再熔化、历史、冷却和 release；kernel、small-domain 和三层 mini-cycle 均以 MKL/OMP=1 重复两次。
  - **Acceptance**: 两档最细 QoI 差≤2%，收紧 solver tolerance 后≤1%；
    kernel、small-domain、prefix、1层和 multi-layer mini-cycle 各有两次
    immutable run id/hash，事件、历史、能量和 release gate 通过。
  - **Verify**: `python -m pytest -q tests/regression/test_kaess_cpu_verification.py`
  - **Files**: `cases/kaess_2023/run_kaess_verification_ladder.sh`,
    `tests/regression/test_kaess_cpu_verification.py`,
    `cases/kaess_2023/analysis/convergence.py`

- [ ] T026 [US3] 实现 gate aggregator 和 stop rules — launcher 在上游证据缺失/失败时拒绝下游正式运行。
  - **Acceptance**: 测试覆盖 missing、fail、pass、stale-input hash，以及 parity pass 但 energy/convergence fail。
  - **Verify**: `python -m pytest -q tests/integration/test_kaess_gate_chain.py`
  - **Files**: `jax_fem_am/verification/cases.py`,
    `cases/kaess_2023/run_kaess_verification_ladder.sh`,
    `tests/integration/test_kaess_gate_chain.py`

### Checkpoint G3 — CPU Verification Baseline

- PAR028–PAR034 全部通过。
- convergence CSV 和 gate JSON 已人工审核。
- immutable CPU reference-set artifact 已列出全部层级的两次 run id/hash。
- 未通过时禁止任何 GPU/hybrid 正式长算。

## Phase 5 — User Story 4: Hybrid GPU Qualification

- [ ] T027 [US4] 创建冻结的 CPU reference、GPU qualification 和 accelerated formal launcher — 三者共享不可覆盖的物理输入、mask、float64 checkpoint 和 acceptance model。
  - **Acceptance**: `--print-plan` 输出 source checkout/commit、全部输入身份、backend placement 和 gate dependencies。
  - **Verify**: 分别运行三个 launcher 的 `--print-plan`，不得执行求解。
  - **Files**: `cases/kaess_2023/run_kaess_cpu_reference.sh`,
    `cases/kaess_2023/run_kaess_gpu_qualification.sh`,
    `cases/kaess_2023/run_kaess_accelerated_formal.sh`,
    `cases/kaess_2023/inputs/paper-parity-config.yaml`

- [ ] T028 [P] [US5] 扩展正式 run manifest — 捕获 dirty diff、实际 checkout、环境、硬件、线程、输入/输出 hash、逐增量收敛 ledger 和三阶段 gate。
  - **Acceptance**: 逐阶段记录 assembly/global-matrix/linear-solver/state
    residency、host transfer、fallback 和原生 float64 checkpoint；
    build/cooling/release 必须各自提供 typed stage evidence，语义验证器
    逐项复哈希并从原始状态复算 gate；在 T031 接入三阶段 typed
    evidence 前 promotion 必须 fail closed；正式加速结果只接受
    `verdict=pass_promotion_eligible` 且 `promotion_eligible=true` 的
    semantic-validation artifact；失败也生成 forensic manifest。
  - **Verify**: `python -m pytest -q tests/integration/test_kaess_formal_manifest.py`
  - **Files**: `jax_fem_am/verification/provenance.py`,
    `jax_fem_am/verification/run_audit.py`,
    `jax_fem_am/verification/checkpoints.py`,
    `jax_fem_am/verification/backend_qualification.py`,
    `specs/001-kaess-paper-reproduction/contracts/backend-qualification.schema.json`,
    `tests/contract/test_kaess_contracts.py`,
    `tests/integration/test_kaess_formal_manifest.py`

- [ ] T029 [US3] 导入并复核 G3 冻结 CPU small-reference suite — 选择已批准的 kernel、small-domain、real-DOF prefix、1层 build/cooling/release 和三层 mini-cycle run ids，不重复生成另一套“参考”；只有 identity stale 时回退 G3 重跑。
  - **Acceptance**: reference-set artifact、manifest、energy、convergence、
    三阶段 gate 和两次重复身份均复哈希通过；不得在 G4 改配置或阈值。
  - **Verify**: reference-set、manifests、ledger、native checkpoint hashes 和 repeatability report。
  - **Files**: 仅新 run output，不修改源码。

- [ ] T030 [US4] 执行 `hybrid_gpu_assembly_cpu_pardiso` 配对 suite — 对 kernel、small-domain、real-DOF prefix、1层完整流程和三层 mini-cycle，唯一允许差异为 JAX/backend placement，候选同样重复两次。
  - **Acceptance**: commit/dirty diff/物理输入/acceptance model/mask 完全相同；backend parity、energy 和 convergence 独立判定。
  - **Verify**: 生成聚合全部已执行 levels 的 backend-qualification bundle；
    当前 20步手工 probe 只能登记为 diagnostic evidence。
  - **Files**: 仅新 run output 和 qualification artifacts。

- [ ] T031 [US4] 实现并裁决 backend qualification — 在原生 float64 active/printed 域比较温度、应力、eqp、位移、事件、能量、收敛和 release，并顺序比较同线程预算性能。
  - **Acceptance**: scientific/performance verdict 分离；hybrid 名称准确；
    六级 qualification bundle 完整；跨文件 validator 实际复算 artifacts；
    温度/位移分别绑定 `active_node_mask`，应力/eqp 分别绑定
    `active_element_mask`；若先投影到共同比较实体，必须记录投影算法和
    identity/hash，禁止让单一 mask 跨 node/element 实体复用，维度不符
    时 fail closed；`performance_pair` 唯一绑定有序 CPU/candidate run
    ids，wall-time 样本从 rehashed execution-order 区间重算，并与
    manifest completion/resource usage 交叉绑定，线性求解次数从
    checkpoint 重算；energy typed wrapper 必须逐 run 覆盖且从 ledger
    重算闭合；build/cooling/release typed stage evidence 必须逐阶段、
    逐 run 覆盖并从原始状态复算，未接入时 promotion 必须 fail closed；
    两个性能样本满足批准的 speedup 与迭代增幅门。
  - **Verify**: `python -m pytest -q tests/regression/test_kaess_backend_qualification.py`
  - **Files**: `cases/kaess_2023/analysis/compare_backends.py`,
    `jax_fem_am/verification/backend_qualification.py`,
    `tests/regression/test_kaess_backend_qualification.py`,
    `specs/001-kaess-paper-reproduction/contracts/backend-qualification.schema.json`,
    `specs/001-kaess-paper-reproduction/contracts/energy-audit-evidence.schema.json`

### Checkpoint G4 — Accelerated Backend Qualification

- PAR035–PAR044 全部具有明确 verdict。
- 明确选择 immutable CPU reference set；正式 accelerated backend 必须
  同时 `numerically_qualified=true`、`performance_qualified=true`、
  `promotion_eligible=true`。
- 未通过时该 backend 只允许诊断；完整十层 CPU 不作为通过前置。

## Phase 6 — User Story 3: Accelerated Scale Bridge and Paper Comparison

- [ ] T032 [P] [US3] 实现冻结路径 QoI 和论文误差提取 — 提取 Figure 8 `σx`、Figure 9 bending，计算标量误差、NRMSE、峰谷和过零深度。
  - **Acceptance**: 单位/插值与 node/element mask 或共同实体投影显式；
    同一 checkpoint 结果确定；合成样例已知答案通过；输出 typed QoI
    evidence，绑定 run/checkpoint、提取器版本、比较实体、单位、原始
    提取向量及 artifact hash，generic/任意 JSON 不可作为 paper metric
    evidence。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_qoi_extraction.py tests/unit/test_kaess_paper_metrics.py`
  - **Files**: `cases/kaess_2023/analysis/extract_qoi.py`,
    `jax_fem_am/verification/metrics.py`,
    `jax_fem_am/verification/backend_qualification.py`,
    `specs/001-kaess-paper-reproduction/contracts/paper-comparison.schema.json`,
    `tests/contract/test_kaess_contracts.py`,
    `tests/unit/test_kaess_qoi_extraction.py`,
    `tests/unit/test_kaess_paper_metrics.py`

- [ ] T033 [US3] 执行合格后端的 `3×30 → 5×60 µm` 加速桥接 — 每一级独立检查能量、收敛、活动域、阶段状态和多层历史。
  - **Acceptance**: 两级全部 pass；`3×100 µm` medium 不可替代。
  - **Verify**: manifests、gate JSON 和 QoI CSV 人工/自动审核。
  - **Files**: 仅新 run output 和 gate artifacts。

- [ ] T034 [US3] 执行合格后端的 10×30 formal candidate — 完成 build、cooling、release 和审计，不要求先运行完整十层 CPU。
  - **Acceptance**: 三阶段 gate、能量、收敛和 native checkpoint 完整；backend placement 与资格记录相同。
  - **Verify**: manifest、exit code、ledger、checkpoint hashes 和 gate chain。
  - **Files**: 仅新 run output。

- [ ] T035 [US3] 生成加速标准工况 paper-comparison report — 分解 digitization、discretization、CPU-reference repeatability 和 input assumption，不手工抄数。
  - **Acceptance**: JSON 覆盖冻结的八个 Figure 8/9 比较项并通过 schema；
    只接受 T032 typed QoI evidence；语义验证器复哈希 typed evidence，
    从其原始提取量复算 metric/threshold/status，并确认 threshold
    artifact/hash/approval 与 G0 配置、run manifest 一致；verdict 由
    冻结 threshold set 产生，失败/partial 不改阈值；Figure 8/9
    digitized artifacts 必须精确绑定 source manifest 的 evidence id、
    path 与 hash；uncertainty 必须校验单位、区间次序，并从原始分量复算
    `2u_c` 聚合值。
  - **Verify**: `python -m pytest -q tests/integration/test_kaess_paper_report.py`
  - **Files**: `cases/kaess_2023/analysis/compare_paper.py`,
    `cases/kaess_2023/analysis/uncertainty.py`,
    `jax_fem_am/verification/backend_qualification.py`,
    `specs/001-kaess-paper-reproduction/contracts/paper-comparison.schema.json`,
    `tests/contract/test_kaess_contracts.py`,
    `tests/integration/test_kaess_paper_report.py`

### Checkpoint G5 — Accelerated Paper Result

- PAR045–PAR052 全部有 pass/fail/partial 证据。
- 未达到门时如实报告，不修改阈值。
- 只有 `paper-comparison.verdict=pass` 才标记 `G5 promotion_passed`
  并进入正式参数矩阵；`partial`/`fail` 只关闭证据链并允许 diagnostic matrix。

## Phase 7 — User Story 4: Optional Full-GPU R&D Track

- [ ] T036 [P] [US4] 增加真实设备 CPU/GPU kernel 与 small-domain tests — 比较热源、材料、相态、J2 和稀疏算子，并断言实际 GPU device。
  - **Acceptance**: 误差门、设备和后端明确；无 mock device、隐式 float32 或静默 CPU fallback。
  - **Verify**: `python -m pytest -q tests/unit/test_kaess_backend_kernels.py`
  - **Files**: `tests/unit/test_kaess_backend_kernels.py`,
    `jax_fem_am/simulation/acceleration.py`

- [ ] T037 [US4] 建立 GPU sparse linear-solver capability gate — 让 PETSc CUDA options 真正被消费，或资格认证 JAX/AMGX 路线；环境无 CUDA PETSc 时明确 fail/unsupported。
  - **Acceptance**: 不再把 no-op flag 计作能力；小矩阵、真实切片、残差和设备驻留均有证据。
  - **Verify**: `python -m pytest -q tests/contract/test_gpu_sparse_solver_capability.py`
  - **Files**: `jax_fem/solver.py`,
    `jax_fem_am/simulation/acceleration.py`,
    `tests/contract/test_gpu_sparse_solver_capability.py`

- [ ] T038 [US4] 逐级资格认证 `full_gpu` — capability 顺序为 thermal small-domain、J2 mechanics、最小 cooling/release；通过后继续执行 real-DOF prefix、1层、缩减三层和 performance pair，任一级失败即停止。
  - **Acceptance**: 三阶段 local/global sparse/linear solve/state 均为 GPU、
    CPU PARDISO 调用为零、无意外 fallback，场/事件/release 门通过；
    `xla_loop` 要求 `full_loop_xla=true`，host 控制的 PETSc CUDA/AMGX
    要求 transfer telemetry；中间探针未运行阶段记 `not_run`。只有与
    FR-029 相同的六级 suite 各重复两次且性能门通过才可
    `promotion_eligible=true`；否则 verdict 为 diagnostic/fail/unsupported。
  - **Verify**: backend-qualification JSON 和 placement evidence。
  - **Files**: 新 run output、qualification artifact 和必要的 targeted regression test。

- [ ] T039 [P] [US4] 完善设备驻留和性能 instrumentation — 捕获 host-device bytes、wall time、线性求解/装配、RAM、VRAM、利用率和 fallback。
  - **Acceptance**: 报告区分 hybrid/full-GPU、后端迁移、物理改变、JIT 冷启动和资源争用。
  - **Verify**: profile JSON contract/manual review。
  - **Files**: `jax_fem_am/simulation/acceleration.py`,
    `cases/kaess_2023/analysis/compare_backends.py`

### Optional Full-GPU Verdict

- full-GPU 每一级具有 pass/fail/unsupported verdict。
- 本 track 不阻塞已合格 hybrid 的论文复现和参数矩阵。

## Phase 8 — Parameter Matrix and Reproduction Package

- [ ] T040 [US3] 冻结参数矩阵 launcher — 覆盖 30/60 µm、预热、功率、速度和 constant LED。
  - **Acceptance**: 每个点继承同一 protocol、QoI、threshold set 和通过
    G4/G5 的明确 accelerated backend mode；G5 非 `pass` 时 launcher
    拒绝 formal matrix，显式 `diagnostic` 模式除外且不得进入论文结论。
  - **Verify**: `--print-plan` 生成无重复、无遗漏的 case matrix。
  - **Files**: `cases/kaess_2023/run_kaess_paper_matrix.sh`,
    `cases/kaess_2023/inputs/paper-parity-config.yaml`

- [ ] T041 [US3] 执行标定与 held-out 参数点 — 标定点先执行，配置冻结后再打开 held-out。
  - **Acceptance**: 趋势和定量结果分开报告，无数据泄漏；所有点的 backend
    placement 与标准例一致。
  - **Verify**: 生成 `parameter-trends.json` 和 comparison figures。
  - **Files**: 仅新 run output 和 comparison artifacts。

### Checkpoint G6 — Accelerated Matrix and Performance

- PAR053 通过。
- 参数矩阵和性能报告明确写为 hybrid 或 full-GPU，不使用“GPU”泛称。

- [ ] T042 [P] [US5] 组装不可变复现包 — 汇总 source、inputs、manifests、logs、VTU、CSV、scripts 和 hashes。
  - **Acceptance**: 缺任一 required artifact 时 package gate 失败。
  - **Verify**: `python -m pytest -q tests/integration/test_kaess_reproduction_package.py`
  - **Files**: `jax_fem_am/verification/provenance.py`,
    `cases/kaess_2023/analysis/package_reproduction.py`,
    `tests/integration/test_kaess_reproduction_package.py`

- [ ] T043 [P] [US5] 自动生成技术报告和 claim matrix — 报告列出 verified、partial、missing、deviation、uncertainty 和 performance。
  - **Acceptance**: 所有图表来自结构化数据；XRD 仅列为附加算子。
  - **Verify**: report links 和 artifact hashes 检查。
  - **Files**: `cases/kaess_2023/analysis/build_report.py`,
    `docs/kaess_2023_reproduction_report.md`

- [ ] T044 [US5] 执行 clean-directory independent rerun — 按 `quickstart.md` 在新输出目录重建最小 evidence case。
  - **Acceptance**: 输入 hash、关键 QoI 和比较图与发布包一致。
  - **Verify**: 生成 `independent-rerun.json`，由不同会话/审阅者签字。
  - **Files**: 仅新 run output 和 rerun record。

- [ ] T045 [US5] 最终一致性和代码审查 — 在当前未安装 Spec Kit CLI 的
  条件下执行手工 cross-artifact review、全量 tests 和 open-code-review；
  日后安装 CLI 时再补 `/speckit.analyze`。
  - **Acceptance**: 无 FR/SC/task 漏映射，无 P0/P1 review finding；若 `ocr`
    不可用，记录后退到 `code-review-and-quality`。
  - **Verify**: `python -m pytest -q tests/unit tests/contract tests/integration tests/regression`
  - **Files**: `specs/001-kaess-paper-reproduction/checklists/requirements.md`,
    `specs/001-kaess-paper-reproduction/checklists/paper-parity.md`

### Checkpoint G7 — Complete

- 所有选择实施的 tasks 完成并有证据。
- PAR054–PAR062 全部通过。
- 人工批准最终 claim 和报告。

## Requirement and Success-Criteria Coverage

| Requirements | Primary tasks |
|---|---|
| FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007 | T001–T006 |
| FR-008 | T007, T014 |
| FR-009 | T008, T015 |
| FR-010 | T009, T016 |
| FR-011, FR-012 | T010, T017 |
| FR-013, FR-014, FR-015 | T011, T012, T018, T019 |
| FR-016, FR-017 | T013, T020 |
| FR-018 | T007–T026 |
| FR-019, FR-020 | T022–T035 |
| FR-021, FR-022, FR-023, FR-024 | T023–T031 |
| FR-025 | T032–T035 |
| FR-026, FR-027 | T040–T041 |
| FR-028, FR-029, FR-030, FR-031, FR-032, FR-033 | T027–T031 |
| FR-034, FR-035, FR-036 | T005, T028, T042 |
| FR-037 | T043 |
| FR-038, FR-039 | T043, T045 |
| FR-040 | T021, T026, T028–T045 |
| FR-041, FR-042, FR-043, FR-044, FR-045 | T005, T023, T027–T031, T036–T039 |

| Success criteria | Primary tasks |
|---|---|
| SC-001 | T001–T006 |
| SC-002 | T008, T015 |
| SC-003 | T011, T018, T022 |
| SC-004 | T009, T016 |
| SC-005 | T022–T025 |
| SC-006 | T025 |
| SC-007 | T023–T024, T029 |
| SC-008, SC-009, SC-010 | T032–T035 |
| SC-011, SC-012 | T030–T031, T036–T038 |
| SC-013 | T042, T044 |
| SC-014 | T001, T004, T043, T045 |
| SC-015, SC-016, SC-017, SC-018 | T005, T023, T028–T031, T036–T039 |

## Dependencies and Execution Order

```text
T001–T006
  → G0 approval
  → T007–T013 failing tests
  → T014–T021 P0 implementation
  → G1/G2
  → T022–T026 CPU small-reference ladder
  → G3
  → T027–T031 hybrid qualification
  → G4
  → T032–T035 accelerated bridge and paper comparison
  → G5
  → T040–T041 accelerated matrix
  → G6
  → T042–T045 package, report, independent rerun
  → G7

Optional parallel track after G3:
  T036–T039 full-GPU R&D; failure does not block qualified hybrid
```

## Parallel Execution Notes

- T001/T002/T003/T005 可并行；T004 依赖 T002 的数据格式。
- T007/T008/T009/T010/T011/T012/T013 可并行编写失败测试。
- T014/T015/T018 可在接口冻结后并行；T016 必须先于 T017。
- T019 可能与 T018 共享材料接口，需先约定 contract。
- T032 的实现可与 G4 资格收尾并行；T033/T034 必须依次通过，T035 依赖
  T032–T034。
- T036 可在 G3 后并行开展；T038 依赖 T036/T037，但不阻塞 T033/T034
  的已合格 hybrid 路线。
- 同一台机器不并行启动 CPU/PARDISO 和 GPU/hybrid 长算，避免 CPU 与内存
  带宽争用。

## Implementation Discipline

- 每个实现 task 都必须先看到对应测试因预期原因失败。
- 每完成一个逻辑 slice，运行目标测试和相关回归，再形成独立 commit。
- 不将格式化、无关重构、论文物理、求解器性能和 GPU 工作混入同一 commit。
- 任何 Open Question 引发范围变化时，先更新 `spec.md`，再更新本文件。
