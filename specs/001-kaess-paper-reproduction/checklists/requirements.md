# Requirements Checklist: Kaess 2023 论文级数值复现

**Purpose**: 在进入实现前检查规格的完整性、清晰度、一致性和可验收性。

**Created**: 2026-07-23

**Feature**: [spec.md](../spec.md)

## Specification Quality

- [x] CHK001 Objective 明确说明了要构建什么、为什么构建以及声明边界。
- [x] CHK002 User stories 按优先级排序，并给出独立测试方法。
- [x] CHK003 Acceptance scenarios 使用 Given/When/Then 表达。
- [x] CHK004 Functional requirements 具有唯一 `FR-*` 标识。
- [x] CHK005 Success criteria 具有可量化 `SC-*` 标识。
- [x] CHK006 Edge cases 包含未知论文输入、历史重入、刚体模态、GPU 差异和覆盖风险。
- [x] CHK007 Assumptions、Dependencies、Out of Scope 和 Boundaries 均已定义。
- [x] CHK008 “公开信息复现”与“作者输入等价复现”已明确区分。
- [x] CHK009 code-to-code 与实验验证边界已明确。
- [x] CHK010 medium regression 与正式 10×30 µm case 已明确分离。

## Requirement Consistency

- [x] CHK011 Constitution、spec、plan 均规定 physics parity 先于性能优化。
- [x] CHK012 Constitution、spec、plan 均规定 CPU 小尺度 reference 先于
  GPU 正式执行，但不要求完整十层 CPU。
- [x] CHK013 验证梯度在所有文档中一致：CPU 覆盖
  `kernel/small-domain/real-DOF-prefix/1×30 build-cooling-release/
  reduced-domain 3-layer mini-cycle`，加速后端执行
  `3×30 → 5×60 → 10×30 µm`。
- [x] CHK014 构建/冷却门禁在 release 之前。
- [x] CHK015 manifest/response gate 不被当作科学充分条件。
- [x] CHK016 现有 Figure 8/9 数字化锚点和不确定度要求已进入规格。
- [x] CHK017 失败结果的保存和不覆盖原则已进入要求。

## Traceability

- [x] CHK018 每个 FR 已映射到至少一个 task。
- [x] CHK019 每个 SC 已映射到一个自动或人工验证输出。
- [x] CHK020 每个 P0 gap 已映射到失败测试、实现任务和回归测试。
- [ ] CHK021 每个正式 QoI 已映射到冻结路径、单位、插值和 artifact。
- [x] CHK022 每个未知作者输入已映射到请求记录或敏感性任务。

## Human Review

- [x] CHK023 用户已批准复现声明等级。
- [x] CHK024 用户已批准 P0 物理范围。
- [x] CHK025 用户已批准收敛和论文比较阈值；阈值集版本、批准者、批准
  artifact 和 SHA-256 已冻结。
- [x] CHK026 用户已批准“CPU 小样本 reference → GPU 资格 → 加速正式
  复现”的总体策略。
- [x] CHK027 用户已批准标定与留出工况划分。
- [x] CHK028 用户已批准外部材料输入的冻结方式。
- [x] CHK029 所有阻塞 Open Questions 均已解决或转为批准的假设。
- [x] CHK030 用户已批准 CPU/GPU 场、release、性能和线性解次数的具体
  数值阈值。

## Spec Kit Analyze Gate

- [ ] CHK031 `spec.md`、`plan.md`、`tasks.md` 无冲突或遗漏。
- [ ] CHK032 任务依赖顺序与 plan 的 phase/checkpoint 一致。
- [ ] CHK033 没有任务通过参数调优绕过 physics gate。
- [ ] CHK034 没有单任务计划修改超过约五个文件。
- [x] CHK035 实现前 Review Gate 已明确解除。
- [ ] CHK036 backend contract 能拒绝 hybrid 冒充 `full_gpu`。
- [ ] CHK037 后端资格必须引用 active/printed mask 和原生 float64
  checkpoint，不能只依赖 VTU。
- [ ] CHK038 跨文件语义验证器及其输出 contract 能拒绝 mode 伪装、
  kernel-only promotion、错误 artifact/threshold hash、错误 dtype、冲突
  gate、不公平性能协议、仅 `pass_not_promotion_eligible` 的记录和不含
  当前 run id 的资格包。

## Notes

- 勾选 `[x]` 表示证据已审阅，不表示相应求解器能力已经实现。
- CHK023–CHK030 是人工门禁，不能由自动测试代替。
- G0 审批记录为 `cases/kaess_2023/inputs/g0-approval.json`；批准时间
  `2026-07-24T03:03:18Z`。
- 任何 requirement 变更先修改 `spec.md`，再更新 plan/tasks。
