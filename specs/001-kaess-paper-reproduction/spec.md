# Feature Specification: Kaess 2023 论文级数值复现

**Feature Directory**: `001-kaess-paper-reproduction`

**Working Branch**: `codex/r3-optimization`

**Created**: 2026-07-23

**Status**: Draft — CPU-small/GPU-formal workflow approved; solver
implementation remains blocked by the other Review Gate items

**Input**: 按照 Spec Kit 标准，为 Kaess et al. (2023) 论文级复现建立可执行、可审计的开发规格。

## Objective

建立一套可追溯、可重复、带分级验证门禁的 Kaess 2023 数值复现流程，能够：

1. 从论文公开信息重建标准工况；
2. 区分论文事实、数字化数据、作者输入、推断和项目假设；
3. 用 CPU float64 小尺度算例建立物理、数值和重复性参考；
4. 对 GPU 或 CPU+GPU 后端进行同配置数值等价和性能资格测试；
5. 资格通过后以加速后端完成中大尺度、十层标准例和论文参数矩阵；
6. 对论文中的残余应力和切割后翘曲曲线进行预注册误差比较；
7. 输出可由第三方复跑和审查的证据包，并量化求解器加速收益。

默认科学声明为：

> 基于公开信息的独立 code-to-code 数值复现。

除非取得实验原始数据并建立独立验证协议，否则不得将结果称为
“实验验证”或“已证明模型预测了真实制造过程”。

## User Scenarios & Testing

### User Story 1 — 建立可审计的复现协议（Priority: P1）

作为研究人员，我希望每一个模型输入和比较指标都有明确来源，以便
审稿人能够区分论文事实、实现选择和未知假设。

**Why this priority**: 如果输入来源和科学声明没有冻结，后续数值吻合
可能来自事后调参，不能构成可信复现。

**Independent Test**: 不运行求解器，仅审查来源矩阵；每个高影响输入
都有来源类别、数值、单位、适用阶段和不确定性说明。

**Acceptance Scenarios**

1. **Given** 论文明确给出某参数，**When** 录入复现协议，**Then**
   参数标记为 `paper_text` 或 `paper_table`，并关联页码、表格或公式。
2. **Given** 参数来自论文图像数字化，**When** 录入参考数据，**Then**
   参数标记为 `figure_digitized`，并记录读图误差。
3. **Given** 论文未公开高影响输入，**When** 建立正式配置，**Then**
   输入标记为 `assumption` 或 `inferred`，并生成敏感性分析要求。
4. **Given** 未获得作者完整输入文件，**When** 生成技术报告，**Then**
   报告只声明“公开信息独立数值复现”。

### User Story 2 — 通过 P0 物理一致性门禁（Priority: P1）

作为求解器开发者，我希望在正式长算前逐项验证关键物理实现，避免
论文差异被求解器参数或经验系数掩盖。

**Why this priority**: 底面约束、热源、元素激活、表面散热、材料模型
和 release 均会直接改变残余应力和翘曲。

**Independent Test**: 使用材料点、单元、patch、小型活动域、单道和
单层测试证明全部 P0 物理要求，无需运行完整十层模型。

**Acceptance Scenarios**

1. **Given** 基板底面，**When** 应用构建阶段边界，**Then** 全部底面
   节点仅约束 `uz`，并以最少 `x/y` 锚点消除刚体模态。
2. **Given** 论文半球三维高斯热源，**When** 对热源体积积分，**Then**
   积分功率等于吸收功率，且不调整吸收率补偿形状错误。
3. **Given** 尚未激活的单元，**When** 装配热学和力学系统，**Then**
   这些单元不贡献热容、导热、刚度、内力或待求自由度。
4. **Given** 新层激活，**When** 更新热边界，**Then** 对流和辐射施加
   在当前活动域真实暴露顶面。
5. **Given** 冷却阶段开始，**When** 更新环境，**Then** 底部温度和
   对流/辐射环境均执行冻结的室温冷却协议。
6. **Given** 熔化、凝固或再熔化，**When** 更新材料历史，**Then**
   不发生无来源的应力参考态或塑性历史重置。
7. **Given** 支撑切割区域，**When** 执行 release，**Then** 使用
   可视化、可哈希的精确单元集合，而不是未验证的宽泛 box。

### User Story 3 — 建立 CPU 小尺度验证基线（Priority: P1）

作为论文复现者，我希望用确定、可重跑的 CPU float64 小尺度算例证明
模型公式、离散、收敛和 build/cooling/release 全流程正确，而不把两次
完整十层 CPU 长算设为 GPU 之前的硬前置。

**Why this priority**: release 翘曲只有在构建和冷却状态已经通过门禁
后才具有解释价值。

**Independent Test**: 固定代码、输入、线程和环境，运行材料点/小网格、
真实自由度短前缀和最小完整 build/cooling/release case；重复运行通过
门禁，并生成相同输入哈希和完整审计清单。

**Acceptance Scenarios**

1. **Given** 单元和 patch 门禁通过，**When** 建立 CPU 基线，**Then**
   基线至少覆盖小活动域、真实全网格短前缀和 `1×30 µm` 的
   build/cooling/release，并以缩减空间域的三层 mini-cycle 覆盖层间
   旋转、recoat、再熔化、历史累积和 release；不要求完整全网格
   `3×30 µm` CPU 长算。
2. **Given** CPU 最小完整 case 进入 release，**When** 执行切割，
   **Then** 系统先独立验收构建后和冷却后温度、力平衡及残余应力。
3. **Given** 加速后端通过 CPU 小尺度等价门，**When** 执行论文标准例，
   **Then** 至少
   输出 Figure 8 的 `σx` 深度曲线和 Figure 9 的弯曲曲线、最大前端
   位移及位移方向。
4. **Given** 某论文输入为假设，**When** 报告结果，**Then** 同时给出
   该假设的敏感性或不确定性范围。
5. **Given** 结果未达到预注册门，**When** 分析失败，**Then** 记录
   偏差原因，不得事后修改误差门或用无来源参数强制拟合。

### User Story 4 — GPU 资格认证与正式加速复现（Priority: P1）

作为性能工程师，我希望在不削弱科学可信度的前提下，让通过资格的
GPU 加速后端承担中大尺度和论文参数矩阵，并用同配置证据展示求解器优势。

**Independent Test**: 使用同一提交、配置、float64 精度、检查点和
验收模型，逐级比较材料核、小网格、真实自由度短前缀、最小
build/cooling/release 及缩减三层 history/release mini-cycle 的 CPU/GPU
输出；通过后再运行 GPU 的 `3×30 → 5×60 → 10×30 µm` 梯度。

**Acceptance Scenarios**

1. **Given** CPU 小尺度基线未通过，**When** 请求正式 GPU 长算，
   **Then** launcher 必须拒绝执行。
2. **Given** `hybrid_gpu_assembly_cpu_pardiso` 通过全部小尺度等价门，
   **When** 执行标准例和参数矩阵，**Then** 它可登记为正式加速后端，
   无需先完成完整十层 CPU 运行。
3. **Given** 热学和装配使用 GPU 但线性解仍为 CPU PARDISO，
   **When** 记录后端，**Then** 必须标记为 hybrid，不得标记为
   `full_gpu` 或“全链路 GPU”。
4. **Given** 候选 `full_gpu`，**When** 请求该标签，**Then** 热学、
   力学装配、线性解和状态驻留均须有 GPU 证据，且无非预期 CPU fallback。
5. **Given** GPU 力学/release 超过预注册差异门，**When** 生成报告，
   **Then** 该后端标记为未通过，保留 CPU 小尺度证据和失败结果，不得
   通过修改物理参数使其通过。
6. **Given** 加速后端资格通过，**When** 运行正式矩阵，**Then** 每个
   结果保留后端模式、各阶段设备、精度、驱动、输入哈希、冷热启动时间、
   迭代数、RAM/VRAM 和意外回退记录。

### User Story 5 — 第三方可复跑的证据包（Priority: P2）

作为审稿人或合作者，我希望从冻结配置重新执行模型并生成相同关键
结果，以便独立检查复现结论。

**Independent Test**: 在新输出目录使用发布命令和锁定环境重跑最小
证据案例，并重新生成清单、关键 CSV 和比较图。

**Acceptance Scenarios**

1. **Given** 一个正式结果，**When** 检查证据包，**Then** 能找到代码
   提交、环境、硬件、命令、输入哈希、日志、检查点、原始场、QoI CSV
   和绘图脚本。
2. **Given** 运行中断，**When** 再次运行，**Then** 不覆盖先前 CPU
   reference、已接受加速正式结果或修改其清单。
3. **Given** manifest 或 response gate 通过，**When** 判断科学结果，
   **Then** 仍需单独检查物理、收敛和论文比较门。

## Edge Cases

- 作者未提供完整材料表、锚点、扫描起点、冷却时长或切割单元集。
- 数字化曲线误差大于预计模型差异。
- 当前活动顶面在完整网格拓扑中为内部面，静态外表面算法无法识别。
- 一步没有活动单元，或激光位于层/单元边界。
- 熔化—凝固—再熔化导致历史变量重入。
- 最小锚点选择不当导致刚体模态或局部约束应力。
- release 集合为空、越界、包含零件本体或不符合论文 Figure 7。
- 求解器 fallback 接受增量，但独立力平衡或位移修正门失败。
- 固定线程重复计算仍出现显著 release 位移差异。
- GPU 内存不足、精度降级、算子回退 CPU 或事件顺序不一致。
- 失败重跑覆盖已冻结的 CPU 验证基线或已接受的 GPU 正式结果。
- 把 JAX 默认设备为 GPU 等同于稀疏线性解和全循环均驻留 GPU。
- GPU 与 CPU 对照修改了 liquid/mushy 刚度、容差或其他物理输入。
- 用包含未激活/未打印自由度的全域 `u_max` 代替活动域或论文测量路径 QoI。
- 仅比较会降为 float32 的 VTU 输出并宣称内部 float64 等价。
- 同一参数同时用于标定和独立验证。

## Requirements

### Scope and Provenance

- **FR-001** 系统 MUST 将本项目标记为 `code_to_code_benchmark`。
- **FR-002** 系统 MUST 明确结果不构成实验验证。
- **FR-003** 每个科学输入 MUST 归类为 `paper_text`、`paper_table`、
  `figure_digitized`、`abaqus_semantics`、`author_artifact`、`inferred`
  或 `assumption`。
- **FR-004** 每个推断或假设输入 MUST 记录理由、影响和敏感性要求。
- **FR-005** 正式运行前 MUST 冻结 QoI、路径、坐标系、插值、误差定义
  和验收阈值；阈值集 MUST 有版本、批准者、批准记录和内容 SHA-256，
  并由语义验证器绑定 G0 配置与当前 run manifest。
- **FR-006** `3×100 µm` medium 结果 MUST NOT 标记为论文正式复现。
- **FR-007** 作者原始文件若取得，MUST 保存原文件哈希并单独审批声明升级。

### P0 Physics Parity

- **FR-008** 构建底面 MUST 实现全底面 `uz=0` 加最小 `x/y` 刚体约束，
  除非作者输入证明采用其他形式。
- **FR-009** 热源 MUST 实现论文半球三维高斯分布并通过吸收功率积分门。
- **FR-010** 未激活单元 MUST 对活动域方程贡献严格为零。
- **FR-011** 系统 MUST 随激活状态动态重建真实暴露热表面。
- **FR-012** 冷却阶段 MUST 使用冻结、可审计的底温和环境温度时间表。
- **FR-013** 材料模型 MUST 覆盖所需温变固体/粉末热物性、潜热、
  热膨胀、弹性和塑性数据及插值语义。
- **FR-014** 系统 MUST NOT 同时使用论文高温曲线和无来源二次刚度缩放。
- **FR-015** 相变历史处理 MUST 有来源或明确假设，并通过循环材料点测试。
- **FR-016** release MUST 使用精确、可视化、可哈希的切割单元集合。
- **FR-017** release 后 MUST 仅保留必要的刚体约束，并审计锚点敏感性。

### Verification and Formal Execution

- **FR-018** 系统 MUST 提供材料点、单元、patch、热源积分、活动域、
  热边界和 release 自动验证。
- **FR-019** CPU 验证 MUST 依次覆盖材料点/单元、小活动域、真实全网格
  短前缀、最小完整 `1×30 µm` build/cooling/release 和缩减空间域三层
  history/release mini-cycle；加速正式梯度 MUST 按
  `3×30 → 5×60 → 10×30 µm` 执行。
- **FR-020** CPU 小尺度、后端资格或加速桥接任一上游门失败时，十层结果
  MUST NOT 登记为正式结果。
- **FR-021** 科学 CPU reference MUST 使用 float64、固定输入和环境，并以
  `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 或经证明等价的单线程直接解
  重复至少两次；多线程 CPU 仅作为独立性能 control。
- **FR-022** 每个接受的力学增量 MUST 保存收敛判据、力残差和位移修正。
- **FR-023** fallback 接受 MUST NOT 替代独立力平衡和敏感性检查。
- **FR-024** 正式运行 MUST 分别保存构建、冷却和 release 检查点。
- **FR-025** 正式比较 MUST 包含 Figure 8 `σx` 深度曲线和 Figure 9
  翘曲曲线、峰值及方向。
- **FR-026** 标准工况通过后，参数矩阵 MUST 覆盖层厚、预热、功率、
  速度和恒定线能量组。
- **FR-027** 标定与留出工况 MUST 在参数矩阵执行前登记。

### CPU/GPU Qualification

- **FR-028** CPU float64 小尺度结果 MUST 是数值资格参考，但完整十层 CPU
  运行 MUST NOT 是合格 GPU 正式执行的强制前置。
- **FR-029** 每个加速模式 MUST 依次通过核、小网格、真实全网格短前缀、
  `1×30 µm` build/cooling/release 和缩减空间域三层 history/release
  mini-cycle 等价测试；完整全网格 `3×30 µm` CPU 对照不是强制前置。
- **FR-030** 后端资格 MUST 在相同 active/printed mask 和冻结论文路径上
  比较温度、相态、能量、激活事件、应力、eqp、位移、收敛判据、线性
  解次数和 build/cooling/release 检查点；未激活自由度只用于零贡献/
  漂移审计，不得进入正式位移 QoI。
- **FR-031** 通过 FR-029/FR-030 只获得 `numerically_qualified`；
  只有同配置性能门也通过并登记 `performance_qualified=true`、
  `promotion_eligible=true` 的模式 MAY 作为正式加速后端运行
  `3×30 → 5×60 → 10×30 µm`、论文比较和参数矩阵。科学等价但未达到
  性能门的模式只允许 diagnostic；某模式失败 MUST 只阻断该模式，不得
  抹除 CPU reference 或另一已合格模式。
- **FR-032** 系统 MUST NOT 为平台一致而修改 liquid/mushy 刚度、材料、
  边界、历史、容差或其他物理/验收参数；消融配置只能登记为
  `diagnostic`。
- **FR-033** GPU 结果 MUST 记录设备、精度、驱动、运行时、每个阶段的
  local assembly/global matrix/linear solver 后端、host-device transfer、
  预期 CPU 操作和意外回退。
- **FR-041** `hybrid_gpu_assembly_cpu_pardiso` MUST 明确记录 GPU JAX
  局部装配与 CPU global sparse matrix/PARDISO；它 MAY 成为正式加速
  后端，但 MUST NOT 标记为 `full_gpu`。
- **FR-042** 正式 `full_gpu` MUST 证明热学、力学、release 的局部装配、
  全局稀疏运算、线性求解和主要状态均驻留 GPU，CPU PARDISO 调用为零，
  且无非预期 CPU fallback。编排可为 `xla_loop` 或调用 PETSc CUDA/AMGX
  的 `host_python`；只有前者要求 `full_loop_xla=true`。逐级能力探针可将
  未执行阶段记为 `not_run`，但不得 promotion 为正式 `full_gpu`。
- **FR-043** 性能资格 MUST 使用与候选后端完全相同的物理、精度、输入、
  source checkout/commit/dirty diff、acceptance model 和 CPU 线程预算，
  顺序运行，分别报告冷启动、稳态、分阶段 wall time、线性解次数、RAM
  和 VRAM。
- **FR-044** CPU/GPU 科学等价 MUST 使用求解器内部或无损保存的 float64
  checkpoint；当前写出为 float32 的 VTU MAY 用于可视化和辅助复核，但
  MUST NOT 是 float64 后端资格的唯一证据。
- **FR-045** 正式加速 promotion MUST 通过跨文件语义验证器：实际打开并
  复算 qualification、run manifest、profiler、mask 和 checkpoint 的
  SHA-256，确认当前 run id 在 candidate 集合内、资格层级完整、commit/
  输入/acceptance model 相同、数组 dtype 为 float64、阶段 placement 与
  mode 一致，且不存在重复冲突 gate。正式 launcher MUST 同时要求
  validation `verdict=pass_promotion_eligible` 和
  `promotion_eligible=true`；验证器还 MUST 核对 dirty diff、精度、CPU
  线程预算、环境/硬件身份和顺序执行证据。validation 引用 G4
  qualification candidate manifest，不得反向引用待接受的 formal
  manifest 形成哈希循环。

### Reporting and Reproducibility

- **FR-034** 每次正式运行 MUST 生成无歧义的 run manifest。
- **FR-035** manifest MUST 包含代码、dirty 状态、环境、命令、线程、
  硬件、输入和输出哈希。
- **FR-036** 证据包 MUST 包含日志、能量账、收敛记录、检查点、场数据、
  QoI CSV、数字化参考和绘图脚本。
- **FR-037** 报告 MUST 分别呈现已验证、部分覆盖、未验证、偏差和不确定性。
- **FR-038** XRD MUST NOT 作为 Kaess 论文复现成功的必要或充分条件。
- **FR-039** 正常退出或 manifest 完整 MUST NOT 单独证明科学复现通过。
- **FR-040** 未达门限的结果 MUST 保存为失败证据，不得隐藏或覆盖。

## Key Entities

- **Reproduction Protocol**: 科学范围、标准工况、QoI 和验收规则。
- **Evidence Record**: 输入的值、单位、来源类别、出处和不确定性。
- **Assumption Register**: 未公开输入、影响、敏感性范围和处理决定。
- **Physics Gate**: 独立物理能力及其测试、阈值和结果。
- **Run Manifest**: 一次运行的代码、环境、命令、硬件及输入/输出身份。
- **CPU Verification Baseline**: 覆盖全部物理转变、重复性和离散检查的
  CPU float64 小尺度参考集合，不要求完整十层。
- **Accelerated Backend Qualification**: 一个明确设备放置模式相对 CPU
  reference 的场、事件、收敛、响应和性能差异。
- **Formal Accelerated Run**: 由通过资格的 hybrid 或 full-GPU 模式完成
  的 3层、5×60、十层标准例或参数矩阵运行。
- **Quantity of Interest**: 温度、能量、相态、应力和翘曲响应。
- **Deviation Record**: 论文与实现之间未消除的差异及影响。
- **Validation Matrix**: 论文声明到证据、测试层级和输出的映射。

## Success Criteria

- **SC-001** 100% 高影响输入具有来源分类，无未登记高影响默认值。
- **SC-002** 热源积分相对误差不超过 0.5%。
- **SC-003** 潜热积分误差不超过 0.5%，能量账闭合误差不超过 1%。
- **SC-004** 小型活动域与物理删除参考解的相对差不超过 `1e-8`。
- **SC-005** 时间步或路径加密后关键 QoI 变化不超过 2%。
- **SC-006** 收紧求解器容差后正式 QoI 变化不超过 1%。
- **SC-007** 两次单线程 CPU reference 的最大 release 位移差不超过
  `max(0.1 µm, 1%)`，方向、事件和相态一致。
- **SC-008** 150°C 标准工况相对约 `14.0 ± 0.3 µm` 数字化前端翘曲，
  同时满足相对误差≤10%和绝对误差≤`max(1.0 µm, 2u_c)`。
- **SC-009** 翘曲曲线 NRMSE 不超过
  `max(10%, 2×数字化不确定度)`。
- **SC-010** Figure 8 拉压顺序一致；峰谷幅值误差不超过 15%，
  过零深度误差不超过一个局部单元高度。
- **SC-011** CPU/GPU 配对的温度场相对 `L2` 和温度 QoI 差均不超过
  0.1%，激活事件和活动单元身份完全一致。
- **SC-012** GPU/混合后端应力与 eqp 场相对 `L2` 不超过 1%，release
  翘曲曲线相对 `L2` 不超过 2%，最大翘曲差不超过
  `max(0.5 µm, 2%)`。
- **SC-013** 第三方可用冻结命令生成相同输入哈希、QoI 和比较图。
- **SC-014** 所有公开结果都使用 `code-to-code numerical reproduction`
  声明，不越界为实验验证。
- **SC-015** 每个正式后端标签均与 manifest 的阶段级设备放置一致；
  `full_gpu` 记录中 CPU PARDISO 调用数和意外 CPU fallback 数均为零。
- **SC-016** 首版性能资格要求两个代表性同配置样本的加速后端端到端
  wall-time 中位数均至少快 `1.20×`，且线性解次数相对 CPU performance
  control 增幅不超过 10%；该阈值在 Review Gate 审批前仅为提案。
- **SC-017** 后端等价报告引用原生 float64 checkpoint，并明确记录比较
  mask；不得以 VTU float32 差异作为唯一数值资格证据。
- **SC-018** 任何 accepted accelerated formal manifest 都包含
  `verdict=pass_promotion_eligible` 且 `promotion_eligible=true` 的
  `backend_qualification_validation` artifact；伪造 mode、kernel-only
  资格、错误 hash/dtype、性能协议身份或未包含当前 run id 的反例均被
  拒绝。

## Assumptions

1. 当前依据为 Kaess 2023 论文、现有 Figure 8/9 数字化数据和仓库元数据。
2. 作者完整 Abaqus 输入、子程序、完整材料表、锚点和切割单元集尚未取得。
3. 标准工况为 10×30 µm、150°C、250 W、850 mm/s、50 µm 光斑半径、
   100 µm hatch。
4. 模型采用弱耦合热—力流程。
5. CPU float64 单线程小尺度结果为数值资格参考；通过资格的加速后端可
   产生论文正式结果，完整十层 CPU 不是前置条件。
6. 当前 medium/full-height 算例只承担回归和故障扫描。
7. 本规格的误差阈值是项目预注册工程标准，不是论文作者的误差声明。

## Boundaries

### Always

- 先更新规格和来源矩阵，再改变正式物理模型。
- 每项 P0 修改采用失败测试、最小修复、回归验证。
- 长算前完成相应小尺度门禁。
- 固定并记录代码、环境、输入和命令。
- 分别保存构建、冷却和 release 状态。
- 以 CPU float64 小尺度集合为数值参考，以通过资格的加速后端执行正式长算。
- 保存失败证据和偏差。

### Ask First

- 修改预注册误差阈值。
- 引入无来源材料参数、历史重置或经验缩放。
- 改变比较路径、坐标系、插值或 QoI。
- 将作者新文件并入正式基准或升级声明等级。
- 新增依赖、改变网格拓扑或更换本构模型。
- 将未通过资格的 GPU 用于正式矩阵。

### Never

- 把 `3×100 µm` medium 算例描述为正式复现。
- 用吸收率、硬化、阻尼或容差补偿已知物理错误。
- 在论文曲线上调参后把同一曲线称为独立验证。
- 在 CPU 小尺度 reference 通过前把 GPU 定义为参考真值。
- 把 `hybrid_gpu_assembly_cpu_pardiso` 描述为 `full_gpu`。
- 仅凭退出码或 manifest 宣称科学验证成功。
- 隐藏未公开输入、失败门禁或平台差异。
- 把 code-to-code 对比描述成实验验证。

## Out of Scope

- 使用真实打印测量完成实验验证。
- 证明 JAX-FEM 与 Abaqus 所有自由度逐位一致。
- 在未通过对应小尺度等价门时强制使用 GPU 力学、GPU release 或
  `full_gpu` 线性解。
- 对与 Kaess 无关的通用求解器做全面重构。
- 以 XRD pipeline 作为本文主要验收指标。
- 未获得作者输入时宣称 Abaqus 输入文件完全等价。

## Dependencies

- Kaess 2023 论文 PDF 和在线公开版本。
- Figure 8/9 数字化数据和读图不确定度。
- 现有 Kaess 网格、路径、材料配置和求解器。
- 可重复 CPU float64 JAX 环境。
- GPU 资格所需固定设备和软件栈。
- 可选：作者 Abaqus 输入、子程序、材料表和原始输出。

## Open Questions

### Blocking Before Formal Accelerated 10-Layer Run

1. 最终目标是公开信息独立复现，还是必须争取作者输入等价复现？
2. 论文具体 `x/y` 防刚体锚点节点如何确定？
3. 冷却时长、底温曲线及环境温度切换规则是什么？
4. 完整扫描起点、方向、路径次序和层间时间是否可取得？
5. release 的精确单元集合是什么？
6. 完整温变塑性数据、热膨胀语义和 Abaqus 插值规则是什么？
7. 作者模型如何处理熔化、凝固和再熔化历史？

### Requires User Approval

8. 是否接受本规格提出的收敛、论文曲线和 CPU/GPU 阈值？
9. 哪些参数工况用于校准，哪些保留为独立验证？
10. 已决定：允许通过 CPU 小尺度等价门的 hybrid/full-GPU 结果作为正式
    论文结果；完整十层 CPU 不作为强制前置。
11. 是否将时间、能耗或成本纳入成功标准？
12. 作者资料无法取得时，哪些假设允许经敏感性包络后继续？

## Review Gate

进入实现前必须确认：

- [ ] 复现声明等级
- [ ] 标准工况和 QoI
- [ ] P0 物理范围
- [ ] 误差与重复性阈值
- [ ] 标定/留出工况划分
- [x] CPU 小尺度 reference、GPU 资格和加速正式执行策略
- [ ] 未公开输入的处理方式
