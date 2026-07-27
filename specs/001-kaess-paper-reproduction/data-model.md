# Data Model: Kaess 2023 Reproduction Evidence

## Overview

本功能采用文件式数据模型。JSON/YAML 表示结构化身份和门禁，CSV 表示
曲线与收敛序列，VTU 表示场数据。所有正式对象都必须能追溯到
`ReproductionProtocol` 和一次不可歧义的 `RunRecord`。

## Entities

### 1. ReproductionProtocol

冻结本轮复现的科学合同。

| Field | Type | Meaning |
|---|---|---|
| `protocol_id` | string | 不变标识，例如 `kaess-2023-public-v1` |
| `claim_level` | enum | `public_code_to_code` 或 `author_input_equivalent` |
| `paper_doi` | string | `10.3390/ma16062321` |
| `standard_case_id` | string | 标准 10×30 µm 工况 |
| `qoi_ids` | array[string] | 正式比较量 |
| `threshold_set_id` | string | 预注册门限集合 |
| `calibration_case_ids` | array[string] | 允许用于校准的工况 |
| `held_out_case_ids` | array[string] | 只能用于独立检验的工况 |
| `approved_by` | string/null | 人工审批记录 |
| `approved_utc` | datetime/null | 审批时间 |

### 2. EvidenceRecord

表示单个论文输入或实现语义的证据。

| Field | Type | Rules |
|---|---|---|
| `evidence_id` | string | 全局唯一 |
| `quantity` | string | 参数、公式或行为名称 |
| `value` | scalar/object | SI 值优先 |
| `unit` | string/null | 明确单位 |
| `source_class` | enum | 七种 constitution 证据类之一 |
| `source_locator` | string | 页、表、图、公式、文件和行 |
| `uncertainty` | object/null | 类型、值、单位和方法 |
| `impact` | enum | `low`, `medium`, `high`, `critical` |
| `status` | enum | `verified`, `inferred`, `assumed`, `missing` |

### 3. AssumptionRecord

扩展 `EvidenceRecord` 中未公开输入。

| Field | Type | Meaning |
|---|---|---|
| `assumption_id` | string | 唯一 ID |
| `evidence_id` | string | 对应 EvidenceRecord |
| `rationale` | string | 为什么采用该值/语义 |
| `range` | object | 敏感性范围 |
| `affected_qoi_ids` | array[string] | 可能影响的结果 |
| `sensitivity_required` | boolean | 是否阻塞正式声明 |
| `decision` | enum | `open`, `accepted`, `rejected`, `replaced_by_author_data` |

### 4. ReproductionCase

定义一个冻结的计算工况，不等同于一次运行。

| Field | Type | Meaning |
|---|---|---|
| `case_id` | string | 例如 `standard-10x30-t150-p250-v850` |
| `purpose` | enum | `verification`, `calibration`, `held_out`, `regression` |
| `layers` | integer | 层数 |
| `layer_thickness_m` | number | 层厚 |
| `process` | object | 功率、速度、半径、hatch、角度 |
| `thermal_schedule` | object | 预热、recoat、冷却 |
| `mesh_ref` | string | 输入身份 |
| `material_ref` | string | 输入身份 |
| `path_ref` | string | 输入身份 |
| `release_set_ref` | string/null | release cell-set 身份 |

### 5. PhysicsGate

| Field | Type | Meaning |
|---|---|---|
| `gate_id` | string | 例如 `P0-HS-INTEGRAL` |
| `requirement_ids` | array[string] | `FR-*` 追踪 |
| `level` | enum | `analytical`, `unit`, `patch`, `integration`, `regression`, `formal` |
| `metric` | string | 验证量 |
| `threshold` | object | 比较符、值、单位 |
| `evidence_paths` | array[string] | 测试或报告 |
| `status` | enum | `not_run`, `pass`, `fail`, `waived` |
| `waiver` | object/null | 必须经人工审批 |

### 6. RunRecord

一次不可变的执行身份，具体 schema 见
[contracts/run-manifest.schema.json](contracts/run-manifest.schema.json)。

关键字段：

- `run_id`, `case_id`, `claim_level`, `status`;
- git commit、dirty diff hash；
- Python/JAX/JAXlib/MKL/PARDISO/CUDA 版本；
- CPU/GPU、线程、驱动、关键环境变量；
- 命令、开始/结束时间和退出码；
- 所有输入与输出的 SHA-256；
- `build_complete`, `cooling_complete`, `release_complete` 门禁；
- QoI 和 artifact 索引。

正式记录还必须区分实际 source checkout 与外层 launcher 所在目录，避免
把不同 worktree/commit 的 CPU、GPU 产物拼成一条资格链。

### 7. FieldCheckpoint

| Field | Type | Meaning |
|---|---|---|
| `checkpoint_id` | enum | `build_complete`, `cooling_complete`, `release_complete` |
| `run_id` | string | 所属运行 |
| `state_path` | string | VTU/状态文件 |
| `sha256` | string | 内容身份 |
| `temperature_summary` | object | min/max/field norms |
| `phase_summary` | object | active/melted/solid counts |
| `mechanics_summary` | object/null | stress, eqp, residual metrics |
| `gate_status` | enum | `pass`, `fail`, `not_applicable` |

### 8. QuantityOfInterest

| QoI ID | Unit | Stage | Definition |
|---|---|---|---|
| `peak_temperature` | K | build | 活动域最大温度 |
| `melt_volume` | m³ | build | 超过冻结液相阈值的体积 |
| `thermal_energy_closure` | fraction | build/cooling | 输入、焓增和损失闭合 |
| `sigma_x_depth_curve` | MPa vs mm | cooling complete | Figure 7/8 冻结路径 |
| `sigma_x_zero_crossing` | mm | cooling complete | 插值规则冻结 |
| `front_bending_curve` | µm vs mm | release complete | Figure 9 冻结路径 |
| `max_front_bending` | µm | release complete | 同一路径最大值 |
| `release_direction` | enum | release complete | upward/downward |
| `newton_force_ratio` | dimensionless | mechanics step | 双判据力残差 |
| `newton_displacement_ratio` | dimensionless | mechanics step | 位移修正比 |

### 9. BackendQualification

| Field | Type | Meaning |
|---|---|---|
| `qualification_id` | string | 后端和环境身份 |
| `execution_mode` | enum | `hybrid_gpu_assembly_cpu_pardiso`, `gpu_dominant_experimental`, `full_gpu` |
| `cpu_reference_run_ids` | array[string] | 至少两次单线程 CPU reference |
| `candidate_run_ids` | array[string] | 至少两次同配置候选运行 |
| `levels` | array[enum] | `kernel`, `small_domain`, `real_dof_scan_prefix`, `1layer_build_cooling_release`, `reduced_3layer_history_release`, `performance_pair`, `formal_10layer` |
| `level_run_pairs` | object | 每一级至少两次 CPU/candidate run、状态和带哈希证据 |
| `source_identity` | object | 相同 commit/dirty diff/input hashes |
| `comparison_scope` | object | active/printed mask、论文路径和排除域 |
| `checkpoint_precision` | const | `float64` |
| `field_metrics` | array[typed metric] | 温度、应力、eqp、位移场范数 |
| `event_metrics` | array[typed metric] | 激活/相态 digest 或类别匹配 |
| `qoi_metrics` | array[typed metric] | 温度、翘曲曲线、峰值和方向 |
| `convergence_metrics` | array[typed metric] | 增量/fallback digest、线性解次数 |
| `placement_evidence` | object | 编排方式、manifest/profile 哈希、PARDISO/fallback 计数 |
| `performance` | object | CPU/candidate 重复样本、冷/热时间、中位加速、RAM、VRAM、线程 |
| `numerically_qualified` | boolean | 科学等价门是否通过 |
| `performance_qualified` | boolean/null | 同配置性能门是否通过 |
| `promotion_eligible` | boolean | 是否可作为正式 accelerated backend |
| `verdict` | enum | `pass`, `fail`, `diagnostic_only`, `unsupported` |

### 9a. BackendPlacement

| Field | Type | Meaning |
|---|---|---|
| `mode` | enum | `cpu_reference`, `hybrid_gpu_assembly_cpu_pardiso`, `gpu_dominant_experimental`, `full_gpu` |
| `orchestration_backend` | enum | `host_python`, `xla_loop` |
| `thermal` | StagePlacement | 热阶段 |
| `mechanics` | StagePlacement | 力学阶段 |
| `release` | StagePlacement | release 阶段 |
| `host_device_transfers` | object | 次数、字节和证据来源 |
| `unexpected_fallbacks` | array[object] | 非预期回退；正式 `full_gpu` 必须为空 |

`StagePlacement` 分别记录 `local_assembly_backend`、
`global_matrix_backend`、`linear_solver_backend` 和
`state_residency_backend`。`hybrid_gpu_assembly_cpu_pardiso` 的全局矩阵
与线性解必须显式为 CPU；`full_gpu` 四项必须为 GPU。

正式 promotion 还必须产生符合
[contracts/backend-qualification-validation.schema.json](contracts/backend-qualification-validation.schema.json)
的独立语义验证记录。该记录复算 artifact hash、candidate membership、
source/acceptance identity、六级覆盖、checkpoint dtype/shape、mask、
placement、gate、metric 和 performance protocol identity，而不是信任
qualification 文件自报的 `pass`。正式结果只接受
`pass_promotion_eligible`；`pass_not_promotion_eligible` 是有效诊断记录，
不能进入 accelerated formal。记录中的
`qualification_candidate_manifest_artifact` 指 G4 低层 candidate，避免
与最终 formal manifest 的 validation 引用形成哈希循环。

### 10. PaperComparison

具体 schema 见
[contracts/paper-comparison.schema.json](contracts/paper-comparison.schema.json)。

每项比较保存：

- 参考值/曲线和数字化不确定度；
- 模拟值/曲线；
- 绝对、相对、NRMSE 或位置误差；
- 离散、重复、输入和数字化不确定度；
- 预注册阈值；
- `pass`, `fail`, `not_comparable` 判定；
- 不改变阈值的偏差解释。

正式报告固定包含 Figure 8 的拉压顺序、峰值、谷值、过零深度，以及
Figure 9 的曲线 NRMSE、最大翘曲相对/绝对误差和方向。比较项通过
`threshold_metric_id` 绑定唯一阈值，且 `pass`/`partial`/`fail` verdict
必须带通过的语义验证 artifact。阈值集保存版本、批准者、批准记录和
冻结 `paper-parity-config.yaml` 的 SHA-256；验证器将其与当前 run
manifest 输入身份绑定。每个不确定度分量必须记录 applicability、method、
value、unit 和带哈希 evidence，空对象或任意 `junk` 字段不合法。

### 11. DeviationRecord

| Field | Type | Meaning |
|---|---|---|
| `deviation_id` | string | 唯一 ID |
| `paper_requirement` | string | 目标行为 |
| `implemented_behavior` | string | 当前行为 |
| `severity` | enum | `P0`, `P1`, `P2` |
| `affected_qoi_ids` | array[string] | 影响面 |
| `resolution` | enum | `open`, `fixed`, `accepted_assumption`, `cannot_resolve` |
| `evidence_paths` | array[string] | 代码、测试、报告 |

## Relationships

```text
ReproductionProtocol
├── EvidenceRecord ── 0..1 AssumptionRecord
├── ReproductionCase ── 1..* RunRecord
├── PhysicsGate ── 0..* RunRecord
└── ThresholdSet

RunRecord
├── 1..3 FieldCheckpoint
├── 1..* QuantityOfInterest
└── 0..1 PaperComparison

BackendQualification
├── 2..* CPU reference RunRecord
├── 2..* candidate RunRecord
└── 1..* level/run evidence pair
```

## State Transitions

### Reproduction Case

```text
draft
  → source_frozen
  → physics_verified
  → cpu_reference_verified
  → accelerated_backend_qualified
  → accelerated_scale_bridge_passed
  → accelerated_formal_accepted
  → paper_compared
  → published
```

Only a required upstream gate or the currently selected formal backend moves
the case to `blocked`; it returns only after a new versioned input or
implementation resolves the failure. A failed optional backend (for example
the full-GPU R&D branch) moves only that backend branch to `blocked` and does
not invalidate an independently promoted hybrid route.

### Run Record

```text
planned → running → build_complete → cooling_complete → release_complete
                         └──────────────┴──────────────→ failed
release_complete → audited → accepted | rejected
```

Failed and rejected records are immutable evidence. A retry receives a new
`run_id`.

## Validation Rules

1. Formal records require clean or fully captured dirty Git state.
2. Every referenced file requires SHA-256.
3. A formal release checkpoint requires passed build and cooling gates.
4. An accelerated formal record requires accepted CPU small-reference and
   lower-level backend qualification records; a full 10-layer CPU run is not
   required.
5. `claim_level=author_input_equivalent` requires `author_artifact` evidence for
   all critical unknowns.
6. A `waived` physics gate requires approver, rationale and expiry/version.
7. Units are explicit; comparison scripts may not infer units from filenames.
8. Existing run directories are never overwritten.
9. Every CPU/GPU pair requires identical commit, captured dirty diff, physical
   inputs, precision, acceptance model, mask and checkpoint definitions.
10. Scientific backend comparison requires native float64 checkpoints. A VTU
    artifact that casts data to float32 cannot be the sole comparison source.
11. Formal `full_gpu` requires GPU placement in every `StagePlacement`, zero
    CPU PARDISO calls and no unexpected fallback. `xla_loop` requires
    `full_loop_xla=true`; host-controlled PETSc CUDA/AMGX may use
    `host_python` with measured transfers and GPU-resident global operations.
12. CPU scientific references use single-thread MKL/OMP and at least two
    repetitions; CPU multithread results are performance controls only.
13. JSON Schema only validates object shape. Accelerated promotion also requires
    a cross-file semantic validator that rehashes artifacts, opens native
    checkpoints, verifies dtype/shape/mask, checks the current candidate run ID
    and qualification levels, and reconciles observed placement with the
    declared mode.
