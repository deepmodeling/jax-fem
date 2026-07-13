# codex_model_update：面向高水平论文的 v06 模型升级计划

> 状态：执行中
>
> 日期：2026-07-10
>
> 冻结对照基线：`159_local/v05`
>
> 主开发目录：`159_local/v06`
>
> 建议论文主线：Ti-6Al-4V LPBF 部件尺度降阶热—弹塑性残余应力与切割释放变形
>
> 首选投稿方向：*Additive Manufacturing*；若数值方法创新和误差分析足够强，可转向 *Computer Methods in Applied Mechanics and Engineering*

## 1. 执行结论

v05 已经从 v03/v04 的“标量等效塑性历史”升级为积分点塑性应变张量历史，并能把建造态历史传给 release 求解；这是必要升级，但还不足以形成论文级的定量模型。

当前最重要的结论不是“继续增加更多物理”，而是先回答一个边界清楚、可证伪的问题：

> 在不解析熔池和逐道微观组织的前提下，能否建立一个能量可审计、状态一致、误差受控的部件尺度降阶模型，以显著低于轨迹级模型的成本，定量预测 Ti-6Al-4V LPBF 的残余弹性应变和切割释放变形？

本篇论文只保留这条主线。高温黏塑性、冶金相变、真实移动熔池热源均采用“证据触发”的可选升级，不同时作为第一篇论文的必选项。

建议冻结 v05 作为可复现实验基线，在 `159_local/v06` 中建立干净的论文模型。v06 不再继续依赖多层 monkey patch，而是显式管理热状态、力学内部变量、激活/重熔和 release 生命周期。

### 1.1 当前执行快照（2026-07-10）

当前 v06 已完成第一轮可执行骨架，但仍处于“预验证”阶段：

- 已建立唯一兼容入口 `v06/driver.py`，当前明确复用 v03 时间循环和 v04 性能层，不依赖 v05 runtime；这只是迁移桥梁，尚不是 AD-1 所述的最终独立架构。
- 已实现纯 JAX 的完整张量 J2 状态、饱和硬化跨越修复、重熔时 `eqp/eps_p` 同步清零，以及 release 状态采用与提交；新增 `eps_ref` 出生/重熔机械参考构型，并由 reference-reset event 强制 mechanics update。
- 已实现 TET4 网格质量、体积加权统计、质量过滤诊断和输出有限值审计；full91 最差单元已定位，1068 MPa 峰值被降级为网格异常证据。
- 已将热账本接入最终 v04 solver 外层：每个热步重新组装未施加 Dirichlet 行替换的残差，记录储能、实际沉积热、体积/表面交换、固定温度边界交换、自由残差、状态重置与温度不变量。它审计当前离散弱式，不等同严格焓守恒。
- 已实现精确凸 `gauge box ∩ TET4` 交体积、覆盖率门禁、VTU 弹性应变张量输出/读取及 microstrain 投影；同时确认 Strantza 使用的菱形衍射体积与当前 box 算子不等价，因此正式实验 claim 被阻断。
- 已建立零输入不变量 smoke、非零宏观直接固化制造 smoke、全瞬态审计、XRD operator smoke、响应门禁和交叉哈希 provenance；完整运行 claim 固定为 `numerical_smoke_only`，失败/过期/篡改运行自动降级为 forensic manifest。
- 已在 JIT 前增加材料表物理域检查，并修复高温状态每步重复强制 mechanics、release 状态形状静默跳过和 VTU 张量静默缺失问题。

当前最优先的未完成项是：热离散单调性/焓状态、积分热膨胀、可恢复 checkpoint、菱形 XRD 算子与真实配准/原始数据，以及冻结参数后的 held-out 比较。详细运行边界见 [v06 README](README.md)。

### 1.2 当前可复核证据

| 证据 | 当前结果 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `v06_smoke_runner_09` | 3 步热账本、审计、XRD、响应门禁、manifest 全部 complete | 零输入不变量与证据链接通 | LPBF 热循环或实验准确率 |
| `v06_nonzero_smoke_04` | 温度、受约束/释放应力、释放位移、沉积能和 XRD 均非零；response gate 通过 | 非零热—力—测量链没有退化为全零 | 真实熔池、材料标定或几何验证 |
| `v06_nonzero_smoke_02` | 粗网格局部热源产生约 0.33 K 节点下冲并被拒绝 | 温度门禁能捕获非单调离散结果 | 尚未给出质量集总/细网格修复 |
| `full91_mesh_audit.json` | 197,266 TET4，无倒置；最差质量约 `8.89e-5`，与 1068 MPa 峰重合 | 当前全局峰值不可信 | 重网格后的收敛场 |

以上路径均位于 `/home/user/work/159/output/`；它们是当前 checkout 的本地证据，不是论文归档 DOI。

## 2. 当前基线与硬性问题

### 2.1 v03-v05 已完成的能力

| 版本 | 已完成能力 | 论文级缺口 |
|---|---|---|
| v03 | 热传导、单向热—力耦合、单元激活、温度相关材料、J2 塑性、弹性地基、冷却与 release | 源码自身仍将 J2 描述为简化模型；塑性历史以 eqp 为主，状态闭环不完整 |
| v04 | XLA/装配性能层、profiling、benchmark harness、flash 快扫 | wrapper 与 dry-run 证明不了物理准确率；flash 误差没有系统量化 |
| v05 | 每单元每积分点 6 分量 `eps_p`、增量径向返回、建造态到 release 的塑性状态传递、边界应力后处理 | 重熔重置、饱和跨越、release 状态提交、网格质量、测量算子和实验验证仍不完整 |

代码证据：

- v05 的塑性张量状态与径向返回位于 [v05 主求解包装器](../v05/am_thermal_stress_macro_intersection_mech100_v05.py) 第 88–215 行。
- 当前全件 flash 配置位于 [v05 快扫入口](../v05/run_fastscan_flash_v05.sh) 第 24–73 行：1 mm 宏观层、每层一个热状态、激光功率为 0、潜热关闭、每 22 步做一次力学、Winkler 地基和终冷后整体 release。
- 当前后处理在 [postprocess_boundary_stress.py](../v05/postprocess_boundary_stress.py) 第 68–130 行直接做单元/积分点均值、最大值和未加权百分位，没有网格质量门控或实验测量体积卷积。
- v03 的旧层 active-window 导热缩减和人工冷却位于 [v03 主求解器](../v03/am_thermal_stress_macro_intersection_mech100.py) 第 1652–1757 行。

### 2.2 2026-07-10 基线审计结果

以下数值只描述输出目录 `/home/user/work/159/output/fastscan_flash_v05_full91_20260709_174441`，不能外推为模型准确率：

| 审计项 | 当前观察 | 结论 |
|---|---|---|
| 回归测试 | `tests.test_v05_plastic_history` 与 `tests.test_v03_physics_fixes` 共 24 项通过 | 证明局部代码回归，不是实验验证 |
| release 峰值 | 1068 MPa 位于质量指标约 `8.9e-5`、最长/最短边比约 1082 的最差 TET4 单元 | 该峰值主要是网格奇异性证据，不能作为物理结果 |
| 宏观 release 位移 | v04/v05 最大位移分别约 1.37114 mm 与 1.37067 mm；点对点最大差约 1.04 μm | v05 的宏观精度增益尚未被证明，必须用消融实验验证 |
| 热学下界 | 日志中 ersatz/unprinted 区出现 -64.4 K；全部激活后仍有已打印节点低于 300 K，最低约 273.62 K | active-window、人工 sink 或时间离散存在能量/最大值原理问题 |
| 塑性生命周期 | v03 重熔只重置 eqp，v05 的 `eps_p` 未同步重置；release 求解后未显式提交新内部状态 | 建造—重熔—释放状态闭环尚不成立 |
| 饱和硬化 | 单步跨过饱和帽时，当前返回映射可能使用错误的分段模量 | 必须用解析材料点算例修复并验证 |
| 实验对比 | 尚无坐标配准、XRD gauge-volume、轮廓法或位移测量窗口算子 | 当前不能给出可信“仿真准确率” |

因此，当前 full91 案例只能作为流程/性能回归案例，不作为论文的验证案例；修网格前也不再引用全局最大 von Mises 应力。

## 3. 论文研究设计

### 3.1 暂定题目

**An energy-auditable and state-consistent reduced-order thermomechanical model for part-scale residual stress and cut distortion in Ti-6Al-4V laser powder bed fusion**

中文表述：

**面向 Ti-6Al-4V 激光粉末床熔融部件尺度残余应力与切割变形的能量可审计、状态一致降阶热力模型**

### 3.2 待验证假设

- H1：经过能量守恒和 active-window 误差控制后，顺序 flash/meta-layer 模型可在测量尺度 QoI 上逼近更细时间分辨率模型，同时获得至少 10 倍端到端加速。
- H2：完整塑性应变张量历史相较“弹性模型”或“仅标量 eqp”模型，能显著改善附着态残余弹性应变和 release 位移的预测。
- H3：仅使用一个校准试样确定少量降阶参数后，冻结参数可在不同几何、扫描策略或切割状态的 held-out 实验上保持可接受误差。

上述均为研究假设，不应在验证完成前写成论文结论。

### 3.3 预期创新点

1. **能量可审计的层级等效热载荷**：每一步记录输入、储能、导热、对流/辐射和人工降阶项，给出 active-window 相对全域参考解的误差界。
2. **状态一致的建造—重熔—冷却—释放闭环**：`eps_p`、eqp、参考温度和激活状态在所有事件中有唯一、可测试的更新规则。
3. **测量感知验证**：仿真结果先通过与实验几何一致的 XRD diffraction volume、轮廓法空间核或位移测量窗口，再与实验比较，而不是直接比较单元峰值。
4. **速度—精度 Pareto 与模型消融**：定量回答 flash 高度、热时间步、力学 cadence、active-window 和塑性历史分别带来多少误差与计算成本。

是否具备真正的方法新颖性，须在 Gate G0 完成系统文献差距表后再确认；“修复 bug”本身不是论文创新。

### 3.4 明确不做

- 不预测熔池自由表面、Marangoni 对流、反冲压力、飞溅或 keyhole。
- 不声称 flash/meta-layer 热场等同于真实逐道熔池热循环。
- 第一篇不同时引入黏塑性、α/β/α′ 相变、晶体塑性、损伤和疲劳。
- 不声称跨材料、跨机器、跨工艺的通用工业数字孪生。
- 不把 wrapper dry-run、单元测试通过或 full91 成功运行称为物理验证。
- 不把未经测量体积平均的尖角/锚点/畸形单元最大应力作为主结论。
- 不声称 full-loop XLA/GPU 加速，除非端到端 profile 和同硬件基线实际证明。

## 4. 架构决策

### AD-1：v05 冻结，论文实现进入 v06

- v05：只用于复现和 ablation，不继续叠加大规模 monkey patch。
- v06：显式模块化 `thermal`、`mechanics`、`state`、`process`、`measurement` 和 `verification`。
- 迁移必须逐能力完成；每迁移一项，v05 基准和新验证测试同时通过。

### AD-2：主模型是部件尺度降阶模型

- 顺序 flash/meta-layer 是被研究的数值近似，不包装为真实熔池物理。
- 轨迹级热模型只用于局部参考解、热参数先验或误差标尺，不扩展成同一篇论文的第二条主线。
- 若逐道模型需要相变，采用焓法而非仅用旧温度判断的表观热容；这属于局部参考模型的独立任务，不是 flash 主模型的 P0 依赖。

### AD-3：保留单向热—力耦合

对小变形部件尺度残余应力问题，温度驱动力学、力学不反向改变热场的弱耦合是可接受假设。论文必须写明适用范围；若要研究熔池流动或大变形，则另立模型。

### AD-4：整体 Jacobian 不作为新贡献

当前求解器已使用自动微分形成整体 Jacobian。需要补充的是：

- 返回映射在饱和跨越和屈服切换处的正确性；
- AD Jacobian 与有限差分的一致性；
- 最终无量纲残差、反力不平衡、迭代次数和 line-search 记录。

### AD-5：显式基板/支撑为参考，Winkler 地基为降阶选项

Winkler 地基和 active-window 都必须在 P0 阶段与更完整参考模型对照并量化误差。若误差不能被稳定限制，它们不得用于论文最终验证。

## 5. 依赖关系

~~~mermaid
flowchart LR
    A["冻结 v05 基线与文献差距"] --> B["网格质量与数值不变量"]
    B --> C["热能量账本与 active-window 误差"]
    B --> D["J2 返回映射与状态生命周期"]
    C --> E["v06 降阶热过程"]
    D --> E
    E --> F["基板、支撑与切割过程"]
    F --> G["实验测量算子"]
    G --> H["单一试样标定"]
    H --> I["冻结参数的盲验证"]
    I --> J["消融、收敛、UQ 与速度—精度"]
    J --> K["full91 展示与论文归档"]
~~~

任何实验拟合都不得绕过 B–G；任何 full91 结果都不得绕过盲验证 I。

## 6. 分阶段实施计划

工作量标记：S 为半天至两天，M 为三天至一周；若某项超过一周，实施前必须继续拆分。

### Gate G0：冻结基线与锁定论文论点

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P0.1 | 生成环境、Git SHA、命令、配置、材料文件哈希和输出清单；新增 `paper/provenance/baseline_manifest.json` | 任一关键图表可追溯到唯一命令、代码、网格、材料和随机种子 | 无 / S |
| P0.2 | 固化 v05 快速回归和一个小网格端到端基准；新增 `tests/test_v06_baseline_contract.py` | 24 项现有目标测试继续通过；小案例两次运行的 QoI 在确定性容差内一致 | P0.1 / S |
| P0.3 | 建立文献差距与 claim-evidence 表；新增 `paper/literature_gap.md` | 每个拟议创新点至少有最近工作、已有方法、差距和本模型证据；删除无法证明的 claim | 无 / M |

**G0 通过条件**：基线可复现，论文只剩一条降阶模型主线，创新点未与现有工作重复。

### Gate G1：数值正确性与状态闭环

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P1.1 | 增加 TET4 Jacobian、体积、长宽比和无量纲质量审计；`159_local/v06/verification/mesh_quality.py` | 测量/QoI 区域不得存在质量小于 0.05 的单元；阈值 0.05/0.1/0.2 的结论敏感性被报告 | G0 / M |
| P1.2 | 将应力统计改为体积加权和测量窗口加权；替代当前未加权 percentile/max | 论文主 QoI 不依赖全局峰值；畸形单元过滤前后结论一致，过滤规则在看实验前冻结 | P1.1 / M |
| P1.3 | 实现热能量账本；`verification/thermal_ledger.py` 记录热输入、储能、边界损失和人工 sink | 解析/制造解能量闭合误差小于 1%；生产案例小于 2%；每个差额均能归因 | G0 / M；solver hook 已完成，生产收敛门槛待验证 |
| P1.4 | 增加均匀场、纯冷却、激活和大步长测试 | 无热源冷却不跨过环境温度；活动材料绝对温度不为负；不再出现无法解释的低于环境温度 | P1.3 / M |
| P1.5 | 修复 J2 饱和跨越并建立单轴、剪切、循环、多轴材料点测试 | 返回后归一化屈服函数残差小于 `1e-8`；平滑区 AD/有限差分切线相对误差小于 `1e-5` | G0 / M |
| P1.6 | 统一 eqp 与 `eps_p` 的激活、重熔、冷却和 release 提交规则 | 重熔后两类塑性状态均为机器零；release 再次提交的附加塑性增量接近零；状态恢复测试通过 | P1.5 / M |
| P1.7 | 记录力学无量纲残差、反力和迭代历史 | 最终相对残差目标小于 `1e-8`；受约束参考算例反力不平衡小于 0.1% | P1.5 / S |

**G1 通过条件**：负温度、畸形网格峰值、塑性状态断裂和饱和跨越均不再污染主 QoI。G1 未通过前禁止实验调参。

### Gate G2：建立 v06 的可信降阶热力过程

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P2.1 | 建立显式状态容器与纯函数更新核；`v06/state.py`、`v06/mechanics/j2.py`、`v06/thermal/model.py` | 状态形状、单位、生命周期有 schema；材料点和小网格结果与修复后的 v05 基线一致 | G1 / M |
| P2.2 | 将 flash 热载荷定义为守恒的层能量/温度脉冲，并支持自适应热子步 | 改变内部子步不改变层总能量；最细两级温度积分 QoI 变化小于 3% | P1.3 / M |
| P2.3 | active-window 与人工 sink 对全域热参考做消融 | 温度/能量 QoI 误差小于 2–3%，测量尺度应力和 release 位移误差小于 5%；否则关闭该近似 | P2.2 / M |
| P2.4 | 将热应变改为温度相关膨胀系数的积分或增量累积，而非简单 `alpha(T)*(T-T_ref)` | 常系数退化测试精确通过；热循环路径测试无伪路径依赖 | P2.1 / M |
| P2.5 | 建立显式基板/支撑/EDM 切割参考过程，并与 Winkler 降阶地基比较 | 相同约束下反力和位移守恒；Winkler 误差若超过 5% 则最终验证改用显式模型 | P2.1 / M |
| P2.6 | 用温度增量、屈服激活和状态变化触发自适应 mechanics cadence | 与每热步都求力学的参考相比，测量尺度应力和 release 位移变化小于 5% | P2.1 / M |
| P2.7 | 输出/检查点保存 `T,u,eqp,eps_p,T_ref,phase,active` 及单位元数据 | 中断恢复后的最终 QoI 与连续运行在数值容差内一致 | P2.1 / S |

**G2 通过条件**：v06 在小型部件上具有完整建造—重熔—冷却—切割状态闭环，所有降阶近似都有可量化误差。

### Gate G3：测量算子与数据协议

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P3.1 | 实现位移刚体配准、测量线/面采样和不确定度；`validation/operators/displacement.py` | 合成平移/转动不改变变形误差；测量点坐标转换有单元测试 | G2 / M |
| P3.2 | 实现 XRD diffraction-volume 对弹性应变张量的体积卷积；`measurement/xrd.py` | 矩形均匀/梯度合成场通过；正式 Strantza 比较还需实现论文菱形体积，不直接用单元中心应力代替 XRD | G2 / M；矩形算子已完成 |
| P3.3 | 实现轮廓法截面空间核和切割前后坐标映射；`validation/operators/contour.py` | 合成应力图可恢复；切割面、法向和符号约定写入数据 schema | G2 / M |
| P3.4 | 统一数据与指标输出；`validation/schema.json`、`validation/metrics.py` | 每次比较生成 CSV、JSON、图和 provenance；禁止手工复制论文数字 | P3.1–P3.3 / S |

**G3 通过条件**：每一种实验都有对应的 forward measurement operator，仿真与实验在同一坐标、同一空间分辨率和同一物理量上比较。

### Gate G4：参数标定与独立实验验证

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P4.1 | 固定材料表来源和参数边界，做可辨识性分析 | 第一篇最多标定 3 个降阶参数；高度相关且不可辨识的参数被固定或删除 | G3 / M |
| P4.2 | 用 Bayat et al. 的 Ti64 薄壁/支撑切割变形案例做唯一训练试样 | 只调整预注册参数；位移场 NRMSE 目标小于 10%，极值误差目标小于 10%；报告拟合不确定度 | P4.1 / M |
| P4.3 | 冻结参数和评估脚本；生成 `validation/frozen_model.yaml` | 在打开 held-out 结果前记录参数、阈值、坐标变换和失败条件；后续改动必须形成版本化新模型 | P4.2 / S |
| P4.4 | 用 Strantza et al. 的 C45 bridge XRD 残余弹性应变做跨几何盲验证 | 不重新调参；预测符号和空间拓扑正确；归一化 NRMSE 目标小于 15%，或 reduced chi-square 不大于 2 | P4.3 / M |
| P4.5 | 用 I0/45° 扫描策略及部分 EDM 切割状态检验迁移性 | 不重新调参；正确预测策略间排序和切割后的应力重分布方向 | P4.4 / M |
| P4.6 | 用 Ahmad et al. 轮廓法或 Mishurova et al. 数据做第二跨几何检查 | 工艺参数不足时只作为趋势/量级检查并显式降级证据等级，不伪装成严格验证 | P4.3 / M |

**G4 通过条件**：至少一个不同几何/边界条件的 held-out 数据集在冻结参数下达到预注册门槛。若只有同一试样拟合，不得使用“validated model”表述。

### Gate G5：收敛、消融、UQ 与可选物理

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P5.1 | 至少三级网格、三级热时间步、三级 mechanics cadence 收敛 | 积分/测量尺度 QoI 最细两级变化小于 5%；局部应力 QoI 小于 10% | G4 / M |
| P5.2 | 做 elastic、scalar-eqp、tensor-`eps_p`、状态修复、active-window、地基模型消融 | 每项报告误差变化、运行时间和显存；能回答 v05 相比 v03/v04 的实际精度收益 | G4 / M |
| P5.3 | 对 5–7 个关键参数做 Morris/Latin-hypercube 筛选和不确定度传播 | 给出参数敏感度排序与 95% 预测区间；不以单一最优参数掩盖不确定度 | P4.1 / M |
| P5.4 | 根据 held-out 残差决定是否加入一种高温黏塑性/松弛模型 | 只有当误差随 dwell/冷却速率系统变化且参数可辨识时，才在 J2 基线上加入 Perzyna、Maxwell 或 Arrhenius 中的一种 | P5.2–P5.3 / M |
| P5.5 | 判断冶金相变是否另立后续论文 | 若误差集中在相变温区且有独立相分数/转变应变数据，另开模型；不得与 P5.4 同时无约束拟合 | P5.2–P5.3 / S |

**G5 通过条件**：论文的每个精度提升都能通过消融归因，参数不确定度和离散误差均进入结果区间。

### Gate G6：全件展示、复现包与论文

| ID | 任务与主要文件 | 验收标准 | 依赖/规模 |
|---|---|---|---|
| P6.1 | 修复/remesh 当前 90 mm full91 几何后，以冻结模型重跑 | 不再以畸形单元峰值汇报；只报告经网格质量和测量尺度处理的场、分位数和积分 QoI | G5 / M |
| P6.2 | 同硬件、同网格、同输出频率比较 flash 与细分辨率基线 | 给出端到端 wall time、峰值显存和 QoI 误差；目标加速至少 10 倍，否则调整论文定位 | P6.1 / M |
| P6.3 | 生成图表、表格、配置、环境锁文件和一键复现实验 | 论文所有数字由脚本生成；公开数据可直接运行，受限数据有获取说明和 checksum | P6.2 / M |
| P6.4 | 按 claim-evidence matrix 写作并进行内部盲审 | 摘要每个定量 claim 均能指向验证图/表；限制条件、失败案例和适用域完整披露 | P6.3 / M |

## 7. 预注册的验证指标

这些是执行前的 go/no-go 目标，不是通用行业标准。应在查看 held-out 结果前冻结；若改变，必须记录原因，不能事后选择有利阈值。

| 层级 | 指标 |
|---|---|
| 材料点 | 屈服函数归一化残差小于 `1e-8`；平滑区 AD/有限差分切线误差小于 `1e-5` |
| 热学验证 | 小型基准能量闭合小于 1%，生产案例小于 2%；无非物理绝对温度；纯冷却不跨越环境温度 |
| 力学平衡 | 最终无量纲残差目标小于 `1e-8`，反力不平衡小于 0.1% |
| 离散收敛 | 三网格、三时间步、三 cadence；测量尺度 QoI 最细两级变化小于 5% |
| active-window | 相对全域参考：热 QoI 小于 2–3%，应力/释放位移小于 5% |
| 局部应力 | 必须先按实验空间分辨率做体积平均；变化目标小于 10%；不使用奇异峰值 |
| 校准试样 | 位移场 NRMSE 小于 10%，极值误差小于 10%，同时报告 bias |
| 盲 XRD | 正确符号/拓扑；NRMSE 目标小于 15%，且 reduced chi-square 目标不大于 2 或至少 90% 点落在联合 2σ 区间 |
| 扫描/切割迁移 | 预测正确的相对排序和应力重分布方向 |
| 性能 | 同硬件端到端加速至少 10 倍，同时上述 QoI 误差门槛成立 |

误差统计默认使用 MAE、bias、RMSE、按测量动态范围归一化的 NRMSE、相关系数和含实验协方差的 reduced chi-square。接近零的应变/位移不使用 MAPE。

## 8. 标定—验证数据矩阵

| 数据/论文 | 用途 | 可标定参数 | 证据等级 |
|---|---|---|---|
| NIST A-AMB2022-01 Ti64 bare-plate | 约束局部热模型或热输入先验；不直接证明部件尺度应力 | 吸收率/局部热输入先验，若数据映射成立 | 热学辅助 |
| Bayat et al. 2020 | 支撑薄件 EDM release 变形；唯一训练试样 | 最多 3 个：如等效松弛温度、地基/支撑等效参数、flash 热尺度 | 标定 |
| Strantza et al. 2018 C45 bridge | XRD 残余弹性应变 | 无，参数冻结 | 主要盲验证 |
| Strantza et al. 2021 I0/45° 与部分切割 | 扫描策略排序和切割后重分布 | 无，参数冻结 | 迁移/盲验证 |
| Ahmad et al. 2018 double-cantilever | 轮廓法应力图与量级 | 无；工艺不全时不得反调参 | 次级跨几何 |
| 当前 full91 | 大规模流程、性能和场展示 | 无 | 展示，不是验证 |

如果原始实验数据无法获得，只能从论文图中数字化，则必须把像素标定、曲线提取和坐标配准误差加入实验不确定度，并在论文中声明。若无法取得任何独立 held-out 数据，高水平“实验验证论文”路线停止，转为数值方法/软件论文或补做实验。

## 9. 推荐仓库结构

~~~text
159_local/v06/
├── state.py
├── thermal/
│   ├── model.py
│   ├── flash.py
│   └── energy_ledger.py
├── mechanics/
│   ├── j2.py
│   ├── lifecycle.py
│   └── release.py
├── process/
│   ├── activation.py
│   ├── support.py
│   └── cutting.py
├── measurement/
│   ├── displacement.py
│   ├── xrd.py
│   └── contour.py
└── cli/

verification/
├── material_point/
├── manufactured_thermal/
├── mesh_convergence/
└── active_window/

validation/
├── datasets/          # raw 大文件默认不入 Git；保留元数据、URL、checksum
├── cases/
│   ├── bayat_2020/
│   ├── strantza_2018/
│   ├── strantza_2021/
│   └── ahmad_2018/
├── metrics.py
└── frozen_model.yaml

paper/
├── provenance/
├── figures/
├── tables/
├── literature_gap.md
└── claim_evidence_matrix.md
~~~

## 10. 论文图表清单

1. 模型边界与“材料点 → patch → 小型构建 → coupon → held-out 几何”的验证金字塔。
2. 建造、重熔、冷却、release 的内部状态生命周期。
3. 热能量闭合、材料点返回映射和三层离散收敛。
4. active-window/flash 的速度—精度 Pareto。
5. elastic、scalar-eqp、tensor-`eps_p` 与状态修复的消融图。
6. Bayat 标定位移场及误差分布。
7. Strantza XRD gauge-volume 盲验证与不确定度带。
8. 扫描策略/部分切割的 held-out 排序与重分布。
9. UQ 敏感度和 95% 预测区间。
10. 修网格后的 full91 场展示；明确标为应用示例。

## 11. 主要风险与停止条件

| 风险 | 缓解措施 | 停止/降级条件 |
|---|---|---|
| 实验原始数据不可得 | 联系作者、使用开放仓库、数字化并传播不确定度 | 无独立 held-out 数据则不声称实验验证 |
| TET4 sliver 污染应力 | 质量门控、局部重网格、必要时 TET10/更高质量网格 | 主结论依赖阈值或单个尖峰则停止汇报局部应力 |
| 参数不可辨识 | 限制为最多 3 个降阶参数，做 profile/Fisher/SVD 分析 | 多组参数产生同等拟合且外推分歧大，则收缩模型 |
| active-window/sink 误差过大 | 与全域参考对照，优先优化求解器而非加强人工 sink | 应力/位移误差超过 5% 则关闭该近似 |
| 高温物理不足 | 先看误差随 dwell/冷却速率的系统性，再加一种黏塑性模型 | 无独立数据约束时不增加自由参数 |
| 论文范围膨胀 | 所有新物理必须通过残差证据和 ablation 决策 | 冶金相变、熔池 CFD、晶体塑性拆为后续论文 |
| 加速不可复现 | 同硬件、同网格、同输出、端到端计时 | 未达到 10 倍时不以高性能为主要 claim |

## 12. 参考论文与实验来源

以下均为本计划直接使用的原始论文或官方数据源；最终稿还需完成系统检索和卷期页码核对。

1. Bayat et al. (2020), part-scale thermo-mechanical modelling and sequential flash heating with experimental validation, *Additive Manufacturing* 36, 101508. [DOI](https://doi.org/10.1016/j.addma.2020.101508)；[DTU 记录](https://orbit.dtu.dk/en/publications/part-scale-thermo-mechanical-modelling-of-distortions-in-laser-po/)
2. Strantza et al. (2018), Ti-6Al-4V bridge residual elastic strain/XRD coupled experimental-computational study. [DOI](https://doi.org/10.1016/j.matlet.2018.07.141)；[NIST 原文](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=925448)
3. Strantza et al. (2021), scan-strategy and partial-cut residual stress study, *Additive Manufacturing* 45, 102003. [DOI](https://doi.org/10.1016/j.addma.2021.102003)；[OSTI 原文](https://www.osti.gov/servlets/purl/1785480)
4. Ganeriwala et al. (2019), thermomechanical residual-stress model evaluation for Ti-6Al-4V LPBF. [DOI](https://doi.org/10.1016/j.addma.2019.03.034)；[NIST 原文](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=926890)
5. Ahmad et al. (2018), Ti-6Al-4V/IN718 residual stress by contour method and simulation. [DOI](https://doi.org/10.1016/j.addma.2018.06.002)；[Coventry 原文](https://pure.coventry.ac.uk/ws/portalfiles/portal/19511597/1_s2.0_S2214860418301799_main.pdf)
6. Tan et al. (2019), thermo-metallurgical-mechanical modelling of Ti-6Al-4V LPBF. [DOI](https://doi.org/10.1016/j.matdes.2019.107642)
7. Promoppatum & Rollett (2021), constitutive/rate sensitivity in LPBF thermomechanics. [DOI](https://doi.org/10.1016/j.addma.2020.101680)
8. De Baere et al. (2021), Arrhenius-creep stress-relief modelling after LPBF. [DOI](https://doi.org/10.1016/j.addma.2020.101818)；[DTU 记录](https://orbit.dtu.dk/en/publications/thermo-mechanical-modelling-of-stress-relief-heat-treatments-afte/)
9. Mishurova et al. (2017), Ti-6Al-4V LPBF residual stress experimental evidence. [DOI](https://doi.org/10.3390/ma10040348)；[DLR 记录](https://elib.dlr.de/115931/)
10. NIST AMBench A-AMB2022-01 Ti-6Al-4V bare-plate benchmark. [官方数据页](https://www.nist.gov/ambench/amb2022-01-benchmark-challenge-problems)

AMBench 2022 的公开残余应力数据集 [NIST 数据记录](https://doi.org/10.18434/mds2-2711) 使用 IN718，只可参考测量/数据管线，不能直接当作 Ti-6Al-4V 的材料验证。

## 13. Definition of Done

只有同时满足以下条件，才把模型称为“论文级、经过实验验证”：

- G0–G5 全部通过，所有阈值和 held-out 数据在结果揭盲前冻结；
- 至少一个训练试样和一个不同几何/边界条件的独立验证试样；
- 数值误差、实验不确定度和参数不确定度被分开报告；
- v05 张量历史的精度收益由消融证明，而非仅凭理论合理性；
- full91 网格通过质量门控，最终结论不依赖全局最大应力；
- 每个论文数字可由固定代码、配置和数据一键再生；
- 论文明确陈述降阶模型不能预测熔池和微观组织。

## 14. 第一轮立即执行的最小任务

1. `[未完成]` 冻结当前 v05 baseline manifest，并保留既有目标测试结果。
2. `[已完成基础门禁]` 给 full91 网格和后处理加入质量审计，停止引用 1068 MPa 奇异峰值；下一步是重网格与三级收敛。
3. `[已完成 solver hook]` 建立逐步热能量账本并复现下冲；下一步比较一致质量、质量集总、时间步和热源尺度网格。
4. `[部分完成]` 饱和跨越、卸载幂等和重熔重置已覆盖；反向循环与算法切线有限差分仍缺。
5. `[已完成第一版]` 修复 `eps_p/eqp/eps_ref` 生命周期、release 状态提交与形状失败快退；checkpoint/restart 仍缺。
6. `[未完成]` 用小网格比较 active-window 与全域参考，决定人工 sink 是否保留。
7. `[未完成且阻断实验 claim]` 取得 Bayat/Strantza 原始数据，或完成带像素、配准和 gauge 身份不确定度的数据集；实现菱形 diffraction-volume 后再开始参数标定。

这七项完成后再进入 v06 的大规模迁移；在此之前不增加冶金相变或高温黏塑性自由参数。
