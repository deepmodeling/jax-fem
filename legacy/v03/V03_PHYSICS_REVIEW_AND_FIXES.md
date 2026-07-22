# v03 热-力模型物理审查与修复（2026-07-08）

范围：对照真实 3D 粉末床熔融（LPBF）物理，审查
`159_local/v03/am_thermal_stress_macro_intersection_mech100.py` 的物理设定。
审查基于真实 h60 网格（`schema/0119_c3d4_only.inp`，197,266 TET4 /
52,739 节点）和真实路径文件（825,876 步、91 层）的实测证据。

## 审查结论摘要

单位制自洽（网格为米制，零件约 91×88×97 mm）；`T_ref` 凝固时写入、重熔回退
的设计正确；潜热（apparent-cp）、相态机、moving window 骨架合理。但存在
两处 bug 级问题和多处物理缺失，修复前模型接近"绝热 + 8 点接地 + 无层间
冷却 + 线弹性"，其温度史和残余应力都不能定量解释。

## 已修复（默认关闭，用 physfix 脚本或对应 CLI 开启）

| # | 问题 | 实测证据 | 修复 | 开关 |
|---|---|---|---|---|
| 1 | 对流/辐射选择器只匹配包围盒平面，真实曲面零件上选中 **0 个面**，模型近似绝热 | `thermal_boundary_face_counts=[0,0]` | `jax_fem/fe.py` 新增外表面判定（面只属于一个单元），location_fn 带 `exterior_only=True` 属性时只选外表面；v03 新增 `--surface-selection exterior`，把对流+辐射施加到基面以上全部外表面 | `--surface-selection {box,exterior}`，默认 box（兼容） |
| 2 | 底面 BC 容差 `1e-8×span≈1nm`，CAD 底面节点有亚毫米抖动，热学固定温度和力学夹持都只作用在 **8/1225** 个节点上 | `dirichlet_node_counts=[8]` | `make_box_locations` 支持绝对容差 | `--boundary-tol 1e-4` → 1225 节点 |
| 3 | 路径文件无层间 recoat/dwell（825,876 步全部 dt=1e-4s，91 层连续扫描），层间冷却窗口缺失 | 路径 CSV 实测 dt 恒定、无 recoat 模式 | `generate_path_file_step_states` 在层号递增处插入 laser-off recoat 状态，dt = recoat_time/recoat_steps；raster 生成器同步改为大步长（原实现 10s/1e-4 = 10 万步不可用） | `--recoat-time 10 --recoat-steps 10` |
| 4 | 末端冷却仅 cooling_steps×dt（默认 20×1e-4=2ms），输出不是冷却后残余应力 | run 脚本参数 | 冷却步支持独立大步长 | `--cooling-steps 300 --cooling-dt 2.0` + `--release-after-cooling` |
| 5 | 新激活层继承 void 单元漂移温度而不是铺粉温度 | 代码只记录不重置 | `reset_new_cell_nodal_temperature()`：新激活单元中不与已打印材料共享的节点重置为预热温度 | `--reset-activation-temperature` |
| 6 | run 脚本用 linear_elastic 覆盖了 config 的 j2_plastic，残余应力无屈服封顶 | 脚本参数 | physfix 脚本切回 j2_plastic | `--mechanics-model j2_plastic` |

新运行脚本：`159_local/v04/run_macro_intersection_h60_mech100_XLA_first5_physfix.sh`
（首 5 层、全部修复开启；结果**有意**与旧 first5 不可比）。

测试：`tests/test_v03_physics_fixes.py`（8 个：外表面判定、容差、recoat 插入、
cooling-dt、激活温度重置），与既有 131 个测试共同通过。真实网格 smoke
（2 层 raster + recoat + release）验证接线：面数 75,459、节点 1225、recoat/
cooling/release 状态齐全，j2 收敛。

## 2026-07-08 追加：表面通量激活掩码（第 7 项修复）

启用 exterior 表面后发现：void（未铺粉）单元的外表面也会拿到对流/辐射通量，
而 void 单元方程近奇异（热质量 ×1e-6，任何温度都满足绝对残差容差），直接法
把它们解到荒谬值并被通量放大。物理上 void 面也不是真实表面。修复：
`TransientThermal` 表面通量乘以 per-face 的 printed 掩码，掩码通过
`internal_vars_surfaces` 每步传入（`set_params` 末尾新增 `surface_mask_quad`
参数）；`--surface-active-mask` 默认在 exterior 模式开启、box 模式传全 1
（与历史行为逐位一致，small-loop VTU 对照 `max_abs=0` 验证）。
效果：200 步探针中冷却段 T_min 从 252K 恢复到 299.9K。

## 2026-07-08 追加：发现 v03 既有数值缺陷 —— 扫描期负绝对温度

真实 h60 网格 + 路径文件的探针（**纯 legacy 参数，无任何 physfix 标志**）
显示 step 50 即出现 `T_min=-320K`；physfix 运行同样存在（step 100 约
`-392K`）。该缺陷存在于所有历史 v03/v04 运行中，此前无人查看 T_min。

根因判断：TET4 热问题只用 1 个积分点，质量矩阵是 rank-1（V/16 全元素阵），
4 个局部模态中 3 个几乎不受质量项约束；粉末低导热（k=1.0）+ dt=1e-4 下
Fourier 数 ~1e-5，刚度项也约束不住，热源附近出现大幅空间振荡（欠积分伪振）。

影响范围：局部、瞬态（熔化后被覆盖）；相态逻辑（负温=保持粉末）、
max_temperature 历史不受影响；表面辐射已被激活掩码限制在 printed 面上。
但对定量温度史是实质污染。

建议修复（未实施）：给热问题提高积分阶（TET4 二阶规则 4 积分点，质量阵满秩）。
需要热/力两个 Problem 同步改（quad 数组 shape 共享），装配成本约 ×4
（batch 修复后装配只占每步 ~3%，可接受）；或改用质量集中（框架改动更大）。

## 2026-07-08 追加：扩展材料表（带出处列）

新表 `materials/Ti-6Al-4V/*_ext.csv`（E/k/cp/yield/alpha），每行带 `source`
列：原始行标 `original v01 dataset`，高温行标 `estimate: Mills 2002 trend /
typical LPBF FE practice; verify`——**估计值，定量使用前需用户核对数据源**。
配套配置 `ti64_material_config_physfix.json`（指向 ext 表 + j2_plastic）。
`PropertyTable` 兼容额外列（只读 T,value）。

## 2026-07-08 追加：宏观路线判定 —— 该模型不可能熔化，改用"激活即固化"（路线 B）

能量核算：路径扫描速度 4 m/s、吸收功率 1500 W → 线能量密度 0.375 J/mm，
摊到 1mm 宏观层的体积能量密度约 1.9e8 J/m³；把 Ti64 从 300K 加热到熔化
需要约 4.9e9 J/m³ —— **差 25 倍**。且光斑（r=1mm）小于单元尺寸（~2.5mm），
热源亚网格。结论：macro1mm 参数化下熔池路线（路线 A）死路；探针实测
（4 积分点修复后）单元级最高温 ~896K，与核算一致。

改用部件级标准做法（路线 B）：`--solidus-temperature 0 --liquidus-temperature 0`
启用"激活即固化"兼容模式，新增 `--stress-relaxation-temperature`（Ti64 常取
1073–1173K）作为应力自由参考温度——高于该温度材料不承载应力，残余应力来自
受约束的冷却收缩。v03/v04 的 history 更新（含 v04 JIT kernel）已同步支持。

## 2026-07-08 追加：力学求解链修复（路线 B 调试中发现）

1. **Newton 无迭代上限**（会无限循环挂死）：`jax_fem.solver.solver()` 新增
   `max_iter`（默认 100，超限抛诊断错误）。
2. **力学容差自虐**：`run_mechanics` 硬编码 tol=1e-9/rel_tol=1e-11；新增
   `--mechanics-tol/--mechanics-rel-tol/--mechanics-max-iter/--mechanics-line-search`
   旋钮（rel 1e-5~1e-6 即工程精度，实测把单次力学求解从 ~15min 压到 ~2min）。
3. **release 锚点落在 void 上**：`make_anchor_mechanics_bc` 从全网格几何极值
   选点，落在刚度 ~1e-8 的 void 单元上导致奇异（NaN）。现在只从已打印节点选。
4. **j2 从裁剪式升级为径向返回**：旧实现 `scale=min(1, yield/seq)` 的屈服面
   在单次求解内不扩张（eqp_old 求解后才更新），求解内切线理想塑性 → 全屈服体
   奇异。新实现 `delta_eqp=(seq-yield)/(3mu+H)`、`scale=1-3mu*delta_eqp/seq`，
   与 `compute_eqp_update()` 同式；H=0 时与旧公式**严格等价**（回归安全），
   H>0 时一致切线正定。配套新增 `hardening_table_ext.csv`（工程估计值，
   带出处列，需核对）。
5. **热激活（--activation-reset-temperature）**：路线 B 首个全量运行在层 2
   激活的力学求解处失败（整层以 T=300K、T_ref=1100K 瞬间固化 = GPa 级载荷
   阶跃，Newton 在弹塑性边界振荡，rel 残差卡 2.7e-2）。修法是修物理而非
   调求解器：新层以松弛温度激活（刚沉积材料本来就是热的、无应力的），
   随冷却逐步建立应力。跨层切换探针验证：层 2 激活平稳收敛，应力演化
   868→905→(recoat 冷却)→982→1000 MPa，终态 ≈ 屈服+硬化。line_search
   同轮升级为 8 级回溯取最优（弹塑性分支翻转的安全网）。
6. **单层 release 判定为物理病态**：1mm 厚、整层屈服、无基板的自由薄层靠
   3 锚点回弹是大变形问题，小应变框架无良解（残差卡 ~1e-2）。单层测试
   跳过 release，交付底面约束下的冷却残余应力；release 适用于 ≥5 层或
   加基板后。

## 2026-07-09 追加：void 幽灵扩散、屈服饱和、弹性地基

1. **void 幽灵扩散**（用户发现"激光上方升温"）：`inactive_thermal_factor` 同时
   缩 k 和 rho，扩散率 k/(rho·cp) 保持实体值 → 温度以实体速度渗入未铺粉区
   5–10mm（实测）。修复：`--inactive-mass-factor 1.0`（质量全值、只缩 k，
   扩散率 ×1e-6），v03+v04 JIT kernel 同步。验证：2.5mm 以上渗透
   440K→300.1K，只剩共享界面节点的真实表面温度。顺带根治 void 自由度
   近奇异漂移。
2. **屈服饱和** `--yield-saturation-stress`（~UTS 1.15e9）：fast-scan 全件
   在底面夹持区出现 vm 2.05 GPa / eqp 0.76——线性硬化被外推远超 ~10% 应变
   有效范围。饱和后转理想塑性（径向返回内 H_eff=0），stress_fn 与
   compute_eqp_update 同式。
3. **弹性地基底面约束** `--bottom-mechanics-bc elastic` +
   `--bottom-foundation-stiffness`（Pa/m，1e12 ≈ 25mm 钢板量级）：
   Winkler 弹簧替代刚性 Dirichlet 夹持（现实基板有限柔度；刚性夹持边缘
   数学奇异）。ThermoMechanical 新增 get_surface_maps 弹簧牵引
   （t=-k_s·u，弱形式内核返回 +k_s·u）。验证（5 层 flash 探针）：
   夹持区 eqp 0.76→0.003，vm_max 2055→960 MPa（封顶之下），release 正常。
   legacy fixed 模式逐位不变。

## 已知未修（按优先级）

1. **材料表高温段截断平延拓**：E/k/cp 止于 923K，yield 止于 1273K。
   推荐补充值见 `materials/Ti-6Al-4V/TODO.md`（需用户核对数据源）。
2. **无基板**：`substrate_thickness=0`，底面 Dirichlet 直接当热沉/夹具；
   建议在网格中加基板层或至少敏感性验证。
3. **周围粉床侧向导热为零**（future_layer_mode=void）；当前层粉末
   k=1.0 W/mK 也偏高（真实 ~0.1–0.3）。
4. **熔池 Marangoni 对流无各向异性 k 放大**：`conductivity_liquid` 是标定
   旋钮，常放大 2–10x 校准熔深。
5. **j2 实现是简化裁剪**（非增量 return-mapping），且 eqp 每
   mechanics-every 步才积分一次；代码已有运行时警告。
6. **apparent-cp 潜热**在大步长跨越糊状区间时丢能量；如 recoat/cooling
   大步长下发现能量不守恒，考虑 enthalpy 法。
7. 宏观固有近似（1mm 层/1mm 光斑是 60μm 实际层的 lumped 替代、忽略蒸发/
   气流/飞溅）：保留，但吸收率/源深度需要对基准数据标定。

## 兼容性说明

所有修复默认关闭，v03/v04 旧脚本行为逐位不变（131 个回归测试通过）。
物理修正结果不与旧输出做数值对比——修的就是物理本身。
