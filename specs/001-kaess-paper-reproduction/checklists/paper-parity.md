# Paper-Parity Checklist: Kaess 2023

**Purpose**: 跟踪论文级数值复现的科学与计算门禁。

**Created**: 2026-07-23

**Feature**: [spec.md](../spec.md)

## G0 — Sources and Claim

- [x] PAR001 论文 PDF、全文和引用信息具有 SHA-256。
- [x] PAR002 来源矩阵覆盖所有 critical/high-impact 输入。
- [ ] PAR003 Figure 8/9 数字化 CSV 记录坐标、单位和读图误差。
- [x] PAR004 `kaess_2023.json` 的数字化状态字段无自相矛盾。
- [x] PAR005 作者输入请求和响应形成审计记录。
- [x] PAR006 所有未公开输入进入 assumptions register。
- [ ] PAR007 QoI、路径、插值、阈值和不确定度组合方法已批准。
- [ ] PAR008 标定工况和 held-out 工况已冻结。
- [ ] PAR009 正式 claim 固定为 code-to-code，除非作者资产审批升级。

## G1/G2 — P0 Physics and Code Verification

- [x] PAR010 底面全部 `uz=0`，最小 `x/y` 锚点消除刚体模态。
- [ ] PAR011 等价锚点改变远场应力/翘曲不超过批准阈值。
- [x] PAR012 半球三维高斯热源解析分布测试通过。
- [x] PAR013 热源体积分与吸收功率相差不超过 0.5%。
- [x] PAR014 未激活单元对热容、导热、刚度、残差和 DOF 零贡献。
- [x] PAR015 小模型活动域与物理删除参考差不超过 `1e-8`。
- [x] PAR016 每个阶段识别当前活动域真实暴露顶面。
- [x] PAR017 对流/辐射表面积及积分误差不超过 0.5%。
- [x] PAR018 冷却起始时底温和环境温度均按冻结协议切换。
- [ ] PAR019 粉末温变导热率、潜热和温变固体表已验证。
- [ ] PAR020 自由热膨胀单元不产生显著假应力。
- [ ] PAR021 J2 加载—卸载—再加载材料点测试通过。
- [ ] PAR022 J2 一致切线有限差分/V 形谷测试通过。
- [x] PAR023 B-bar 均匀应变、近不可压缩和零能模态测试通过。
- [ ] PAR024 不存在无来源 liquid/mushy 二次刚度缩放。
- [ ] PAR025 不存在无来源 stress-free/eqp 历史重置。
- [x] PAR026 release 使用显式、可视化、可哈希 cell set。
- [ ] PAR027 release 前后无新增刚体漂移。

## G3 — CPU Small-Scale Verification Baseline

- [ ] PAR028 single-track 能量账闭合误差不超过 1%。
- [ ] PAR029 `1×30 µm` 的时间、路径和网格收敛通过。
- [ ] PAR030 真实全网格 12–20 步前缀使用 MKL/OMP=1 重复两次并通过。
- [ ] PAR031 kernel、small-domain、1层完整 CPU case 和缩减空间域三层
  mini-cycle 均以 MKL/OMP=1 重复两次并通过；共同覆盖 build、recoat、
  多层历史、cooling 和 release，run id/hash 已冻结。
- [ ] PAR032 两档最细离散的关键 QoI 差不超过 2%。
- [ ] PAR033 求解器容差收紧后关键 QoI 差不超过 1%。
- [ ] PAR034 不存在未来层非物理贡献；全域未激活 `u_max` 未被误用为 QoI。

## G4 — Accelerated Backend Qualification

- [ ] PAR035 CPU/candidate 配对具有同 commit、dirty diff、输入、float64、
  mask 和 acceptance-model 身份，且当前 candidate run id 属于资格包。
- [ ] PAR036 比较使用原生 float64 checkpoint；VTU 只作辅助可视化。
- [ ] PAR037 CPU reference 与候选各至少重复两次，科学 reference 使用
  MKL/OMP=1。
- [ ] PAR038 manifest 分阶段记录 local assembly、global matrix、linear
  solver、state residency、host transfer 和 fallback；跨文件 validator
  已复算 profiler/manifest/checkpoint hashes。
- [ ] PAR039 active/printed 域温度场差≤0.1%，应力/eqp 场差≤1%，事件一致。
- [ ] PAR040 候选的收敛门和线性解次数增幅通过冻结阈值。
- [ ] PAR041 最小 build/cooling/release CPU/GPU 对照通过，翘曲曲线
  `L2≤2%` 且最大差满足 `max(0.5 µm, 2%)`。
- [ ] PAR042 backend parity、energy audit 和 convergence audit 分别判定；
  任一正式 gate 失败都不得 promotion。
- [ ] PAR043 两个同配置性能样本的中位 wall speedup 达到批准门，并分别
  报告冷启动、稳态、RAM、VRAM 和 CPU 线程数。
- [ ] PAR044 `hybrid_gpu_assembly_cpu_pardiso` 未误标为 `full_gpu`；
  真正 full-GPU 条件未满足时状态为 unsupported/experimental。

## G5/G6 — Accelerated Formal Reproduction and Matrix

- [ ] PAR045 合格后端的 `3×30 µm` 多层历史桥接通过。
- [ ] PAR046 合格后端的论文 `5×60 µm` 预正式桥接通过。
- [ ] PAR047 正式配置为 10×30 µm、150°C、250 W、850 mm/s。
- [ ] PAR048 使用 29,568 C3D8 参考网格和冻结路径，构建、冷却和 release
  检查点分别通过。
- [ ] PAR049 Figure 8 `σx` 深度曲线的符号、峰谷和过零位置通过。
- [ ] PAR050 Figure 9 翘曲方向、曲线和最大前端位移通过。
- [ ] PAR051 数字化、离散、CPU-reference 重复和输入不确定度分别报告。
- [ ] PAR052 未通过项以 fail/partial 状态保存，不事后调整阈值；报告中的
  threshold artifact/hash/approval 与 G0 和 run manifest 绑定。
- [ ] PAR053 只有已通过的明确 backend mode 进入参数矩阵；失败 GPU 结果
  未进入正式声明。

## G7 — Reproduction Package

- [ ] PAR054 run manifest 通过 schema。
- [ ] PAR055 paper comparison report 通过 schema。
- [ ] PAR056 所有输入和正式输出具有 SHA-256。
- [ ] PAR057 solver command、环境和 dirty diff 可重建。
- [ ] PAR058 原生 float64 checkpoint、VTU、CSV、日志、能量账和逐增量收敛记录完整。
- [ ] PAR059 报告图表由结构化数据自动生成。
- [ ] PAR060 新输出目录的独立复跑生成相同 QoI。
- [ ] PAR061 技术报告区分 verified、partial、missing 和 out-of-scope。
- [ ] PAR062 XRD 明确标记为附加算子，不作为 Kaess 成功门。
