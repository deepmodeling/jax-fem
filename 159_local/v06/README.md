# v06：论文模型的验证主线

v06 是当前主开发目录，v05 只保留为消融和复现基线。这里的目标不是把一次成功运行称为“论文级”，而是逐层建立可复现、可审计、能与实验同量纲比较的热—弹塑性模型。

当前成熟度：**求解器级数值审计与实验 forward-operator 骨架已建立，尚未完成实验验证**。完整路线见 [codex_model_update.md](codex_model_update.md)。

## 已实现

- `driver.py`：v03 时间循环 + v04 性能层的唯一兼容入口，不依赖 v05 runtime；会在 v04 JIT/cache 安装后重新挂载 v06 生命周期钩子。
- `mechanics/j2.py`：纯 JAX 小应变关联 J2、精确的线性硬化饱和跨越、完整 `eps_p` 张量和弹性应变反算。
- `mechanics/lifecycle.py`：出生/重熔 `eps_ref`、高温松弛增量截断、事件强制 mechanics update；release 同时继承 `eps_p/eps_ref`。
- `material_validation.py`：材料表进入 JIT 前检查温度轴、有限值以及 `E>0`、`-1<ν<0.5`、`σy>0`、`H≥0`、正热容/导热率/密度。
- VTU 输出：`elastic_strain_quad_*`、`eps_p_quad_*`、`eps_ref_quad_*`，为 XRD 同物理量比较提供输入。
- `verification/mesh_quality.py`、`mesh_audit.py`：TET4 体积、方向、边长比、非有限坐标和无量纲质量审计。
- `verification/run_audit.py`：全瞬态而非仅末步审计；检查有限值、负 von Mises/eqp、二值状态、低于环境温度、显式无源上界、质量门控和积分点体积加权 QoI。
- `verification/thermal_balance.py`、`thermal_ledger.py`：在最终 v04 solver 外层对每个已接受热步重新组装未施加 Dirichlet 行替换的残差，记录储能、实际沉积热、人工 sink、表面交换、固定温度边界交换、自由残差、状态重置和温度不变量。
- `verification/response_gate.py`：零输入运行标记为不变量 smoke；非零制造解必须同时产生温度、受约束应力、释放应力/位移和 XRD 响应。
- `measurement/xrd.py`：精确的凸 `gauge box ∩ TET4` 交体积、覆盖率门禁、P0 弹性应变卷积和 microstrain 投影。
- `measurement/xrd_vtu.py`：带尺度/旋转/平移/配准残差的 VTU→XRD 管线；只接受已知附着态、最终 cooling、测量温度带内、配准 RMS 合格且无翻转/退化单元的结果。
- `validation/cases/strantza_2018.json`：经 NIST 原文核对的 Ti-6Al-4V **C45** bridge 工艺、坐标、gauge 尺寸、不确定度和论文锚点。
- `validation/screening.py`：文字锚点只能生成 `manual_unverified_screening`；正式 pointwise API 在版本化原始数据 schema 完成前保持禁用。
- `provenance.py`：交叉核对热账本、审计 VTU、XRD 输入、响应门禁、材料表、运行源码树和全部输出的 SHA-256；不完整、过期、被篡改或失败运行降级为 `forensic_manifest_only`。

## 一键数值 smoke

```bash
cd /home/user/work/159/jax-fem
conda activate jax-fem-env
bash 159_local/v06/run_smoke.sh
```

也可指定一个**尚不存在**的目录；runner 以原子 `mkdir` 拒绝复用任何已有目录，避免并发运行或旧 VTU 污染审计：

```bash
OUT_ROOT=/home/user/work/159/output/v06_smoke_new_id \
  bash 159_local/v06/run_smoke.sh
```

该脚本使用 6-TET 单位立方体与零激光功率，只验证驱动、无源不变量、release、全历程审计、XRD 算子和 provenance 的端到端连通性。它不会激发真实 LPBF 热循环，也不构成物理准确率证据。

主要输出：

- `solver_command.txt`、`used_config.json`：确切命令与有效参数；
- `step_*.vtu`、`release.vtu`：含 v06 张量状态的场结果；
- `v06_run_audit.json`：全瞬态不变量、网格和加权 QoI；
- `xrd_operator_smoke.json`：完整覆盖的合成 gauge 算子结果；
- `thermal_energy_ledger.jsonl`、`thermal_energy_ledger_summary.json`：逐热步离散弱式账本与门禁汇总；
- `v06_response_gate.json`：零输入/非零响应语义和输入哈希；
- `v06_manifest.json`：完整运行的 `run_status=complete_valid`、`claim_level=numerical_smoke_only`；
- `profile.json`：含 v06 身份与运行边界的时间分解。

已验证输出示例：`/home/user/work/159/output/v06_smoke_runner_09`。其中 3 个热步的组装恒等式误差最大约 `4.5e-18 J`，账本和 manifest 均为 complete；由于零热载荷，预测为零 microstrain，这只是 invariant/operator smoke。

## 非零制造解 smoke

```bash
cd /home/user/work/159/jax-fem
conda activate jax-fem-env
bash 159_local/v06/run_nonzero_smoke.sh
```

该脚本使用 0.01 W、扩展热源和 `solidus=liquidus=0` 的**宏观直接固化制造设定**，专门检查非零耦合链，不代表真实熔池。已验证输出 `/home/user/work/159/output/v06_nonzero_smoke_04` 同时得到：`Tmax=300.0012207 K`、约 `483 Pa` 的受约束质量门控应力、约 `383 Pa` 的 release 应力、`4.61e-12 m` release 位移、非零 XRD 投影以及 `2.12e-7 J` 实际沉积热；响应门禁和 manifest 均通过。

## 已暴露的真实数值问题

旧试跑 `/home/user/work/159/output/v06_smoke_cube_01` 在零激光首步由 1100 K 激活，却达到 1772.5519 K。新审计加 `--source-free-upper-bound 1100` 后会明确失败：首步无效，3 个节点超上界。该结果说明一致质量矩阵/时间步组合存在非单调振荡，不能用末步看似正常来掩盖。

当前账本已经证明这些试跑与**当前离散弱式**一致，但不能把表观热容项说成严格焓守恒。另一个非零探针 `/home/user/work/159/output/v06_nonzero_smoke_02` 在粗网格局部热源下产生约 0.33 K 的节点下冲并被门禁拒绝，说明真实热源运行必须满足热源尺度网格分辨率和时间/质量矩阵收敛要求。

## 网格审计

```bash
cd /home/user/work/159/jax-fem
PYTHONPATH=159_local conda run -n jax-fem-env \
  python -m v06.verification.mesh_audit \
  /home/user/work/159/schema/0119_c3d4_only.inp \
  --output /home/user/work/159/output/v06_verification/full91_mesh_audit.json
```

full91 网格含 197,266 个 TET4，无翻转或零体积单元，但最差质量约 `8.89e-5`、最长/最短边比约 1082；原 1068 MPa 峰值正位于该最差单元。因此论文不得使用该全局峰值，必须先重网格，且正式 gauge 内不能靠删除 sliver 留下空间空洞。

## Strantza 2018 对比边界

当前已具备：C45 元数据、附着态/温度/配准门禁、显式 specimen xyz、VTU 弹性应变输入、microstrain 单位和覆盖率诊断。

尚缺：

1. 完整 C45 bridge CAD/网格与可验证的 mesh→specimen 配准；
2. 作者逐点实验数据，或对 Figure 4 黑色实验点进行带像素/配准不确定度的数字化；
3. 论文实际使用菱形/斜平行六面体衍射体积；当前算子只实现矩形 box，因此不得声称几何完全等价；
4. 无应力晶格参数与相关实验不确定度的协方差；
5. 冻结参数后的附着态仿真；
6. 将 gauge 预测与六个文字锚点做 `manual_unverified_screening`，再在取得带 ID/坐标/哈希的数据后解锁正式指标。

来源：[Strantza et al. 2018 DOI](https://doi.org/10.1016/j.matlet.2018.07.141)，[NIST 全文](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=925448)，[Strantza et al. 2021 OSTI 全文](https://www.osti.gov/servlets/purl/1785480)。

## 当前科学缺口

- 离散弱式账本已接通，但严格焓状态、质量集总/单调格式及激活质量携带的参考焓仍未完成。
- 温度相关热膨胀仍是 `alpha(T)·ΔT`，尚未改为积分/增量形式。
- `eps_ref/eps_p` 已写入 VTU，但可恢复的二进制 checkpoint/restart 仍缺失。
- 显式基板、支撑与 EDM 切割过程以及 Winkler 降阶误差尚未建立。
- Bayat 标定原始数据与 Strantza held-out 逐点数据尚未取得。
- Strantza 菱形衍射体积、约 2000 个 gauge 的空间索引和位置不确定度传播尚未实现。
- full91 必须重网格后才能进入论文展示。

在这些门槛通过前，正确表述是“v06 数值验证框架/预验证模型”，不是“经过实验验证的论文级模型”。

## 测试

```bash
cd /home/user/work/159/jax-fem
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  conda run -n jax-fem-env python -m unittest discover \
  -s tests -p 'test_v06*.py' -v
```
