# v01-v07 → jax_fem_am 重构迁移清单

日期：2026-07-21。依赖图由 grep/AST 扫描生成（见 §4 补丁台账）。
原则：**行为保持性搬迁**（不夹带数值修改）；shim **一次性硬切**（全部路由改到新包，
`159_local` 不留转发）；每步过三道等价性门（§6）。

## 1. 存活集 vs 遗产集

只迁移"活代码"（当前 phase2/bench 链路可达 + 测试可达），其余冻结进 `legacy/`：

| 集合 | 文件 | 去向 |
|---|---|---|
| 存活 | v03/am_thermal_stress_macro_intersection_mech100.py（3153 行） | 拆解迁移（§2） |
| 存活 | v04/am_thermal_stress_macro_intersection_mech100_XLA.py（3276 行） | 解体迁移（§2） |
| 存活 | v06/ 全树（driver/j2/lifecycle/verification/measurement/provenance ≈3300 行） | 迁移 |
| 存活 | v07/pardiso_variants.py（296 行）＋ bench 三件套 | 迁移/实验区 |
| 存活 | v01/inp_initial_guess_smoke.py（读网格 208 行，被 8 个脚本引用） | mesh/readers.py |
| 存活 | v06/benchmarks/kaess_2023/（网格生成器、run 脚本、inp） | cases/kaess_2023/ |
| 遗产 | v01 其余 3 个脚本、v02 全部（≈6900 行）、v03 XLA(322)、v05(460) | legacy/ 冻结只读 + README |

## 2. 逐文件映射表（存活集 → 新结构）

| 现位置 | 内容 | 新位置 |
|---|---|---|
| v01/inp_initial_guess_smoke.py | read_tet4_inp / compact_mesh / orient_tet4 | jax_fem_am/mesh/readers.py |
| v03 基模块 | read_solid_inp / read_inp_cell_set | jax_fem_am/mesh/readers.py |
| v03 基模块 | PropertyTable / load_property_tables | jax_fem_am/materials/tables.py |
| v03 基模块 | 相态常量 / mechanics_material_quads / 粉末弱固体覆盖 | jax_fem_am/materials/phases.py |
| v03 基模块 | stress_fn 弹性分支 | jax_fem_am/materials/elasticity.py |
| v03 基模块 | stress_fn J2 径向返回（legacy 版） | jax_fem_am/materials/j2.py（与 v06 版合并，v06 状态安全版为准） |
| v06/mechanics/j2.py + lifecycle.py | radial_return / PlasticState / 参考应变更新 | jax_fem_am/materials/j2.py + domain/state.py |
| v03 基模块 | TransientThermal | jax_fem_am/physics/thermal.py |
| v03 基模块 | ThermoMechanical（含 B-bar universal kernel、_u_grads） | jax_fem_am/physics/mechanics.py |
| v03 基模块 | B-bar 数学（平均膨胀修正） | jax_fem_am/mesh/quadrature.py |
| v03 基模块 | apply_thermal_mass_lumping | jax_fem_am/mesh/quadrature.py（改经 fe.py `quadrature_rule` 正式参数，废除运行时改 shape_vals） |
| v03 基模块 | make_kaess/box 位置函数、边界选择 | jax_fem_am/mesh/model.py |
| v03 基模块 | 激光高斯源 set_params 部分 | jax_fem_am/process/heat_source.py |
| v03 基模块 | 扫描路径读取/前沿坐标 | jax_fem_am/process/scan_path.py |
| v03 基模块 | 激活分类（layer_on_scan/intersection/moving window） | jax_fem_am/process/activation.py |
| v03 基模块 | recoat/冷却/终冷调度 | jax_fem_am/process/schedule.py |
| v03 基模块 | update_phase_reference_and_eqp | jax_fem_am/domain/events.py |
| v03 基模块 | run_mechanics / run_mechanics_with_cutback | jax_fem_am/solvers/nonlinear.py |
| v03 基模块 | 主时间循环（main() 后半） | jax_fem_am/simulation/stepper.py |
| v03 基模块 | release（锚箱/锯切/去粉/求解） | jax_fem_am/physics/release.py |
| v03 基模块 | save_step / make_quad_stress_cell_infos | jax_fem_am/io/vtu.py |
| v03 基模块 | build_parser/read_config（600 行 CLI） | jax_fem_am/config/schema.py + loaders.py（case.yaml 化） |
| v04 XLA | install_solver_patch / linear options 重写 | jax_fem_am/solvers/registry.py + linear.py |
| v07/pardiso_variants.py | VariantSolver（phase23/回代/iparm12 转置） | jax_fem_am/solvers/pardiso.py |
| v04 XLA | ProfilingReport / 分层计时 | jax_fem_am/simulation/profiling.py |
| v04 XLA | 激活/掩码缓存（cached_layer_on_scan 等 8 处 patch） | 并入 process/activation.py（显式缓存参数） |
| v04 XLA | 热学-only 力学 surrogate | jax_fem_am/physics/mechanics.py（显式工厂函数） |
| v06/driver.py | StateSafeThermoMechanical → 类替换补丁 | 消失：physics/mechanics.py 直接以 materials/j2.py 为本构 |
| v06/driver.py | REGISTRY 全局态 | jax_fem_am/domain/state.py（ThermalState/MechanicsState/ProcessState dataclass；params[N] 位置下标 → 具名字段） |
| v06/driver.py | 7 处函数替换补丁（§4） | domain/events.py 显式事件处理器 + stepper 显式调用 |
| v06/verification/*（ledger/run_audit/thermal_balance/mesh_quality） | | jax_fem_am/verification/ |
| v06/measurement/xrd*.py | | jax_fem_am/verification/xrd.py |
| v06/provenance.py | | jax_fem_am/verification/provenance.py |
| v06/material_validation.py | | jax_fem_am/materials/tables.py（校验并入） |
| v06/benchmarks/kaess_2023/make_kaess_mesh.py + *.inp + run 脚本 | | cases/kaess_2023/{mesh/, case.yaml, scan_path}；bash 退化为薄包装 |
| v07/bench_{apps,adjoint,scal_ladder}.py + V07 文档 | | experiments/solver/ |
| jax_fem/xla_fem/ 子包 | GPU 装配实验 | experiments/solver/xla_fem/ |
| jax_fem/solver.py 内 pardiso/计时逻辑 | | 移出到 jax_fem_am/solvers/（jax_fem 只留 linear_solver 薄分发钩子） |
| tests/（23 个文件） | | tests/{unit,contract,integration,regression}/ 按性质分箱，import 全部改新路径 |

## 3. jax_fem fork 清理边界

按"只放可回馈上游"标准：保留 universal kernel、newton 选项体系、`linear_solver(A,b,x0,options)`
薄分发钩子、`quadrature_rule` 参数；移出 pardiso 绑定、分层计时细节、xla_fem/。
上游 vendor 基线 = 7966436，divergence 以此文档 §3 为准。

## 4. 补丁台账（每个 monkey-patch 的显式化去向）

v06/driver.py（7 点）：update_phase_reference_and_eqp→events；should_run_mechanics→stepper；
make_quad_stress_cell_infos→io/vtu；solver(ledger 包装)→verification/ledger 显式调用；
ThermoMechanical 类替换→physics 直用 j2；run_mechanics→solvers/nonlinear + events；
save_step→io/vtu；load_property_tables→materials/tables 校验并入。
v04 XLA（14 点）：solver→solvers/registry；8 个激活/掩码缓存→process/activation；
phase_cell/material_state/make_quad_scalar lazy→domain/state；should_*_cached→stepper；
TransientThermal stash+surrogate→physics 工厂。
动态加载链（v06→v04→v03 spec_from_file_location）→ 全部变正常 import。
fe 内部触碰（fes[0].num_quads×110、cells×73、shape_grads×14、shape_vals×3 改性）→
contract 测试锁定的访问器；shape_vals 改性废除（经 quadrature_rule）。

## 5. 硬切路由（用户决定：不留 shim）

- 159_local/ 目录整体消失：存活集进 jax_fem_am/，遗产集进 legacy/（冻结只读，
  README 注明不可运行、仅历史记录，不修其 import）；
- 所有脚本（run_*.sh）、tests/、cases/ 的 import/PYTHONPATH 一次性改到新包；
- 验证：全仓 grep 无 `159_local`、无 `spec_from_file_location`（legacy 除外）。

## 6. 等价性门（每个迁移 commit 必过）

1. 全量测试绿（320，含 9 个 B-bar 测试）；
2. TET4 1 层迷你金标：CPU、固定线程，热学 ledger + 末帧 VTU 与重构前**逐位一致**；
3. HEX8 指纹复测：step-200 压力比值 ≈0.95 不变；
4. 收尾主功能调用冒烟：mesh 读取（TET4/HEX8）→ thermal 单步 → mechanics+B-bar 单解 →
   release → verification 各入口，均从新包 import 调用成功。

## 7. 迁移顺序（每步一个 commit）

① 包骨架 + config schema；② mesh/ + materials/（叶子）；③ solvers/（注册制，废 v04 solver patch）；
④ physics/（thermal/mechanics/B-bar/release）；⑤ domain/ + process/（状态与事件显式化）;
⑥ simulation/（stepper/runner/profiling，废 v06→v04 加载链）；⑦ io/ + verification/；
⑧ cases/ + experiments/ + tests/ 分箱与路由硬切；⑨ legacy/ 冻结 + 全仓路由验证。
