# legacy/ — 冻结的历史层栈（原 159_local/）

**状态：冻结只读。不可运行。不要修改，不要在新代码中 import。**

本目录是 2026-07 重构（RESTRUCTURE_PLAN.md 步骤⑧⑨）时从 `159_local/` 原样迁移的
v01–v06 迭代层栈与杂项脚本，仅作为历史记录与出处（provenance）留存。
原 `159_local/README.md` 保留为本目录的 `README_159_local.md`。

- 目录内文件的 **import 路径 / 脚本内相对路径未修复**，直接运行大概率失败，这是预期行为。
- 各版本 README / 报告中提到的路径仍指向迁移前的 `159_local/...` 布局，按历史文本保留。
- 活代码（当前生效的求解器、物理内核、验证工具）已全部收敛到 **`jax_fem_am/`**；
  基准与冒烟入口在 **`cases/`**，实验脚本在 **`experiments/`**，测试在 **`tests/`**。
- `tests/regression/` 中的回归测试仍按路径加载本目录内的历史文件
  （v02/v03/v05 与 v01 后处理），用于锁定历史行为；除此之外不应有任何消费者。

| 子目录 | 内容 |
| --- | --- |
| `v01/` | 初版 inp 热应力脚本与四边形应力后处理 |
| `v02/` | Ti-6Al-4V 材料包 runner 与升级版热应力脚本 |
| `v03/` | macro intersection mech100 + XLA 架构（残留 XLA/test.py/文档） |
| `v04/` | bench_mech100_xla 基准与 fastscan 脚本（wrapper 已提炼为 `jax_fem_am/simulation/acceleration.py`） |
| `v05/` | 边界应力分析与 flash/track 对比 |
| `v06/` | v06 残留（README、kp ladder 脚本、validation 包 `__init__`）；活体已迁 `jax_fem_am/` 与 `cases/` |
| `thermal_mechanical/`、根部脚本 | 早期冒烟与扫描入口 |
