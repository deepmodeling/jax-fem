# 阶段 2 提速决策树（HEX8+B-bar，当前 ~44s/步 → 目标 ≤5s/步）

写于 2026-07-21，冒烟 profile.json 出来前。回来后按 §1 的实测分解走对应分支。
背景数据：TET4 生产 ~4s/步；HEX8 冒烟 ~44s/步；GPU 平台切换在此规模实测负收益
（120s/步，GPU 利用率 4%，见 kaess_2023_benchmark_plan.md 2026-07-21 条目）。

## 1. 先读 profile.json，把 44s 分解到四桶

| 桶 | 来源 | 预期占比假设 |
|---|---|---|
| A 热学装配（每步） | universal/laplace kernel jit 执行 | ? |
| B 热学 pardiso（每步） | 33k dofs，27 点 stencil 数值重分解 | ? |
| C 力学装配（每20步摊销） | B-bar universal kernel jacfwd 24 通道 × Newton 迭代 | 嫌疑最大 |
| D 力学 pardiso（每20步摊销） | 98k dofs HEX8 填充 × Newton 迭代 | 嫌疑第二 |

注意：profile 的分层计时在 v04 ProfilingReport 里；若粒度不够，用
jax_fem solver 的 per-Newton timing 日志（local/global/linear 三行）补。

## 2. 分支

### 分支 C 主导（力学装配贵）→ 两步走
1. **B-bar 切线解析化/降宽**：残差 R = R_dev(u) + R_vol(θ̄(u))，B-bar 修正对切线的
   贡献是 rank-1 结构（θ̄ 对 24 个 dof 的行向量外积）。可以：
   a) 只对 dev 部分走 jacfwd，vol 部分手写 K_vol = (∫p'dV)·(avg_grad ⊗ avg_grad)·V；
   b) 或整体手写小应变 J2 一致切线（radial return 的 C^ep 有闭式），彻底去掉 jacfwd。
   预期：装配降 2-5x。风险：切线错 → Newton 变慢但结果不变（残差是对的），
   用"迭代数不变 + 收敛路径一致"验收。
2. **modified Newton**：见分支 D 的同一机制，装配次数也随之降。

### 分支 D 主导（力学分解贵）→ modified Newton / phase33 复用
- 每次力学求解只在第 1 迭代做数值分解（phase 22），后续迭代 phase 33 纯回代；
  停滞守卫：残差下降率 < 0.5/迭代 时重分解一次（Abaqus 同款策略）。
- 已有基建：pardiso_variants.VariantSolver 的 backsolve 捷径（wave 工况 199/200
  复用验证过）。需要把"矩阵没变"的判据换成"允许矩阵变了但仍用旧分解"。
- 预期：Newton 5-10 迭代 → 1-2 次分解，力学 solver 降 3-6x。
- 数值影响：解不变（收敛判据不动），只是路径变长 1-2 迭代;j2 停滞底(~2e-5)
  不受影响（5e-5 生产容差本来就在其上）。

### 分支 A/B 主导（热学侧贵）→ 两个便宜活
- 热学矩阵 pattern 不变、值随 k(T)/激活变——试 lagged conductivity：隔 N 步才
  重分解，中间步 phase 33 回代 + 残差校验兜底；
- 激活不变的稳态段（recoat/冷却大 dt 步）直接复用分解。

### 通用兜底（与上面正交，最后做）
- mechanics-every 从 20 放宽的敏感度实验（experiments/sensitivity/）;
- 边距粉末单元的力学自由度缩减（弱固体 E=10GPa 仍是全刚度装配对象，
  但它占 17,728/29,568 单元——若 C/D 主导，考虑粉末子矩阵静态凝聚或
  粗化边距网格【偏差需记档，因为破坏逐单元同构】——放最后,先不动奇偶性）。

## 3. 不做清单（已实证或已决策）
- GPU 平台切换：此规模负收益，不再试;留 AM-Sim 197k+ 生产 lane。
- TET10 / 节点平均 TET4：锁定已由 HEX8+B-bar 解决。
- 降 quadrature：HEX8 单点=沙漏，不可行。

## 4. 实施位置
重构后代码就位于 jax_fem_am/solvers/{pardiso.py,nonlinear.py}——分支 D 的
modified Newton 落 nonlinear.py（Newton 策略参数化），分支 C 的解析切线落
physics/mechanics.py + mesh/quadrature.py。先重构后提速（任务 #5 → #6),
避免在旧结构上写一遍、搬家再改一遍。
