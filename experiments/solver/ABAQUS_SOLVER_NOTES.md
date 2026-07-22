# Abaqus/Standard 求解器机制调研笔记（对照我们的 jax-fem/jax_fem_am 求解器）

写于 2026-07-21。背景：复刻 Kaess 2023（LPBF 悬臂梁，C3D8/DC3D8，弱固体粉末
E=10 GPa / σy=1 MPa）时的两个卡点：
①J2 径向返回切线/残差失配 → Newton 停滞底 ~2e-5（生产容差被迫放宽到 5e-5）；
②B-bar + 近理想塑性粉末（H ≈ 0.001·E）→ 切线近奇异，Newton 停在 1e-3 量级，
而 Abaqus 同一单元/材料组合能跑完。

资料来源以 Abaqus 6.10/6.11 文档为准（章节编号随版本略变，机制在 2023+ 版本未变）：
- 理论手册 (Theory Manual, stm)：2.2.1、2.2.2、3.2.3、3.2.4、4.2.2、4.3.2
- 分析用户手册 (Analysis User's Manual, usb)：6.1.4、7.1.1、7.2.2、7.2.3
- User Subroutines Reference Manual：UMAT (1.1.4x)
- 论文：Simo & Taylor 1985 (CMAME 48:101–118)；Matthies & Strang 1979；
  Nagtegaal, Parks & Rice 1974；Michaleris 2014 (FEAD 86:51–60)
每节按「Abaqus 机制 → 我们的现状 → 差距 → 优化建议」组织。

---

## 1. C3D8 的选择性缩减积分（B̄）与 hybrid 单元适用条件

**Abaqus 机制**（理论手册 3.2.4 "Solid isoparametric quadrilaterals and hexahedra"）
- C3D8 名义上"全积分"（2×2×2 Gauss），但对一阶全积分单元，**各 Gauss 点的体积
  变化被替换为单元平均体积变化**——即选择性缩减积分 / B̄ 技术（strain-displacement
  矩阵被修改），引用 Nagtegaal, Parks & Rice (1974)。有限应变形式的确切公式：
  - 修正变形梯度 `F̄ = (J̄/J)^{1/n} F`，n=3（3D）；`J̄ = ∫_Vel J dV / ∫_Vel dV`
    为单元平均 Jacobian（Gauss 点体积比被单元平均体积比替换）；
  - 修正变形率 `ε̇̄ = ε̇ + (1/n)(J̄̇/J̄ − J̇/J) I`（偏量部分逐点、体积部分取单元平均）；
  - 该修正式**同时用于虚功方程（由应力算节点力，即残差）**——残差与切线共用同一 B̄。
- 文档明确：这类单元"do not lock with almost incompressible materials"
  （Getting Started 4.1 "Element formulation and integration"），且理论手册 3.2.4
  指出：大应变塑性、极限载荷分析等有应变间断的问题**推荐一阶单元**（C3D8 系）。
- **Hybrid（C3D8H 等，理论手册 3.2.3）**：压力作为独立插值变量，经 Lagrange 乘子
  以增广变分原理与位移耦合（严格说是 mixed formulation）；引入小压缩因子 ρ 避免
  方程求解器困难。适用条件（GSA "Hybrid elements" 章）：**完全不可压（ν=0.5）
  或 ν>0.475 的近不可压弹性材料**；对 deformation plasticity/流动主导问题文档
  推荐"selectively reduced integration 或 hybrid"。
- 关键结论：**金属 J2 塑性（塑性等容 + 弹性体积模量有限）在 Abaqus 里用普通
  C3D8 的 B̄ 即可，不需要 hybrid**——hybrid 针对的是弹性体积模量趋于无穷的
  情况。Kaess 用 C3D8 而非 C3D8H 与文档推荐一致。

**我们的现状**：B-bar universal kernel（θ̄ 单元平均体应变），jacfwd 自动一致切线，
残差与切线同源。公式上与 Abaqus 的小应变 B̄ 等价。

**差距**：单元格式本身没有差距；差距在"同组合 Abaqus 能跑完"的原因不在单元，
而在下文 §3/§4 的收敛判据与增量机器（Abaqus 的 0.5% 时间平均力容差远松于我们的
5e-5 相对残差，且有线性收敛 fallback 到 2%）。

**优化建议**：不引入 hybrid（我们的问题不是弹性不可压锁死）；保持 B̄；把精力放在
§2 的解析切线与 §3 的判据体系上。

---

## 2. J2 radial return 的一致切线（CTO）闭式表达

**Abaqus 机制**
- Simo & Taylor, "Consistent tangent operators for rate-independent
  elastoplasticity", CMAME 48 (1985) 101–118：切线算子必须与**增量积分算法**
  （closest-point projection / radial return）一致，而非与连续率方程一致，
  Newton 才保持二次渐近收敛；用连续介质切线（continuum/elastoplastic tangent）
  配 radial return，收敛退化为线性，迭代数显著上升（该文数值算例即演示此差异）。
- 闭式 CTO（J2 各向同性硬化 + radial return；Simo & Hughes《Computational
  Inelasticity》Box 3.2 记法；Abaqus 理论手册 4.3.2 "Isotropic elasto-plasticity"
  给出等价的显式 material stiffness，并强调该模型"the material stiffness matrix
  can be written explicitly … particularly efficient code"，**无需矩阵求逆**）：

  ```
  n̂ = s_tr / ‖s_tr‖ ,  Δγ 由标量径向返回方程解出
  β  = 1 − 2G·Δγ/‖s_tr‖                    （半径收缩因子）
  γ̄  = 1/(1 + H'/(3G)) − (1 − β)
  C_ep = K·1⊗1 + 2G·β·(I_dev) − 2G·γ̄·n̂⊗n̂
  ```

  与连续切线的差别正是 β 中的 `2GΔγ/‖s_tr‖` 项（增量越大差别越大）。
  理想塑性（H'=0）时 γ̄ = β，C_ep 沿 n̂⊗n̂ 与偏量方向的刚度都由 β 控制，
  **矩阵不奇异（K·1⊗1 恒在），但偏量特征值 ~2Gβ、n̂ 方向 ~0，条件数 ~K/(GΔγ 相关项)
  随增量塑性应变增大而恶化**——这是 ② 的机理注脚。
- Abaqus 对切线不精确的官方立场（UMAT 文档，Sub. Ref. Manual）：
  "An incorrect definition of the material Jacobian **affects only the convergence
  rate; the results (if obtained) are unaffected**"——前提是它的收敛判据能在
  线性收敛速率下把解"收下来"（见 §3 的 2% fallback）。

**我们的现状**：jacfwd 穿过 radial return 求切线。若残差函数与被微分函数是同一段
代码，jacfwd 给出的就是精确算法一致切线——理论上不该有 2e-5 停滞底。

**差距/诊断**：停滞底说明**切线与残差实际上不同源**。常见来源（按嫌疑排序）：
(a) 弹/塑分支或 return-mapping 内层迭代在 jacfwd 下的自定义 VJP/展开与残差路径
不一致（如 `lax.cond`/`stop_gradient`/定次数内迭代未收敛到微分点）；
(b) B̄ 的 θ̄ 平均在残差与切线中用了不同的积分权重路径；
(c) float32/混合精度导致的微分噪声底。Abaqus 侧没有此问题是因为解析 CTO 与残差
按同一组积分式（理论手册 4.2.2 的变分推导）构造，一致性是构造出来的。

**优化建议**
1. 用中心差分 directional derivative 校验装配级切线：`‖(R(u+εv)−R(u−εv))/2ε − K v‖ / ‖Kv‖`
   在 ε 扫描下应出现 V 形谷；谷底若 ~2e-5 即坐实失配并可定位到单元/材料级。
2. 落地 SPEEDUP_DECISIONS 分支 C-1b：**手写上式闭式 C_ep**（Abaqus 4.3.2 同款），
   B̄ 修正解析化（体积块 rank-1 结构）。一石二鸟：去掉 jacfwd 装配开销（预期 2–5x）
   且从根上消除失配。验收：FD 校验谷底降到 ~1e-9（fp64），Newton 恢复二次收敛段。

---

## 3. Abaqus/Standard 收敛判据体系（对比我们的单一相对残差）

**Abaqus 机制**（usb 7.2.3 "Convergence criteria for nonlinear problems"、7.2.2）
- **力残差判据**：`r_max^α ≤ R_α^n · q̄^α`，其中 r_max 是场 α 的**最大逐自由度残差
  （max-norm，不是 L2）**，q̄^α 是该场"**时间平均通量（力）**"：每次迭代重算的
  空间平均单元节点力+外力的量值均值，再对本步各增量做时间平均。默认
  `R_α^n = 5×10⁻³`（即"0.5% 的时间平均力"，文档称之 rather strict by
  engineering standards）。
- **位移修正判据**：`c_max^α ≤ C_α^n · Δu_max^α`，最大修正对最大增量位移，
  默认 `C_α^n = 10⁻²`；或用外推的"再迭代一次的估计修正"满足同式即可。
  两个判据**同时满足**才收敛。
- **空间平均只计"活动"区域**：若整体平均通量远小于活动区平均（AM 场景：大量
  quiet/粉末区通量近零），判据会过严；Abaqus 用 `ε=10⁻⁵` 阈值把低通量自由度
  标记 inactive，**只在活动区域算空间平均力**——对逐层激活的 AM 模型这是内建的
  "分区归一化"。
- **零通量增量**：`q̃^α ≤ ε·q̄^α` 时改用 `r_max ≤ R_α^ε·q̄`（R_α^ε=10⁻⁸）与
  `c_max ≤ C_α^ε·Δu_max`（C_α^ε=10⁻³）。
- **线性增量**：`r_max ≤ R_α^l·q̄`（R_α^l=10⁻⁸）→ 一次迭代直接接受，不查修正。
- **非二次收敛 fallback（对我们最关键）**：若迭代若干次后（msg 行
  "EQUIL. ITER. AFTER WHICH ALTERNATE RESIDUAL IS USED"，默认 9）收敛速率
  只有线性，**残差容差自动放宽到 `R_α^P = 2×10⁻²`**（但仍须过位移修正判据）。
  即：Abaqus 对"切线不精确/病态导致的线性收敛"的官方答案是**换一档更松的
  残差容差 + 用位移修正判据兜底解的质量**。quasi-Newton 步不启用此逻辑。
- **对数收敛率检测**：从第 I_R=8 次平衡迭代起，用对数收敛率外推所需总迭代数，
  预计超过 I_C=16 → 放弃增量并 cutback（因子 0.5，见 §4）。
- **发散检测**：从第 I_0=4 次迭代起，最大残差连续两次上升 → 放弃增量
  （cutback 0.25）。
- **线搜索**：默认仅在 quasi-Newton 步激活（N_ls=5）；纯 Newton 默认关闭
  （N_ls=0）。算法：沿修正方向最小化残差在该方向的分量，标量因子有界，
  精度要求宽松（默认残差方向分量降到 0.25 即停，最多 5 次内层残差评估，
  不动刚度阵）。可手动开启并加强（N_ls=10、更严容差）。

**我们的现状**：单一全局相对残差 `‖R‖/‖R₀‖ ≤ 5e-5`（L2、全模型统一归一），
自研线搜索常开。

**差距**（这是 ①② 两个卡点在"验收层"的直接解释）
1. 归一化基准不同：我们用初残差归一——增量初始残差大时判据松、初始残差小时
   （AM 小时间步、大量粉末区）判据可能严到物理噪声以下；Abaqus 用时间平均力
   归一，是跨增量稳定的"物理量纲"基准，且只在活动区平均。
2. 范数不同：max-norm 逐自由度 vs 我们的全局 L2——L2 会被海量近零粉末自由度
   稀释或（反向）被单点病态支配。
3. 无第二判据：我们没有位移修正判据，也就没有"残差停滞但解已不动了"的
   合法收敛出口——Abaqus 的 2e-2 fallback + c_max 判据正是为此设计。
   按 Abaqus 标准，我们停在 2e-5/1e-3 的两个状态**大概率早已"收敛"**。

**优化建议**（优先级见文末）
- 实现 Abaqus 式双判据：时间平均活动区力归一的 max-norm 残差判据（0.5%）
  + 位移修正判据（1%）；保留现有 5e-5 相对残差作为"严格模式"回归开关。
- 加"线性收敛 fallback"：检测到收敛率线性（连续 k 次残差比 ≈ 常数<1）后
  切换到 2% 档 + 修正判据兜底——①② 立即不再是失败模式，而是正常收敛路径。
- 发散/对数收敛率外推接入现有 2/4/8 自动切割的触发逻辑（见 §4）。

---

## 4. 自动增量与 cutback 规则、automatic stabilization

**Abaqus 机制**
- 增量控制（usb 7.1.1 + 7.2.2 msg 参数表，默认值）：
  | 规则 | 默认 |
  |---|---|
  | 放弃增量：>16 次平衡迭代不收敛，或发散检测触发 | I_C=16 |
  | 发散后 cutback 因子 | **0.25** |
  | "收敛太慢"（对数收敛率外推超限）cutback 因子 | **0.50** |
  | "平衡迭代过多"（>10 次收敛）→ 下个增量缩 | **0.75** |
  | 单元反转/材料积分失败重试因子 | 0.25 |
  | 连续 2 个增量 ≤4(5) 次迭代收敛 → 增量放大 | ×1.5 |
  | 单增量最大 cutback 次数 | 5 |
  即：**cutback 因子按失败原因分档（0.25/0.5/0.75），并有对称的增量放大通道**。
- **Automatic stabilization**（usb 7.1.1 "Automatic stabilization of unstable
  problems"，`*STATIC, STABILIZE`）：向全局平衡方程加体积比例粘性力
  `F_v = c·M*·v`（M* 为单位密度人工质量阵，v=Δu/Δt）。damping factor c 由
  **耗散能比**标定：使一个典型增量的粘性耗散 ≈ 外推应变能的
  `2.0×10⁻⁴`（默认 dissipated energy fraction）。
  - **Adaptive 方案默认随之激活**：c 可随时间与空间（逐单元）调整，由收敛历史
    （SDI/平衡迭代数、cutback 次数）驱动增大、失稳消退后减小；约束条件是
    粘性耗散能/应变能比 `ALLSD/ALLIE ≤ ALLSDTOL = 0.05`（默认），全局与逐单元
    双重限制。
  - 文档要求事后核查 VF（粘性力）与 ALLSD/ALLIE 确保阻尼未污染解。
  - 首增量本身不稳/奇异（刚体模态）时：按"平均单元阻尼阵/步长 ≈ 平均单元刚度
    ×耗散能比"直接给初始 c——**即用人工阻尼直接补奇异刚度**，并发警告。
- 对弱材料大变形/失稳这是 Abaqus 用户的标准手段；LPBF 悬臂梁类问题若粉末近
  理想塑性造成局部失稳（塑性坍缩=材料软化类失稳），STABILIZE 把停滞变为
  可收敛的准静态粘性路径。这很可能是"Abaqus 同组合能跑完"的第二支柱
  （第一支柱是 §3 的判据；Kaess 是否显式用 STABILIZE 未能从公开文本核实——
  **未核实条目**，但即使不用，0.5%+2% 判据档也足以解释）。

**我们的现状**：增量自动切割（2/4/8 子步，单一因子 0.5 语义）、线搜索；
无稳定化/阻尼；无增量放大通道（步长由热学时间步驱动，力学每 20 步）。

**差距**：cutback 只有"对半切"，无失败原因分档、无对数收敛率外推提前放弃
（我们会把 16+ 次迭代打满才切）；对近奇异切线没有任何正则化手段。

**优化建议**
- 把发散检测（连续两次残差上升，从第 4 次迭代起）与对数收敛率外推（第 8 次起）
  接入子步切割：提前止损，省掉白打的迭代（直接复用 Abaqus 的 4/8/16 与
  0.25/0.5 参数作为起点）。
- 针对 ②：实现**最小版 adaptive stabilization**——对力学步加
  `c·diag(M*)/Δt` 或 Levenberg 型 `λ·diag(K)` 正则；c 由首迭代应变能 ×2e-4
  标定，按 ALLSD/ALLIE≤5% 自适应衰减。对近理想塑性粉末，这等价于把切线
  最小特征值抬离零，Newton 停滞 1e-3 的病直接对症。
  （替代/并行方案：给粉末塑性加率相关正则——Perzyna/overstress 幂律，
  Abaqus `*RATE DEPENDENT` 同款，物理上也更接近粉末真实行为。）

---

## 5. 直接稀疏求解器对病态/近奇异矩阵的处理

**Abaqus 机制**（usb 6.1.4 "Direct linear equation solver" + msg/dat 警告体系）
- 求解器为 sparse 直接 Gauss 消去、multifront 技术；文档定位是"求出线性方程组
  的精确解（到机器精度）"。**官方文档不暴露 pivot 扰动参数**（不同于 PARDISO 的
  CNR 扰动 iparm(10)）；对称系统消去中监测 pivot，异常时不中止而是发警告并继续：
  - `WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE ...
    D.O.F. ... RATIO = ...`（pivot 与对角期望值之比过大，典型为欠约束/刚体模态、
    或极端刚度反差——AM 粉末场景常见）；
  - `ZERO PIVOT` 解算失败类消息；
  - `THE SYSTEM MATRIX HAS N NEGATIVE EIGENVALUES`（矩阵不定，失稳路径上常见，
    配合 §4 的 cutback/stabilize 机器消化）。
- 行为哲学：**解照常往下走，把病态信息变成警告+增量机器的触发信号**，而不是
  在线性代数层硬失败。（内部是否做微小 pivot 替换未见官方文档——未核实条目，
  第三方资料一致的说法是"发警告继续"。）

**我们的现状**：MKL PARDISO 直接解，phase23/33 变体；未监控扰动 pivot。

**差距**：我们没有把线性代数层的病态信号（PARDISO iparm(14)=扰动 pivot 数、
残差回代校验）上传给非线性层做决策。

**优化建议**：低成本加两件事：
1. 每次分解读 iparm(14)（perturbed pivots 计数）+ 开 iparm(8) 迭代精化；
   扰动数>0 记入 per-Newton 日志，作为"该步需要 stabilization/重分解"的触发器
   （与 SPEEDUP_DECISIONS 分支 D 的"残差下降率<0.5 重分解"守卫互补）；
2. 对粉末主导步做一次条件数抽检（MKL `?gecon` 或幂迭代估计），归档到 profile，
   为 §4 的正则化强度标定提供数据。

---

## 6. AM 文献中的粉末 / quiet element / element birth 数值处理

**文献机制**
- Michaleris 2014（"Modeling metal deposition in heat transfer analyses of
  additive manufacturing processes", FEAD 86:51–60）系统对比：
  - **quiet**：未沉积单元留在方程组里，属性缩放（热学：热导 ×s_k、比热 ×s_c，
    典型 s ~1e-4 量级）；优点是矩阵结构/编号不变，缺点是**缩放因子过小 →
    全局矩阵病态，过大 → 人工传热/刚度污染解**——精度与条件数是显式 tradeoff；
  - **inactive**：单元完全移出方程组，激活时重编号/重装配；无污染无病态，
    但方程组管理开销大；
  - **hybrid（推荐）**：远未来层 inactive、下一层 quiet、激光到达时激活——
    同时规避病态与重编号开销。
- 力学侧粉末/未激活刚度下限的常见选取：ersatz/quiet 刚度取本体的
  ~1e-3（如 Proell/Wall 系 scan-resolved 框架 arXiv:2302.05164 等），
  经验窗口 1e-2…1e-6，下限由**全局矩阵条件数与直接解算器可容忍度**决定，
  上限由对翘曲变形的伪约束误差决定——文献里这是标定量而非推导量。
- Kaess 2023 的"弱固体粉末"（E=10 GPa ≈ 本体 ~5–10%，σy=1 MPa）属于
  "偏大刚度 + 极低屈服"的 quiet 变体：**弹性缩放对条件数很友好（他们的 E 比
  常见 ersatz 大 1–2 个量级），病态来自 σy=1 MPa + H≈0.001E 的近理想塑性**
  ——大片粉末同时进入塑性平台才是切线近奇异的源头（与 §2 的 β 分析一致）。
- Abaqus 原生支持：`*MODEL CHANGE, REMOVE/ADD`（ADD 默认 strain-free 重激活，
  usb "Element and contact pair removal and reactivation"）；2017+ 的 AM 特化
  接口（progressive element activation / `*ACTIVATE ELEMENTS`）。Abaqus 侧
  quiet 粉末照样参与 §3 的"活动区平均力"判据——粉末区小残差不会拖垮容差基准。

**我们的现状**：粉末为全刚度装配对象（17,728/29,568 单元），弱固体参数照搬
Kaess；激活逻辑自研。

**差距**：无 inactive/hybrid 通道；收敛判据不区分粉末/实体区（§3）；粉末塑性
无正则化。

**优化建议**
- 短期不动拓扑（保 SPEEDUP_DECISIONS 的"逐单元同构"约束），优先做：
  (a) 判据分区归一（§3 的活动区平均即覆盖）；(b) 粉末塑性正则化：H 从 0.1%E
  提到 1–5%E 或加 overstress 率相关（对宏观翘曲/残余应力的影响做一次敏感度
  实验并记档——文献依据：粉末屈服面本身就是标定量）。
- 中期若走分支 D/静态凝聚：hybrid quiet/inactive（Michaleris 2014 结论）是
  文献背书的方向,与"粉末子矩阵静态凝聚"殊途同归。

---

## 7. quasi-Newton（BFGS）与 modified Newton 在 Abaqus 的使用场景

**Abaqus 机制**（理论手册 2.2.1/2.2.2；usb 7.2.3 "Specifying the quasi-Newton method"）
- 全 Newton 是默认，理由是收敛率（2.2.1）；modified Newton（Jacobian 偶尔
  重算或不重算）被评价为"适合温和非线性软化问题（受限塑性、单调加载），
  不适合强非线性"——线性收敛。
- **BFGS quasi-Newton（`*SOLUTION TECHNIQUE, TYPE=QUASI-NEWTON, REFORM KERNEL=n`）**
  按 Matthies & Strang (1979) 实现：kernel = 实际 Jacobian 的**分解**，
  迭代间不重分解，而是对 kernel 前后乘 rank-2 修正（只需向量内积与缩放，
  秩更新不存矩阵）；**默认 8 次迭代才重构 kernel**，收敛顺利时
  **同一 kernel 可跨多个增量复用**；收敛率介于线性与二次之间。
- 适用场景（usb 7.2.3 原文）：系统大、每增量迭代多、或刚度阵迭代间变化不大
  （隐式动力学小步长、局部塑性的小位移分析）；限制：仅对称系统。
- 配套策略：quasi-Newton 步**默认开线搜索**（补偿不精确 Jacobian 的发散风险）、
  **关闭对数收敛率检测**（线性偏慢是预期行为）、迭代计入 cutback 统计时按
  是否重构 kernel 加权。

**我们的现状/计划**：分支 D = "第 1 迭代 phase22 数值分解，后续 phase33 回代，
残差下降率<0.5/迭代时重分解"——这正是 Abaqus 的 REFORM KERNEL 策略骨架
（我们的守卫比它的"固定 8 次+按需"更主动），但没有 rank-2 修正，
等价于纯 modified Newton。

**差距**：modified Newton 线性收敛且没有曲率更新；Abaqus 的经验是 BFGS 修正
+线搜索能显著压低"旧分解"下的额外迭代数。

**优化建议**：分支 D 落地时按 Abaqus 配方走全套：
1. 复用分解（phase33）+ **Matthies–Strang 两回代 BFGS 更新**（每迭代多一次
   回代 + 若干内积，远小于一次重分解；对称正定守恒）；
2. 复用期**强制开线搜索**（我们已有）；
3. 复用期放宽/关闭对数收敛率类提前放弃逻辑，迭代计数加权（防止误触发子步切割）；
4. 重分解守卫保留"残差下降率<0.5"+ 上限 8 次（Abaqus 默认）双条件。
   预期与 SPEEDUP_DECISIONS 一致：Newton 5–10 迭代 → 1–2 次分解。

---

## 8. 对我们求解器的优先级排序建议（对齐 SPEEDUP_DECISIONS 三分支）

判断基准：①②是**正确性/鲁棒性卡点**（挡住基准复刻），三分支是**速度**；
Abaqus 的经验表明卡点的最大头在验收层而非线性代数层。

| 优先级 | 事项 | 对应 | 预期效果 | 成本 |
|---|---|---|---|---|
| **P0** | Abaqus 式双判据：活动区时间平均力归一 max-norm 残差（0.5%）+ 位移修正判据（1%）+ 线性收敛 2% fallback | §3（三分支之外，但为其守门） | ①② 从"失败"变"正常收敛"；5e-5 魔数退役 | 低（纯判据层） |
| **P1** | J2+B̄ 解析一致切线（闭式 C_ep + 体积 rank-1），FD 谷底校验 | 分支 C-1b，§2 | 装配 2–5x；消除切线/残差失配根因 | 中 |
| **P2** | modified Newton 分解复用 + BFGS rank-2 修正 + 线搜索常开 + 重分解双守卫 | 分支 D，§7 | 力学 solver 3–6x（1–2 次分解/步） | 中（基建已有） |
| **P3** | 最小版 adaptive stabilization（λ·diag 正则，2e-4 能量标定，ALLSD/ALLIE≤5%）或粉末 overstress 正则 | §4/§6，对 ② 的第二道保险 | 病态步兜底；P0 后可能不再必需 | 中低 |
| **P4** | 发散检测/对数收敛率外推接入子步切割（4/8/16 迭代、0.25/0.5 因子） | §4 | 止损省时，配合 P2 | 低 |
| **P5** | PARDISO iparm(14) 扰动 pivot 监控 + iparm(8) 精化，作病态遥测 | §5 | 诊断基建，喂 P3 触发器 | 很低 |
| **P6** | lagged conductivity / 稳态段分解复用 | 分支 A/B | 热学侧收益，与 P2 同一机制族 | 低（等 profile 分桶确认 A/B 占比后做） |
| 记档实验 | 粉末硬化 0.1%E→1–5%E 敏感度；判据分区归一后的基准复刻偏差 | §6 | 文献背书的标定自由度 | 低 |

一句话版：**先把"什么叫收敛"改成 Abaqus 的定义（P0），再把切线做对做便宜（P1），
然后才轮到少分解（P2）**；stabilization（P3）作为粉末病态的保险丝，其余是遥测与止损。

---

## 引用清单

- Abaqus Theory Manual：2.2.1 Nonlinear solution methods；2.2.2 Quasi-Newton
  solution technique；3.2.3 Hybrid incompressible solid element formulation；
  3.2.4 Solid isoparametric quadrilaterals and hexahedra；4.2.2 Integration of
  plasticity models；4.3.2 Isotropic elasto-plasticity
- Abaqus Analysis User's Manual：6.1.4 Direct linear equation solver；
  7.1.1 Solving nonlinear problems（automatic incrementation & automatic
  stabilization）；7.2.2 Commonly used control parameters；7.2.3 Convergence
  criteria for nonlinear problems；Element and contact pair removal and
  reactivation（*MODEL CHANGE）
- Abaqus User Subroutines Reference Manual：UMAT（Jacobian 精度只影响收敛率语句）
- Getting Started with Abaqus：4.1 Element formulation and integration；
  Hybrid elements（ν>0.475 适用条件）
- Simo, J.C. & Taylor, R.L. (1985). Consistent tangent operators for
  rate-independent elastoplasticity. CMAME 48(1):101–118
- Simo & Hughes (1998). Computational Inelasticity, Box 3.2（闭式 CTO 记法）
- Matthies, H. & Strang, G. (1979). The solution of nonlinear finite element
  equations. IJNME 14:1613–1626（Abaqus BFGS 实现依据）
- Nagtegaal, Parks & Rice (1974). On numerically accurate finite element
  solutions in the fully plastic range. CMAME 4:153–177（B̄ 依据）
- Michaleris, P. (2014). Modeling metal deposition in heat transfer analyses of
  additive manufacturing processes. FEAD 86:51–60（quiet/inactive/hybrid）
- Proell et al., scan-resolved AM 框架 arXiv:2302.05164（ersatz 刚度 ~1e-3）

**未核实条目**：(1) Kaess 2023 是否显式使用 `*STATIC, STABILIZE`（论文文本未取到，
但 §3 判据差异已足以解释可跑通性）；(2) Abaqus 直接求解器内部是否做 pivot 微扰
（官方文档未公开，第三方资料仅确认"警告+继续"行为）。
