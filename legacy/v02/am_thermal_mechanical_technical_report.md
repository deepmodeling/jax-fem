# 增材制造 3D 粉末打印热-力耦合模拟技术报告

> 适用代码：`am_thermal_stress_upgraded.py`
> 建模对象：基于四面体网格的 3D 粉末/实体连续介质等效增材制造过程模拟
> 求解类型：瞬态热传导 + 准静态力学单向耦合
> 主要改造：扫描路径、材料相态历史、凝固参考温度、粉末/固体/液体物性分流、路径输出与后处理字段

---

## 1. 报告目的

本报告用于说明当前代码中已经考虑的控制方程、数值离散方式、扫描路径生成方法、材料状态更新逻辑，以及热场数据如何传递给力学场。该报告可作为项目说明书、论文方法章节或代码技术文档的基础版本。

当前模型的目标不是进行粉末颗粒级 CFD/多相流模拟，而是建立一种适合工程参数扫描和后续机器学习代理模型训练的 **连续介质等效粉末床热-力耦合模型**。模型把粉末、已凝固实体、熔融液体、糊状区、基板和支撑结构作为不同材料状态，在积分点层面追踪状态并更新热物性和力学响应。

---

## 2. 基本假设

当前代码采用以下建模假设：

1. **连续介质假设**
   粉末床不解析单个粉末颗粒，而是用等效密度、比热、导热系数描述。

2. **单向热-力耦合**
   温度场影响材料状态、热应变和力学响应，但位移场不反过来影响热传导、扫描路径或接触边界。

3. **准静态力学平衡**
   力学场忽略惯性项，不求解动力学波传播，只求解每个指定时间步下的静力平衡。

4. **移动体热源**
   激光能量以高斯平面分布 + 构建方向指数衰减的体热源形式进入域内，不显式求解自由表面吸收、反射、多重散射和匙孔效应。

5. **相变采用温度区间判据**
   通过 solidus temperature 与 liquidus temperature 判断固态、糊状区和液态。潜热通过表观比热法加入热方程。

6. **重熔应力释放**
   材料重新进入液态或糊状区时，可选择重置等效塑性应变；材料重新凝固时写入新的应力自由参考温度。

7. **路径为工艺路径近似**
   目前支持 raster 扫描、蛇形扫描、层间旋转、hatch spacing、layer thickness、jump、dwell、recoat 和 cooling。

---

## 3. 符号说明

| 符号 | 含义 |
|---|---|
| $\Omega$ | 计算域 |
| $\Gamma_D$ | Dirichlet 边界 |
| $\Gamma_N$ | Neumann 边界 |
| $T$ | 温度 |
| $T_0$ | 初始温度或环境温度 |
| $T_{old}$ | 上一时间步温度 |
| $T^n$ | 当前时间步温度 |
| $\rho$ | 密度 |
| $C_p$ | 定压比热 |
| $C_p^{eff}$ | 含潜热修正的表观比热 |
| $k$ | 导热系数 |
| $L$ | 潜热 |
| $T_s$ | 固相线温度 solidus temperature |
| $T_l$ | 液相线温度 liquidus temperature |
| $Q_{laser}$ | 体积激光热源 |
| $Q_{front}$ | 移动前沿体积等效散热项 |
| $P$ | 激光功率 |
| $\eta$ | 吸收率 absorptivity |
| $r_b$ | 光斑半径 beam radius |
| $d_s$ | 热源深度 source depth |
| $h$ | 对流换热系数 |
| $\epsilon$ | 发射率 emissivity |
| $\sigma_{SB}$ | Stefan-Boltzmann 常数 |
| $\boldsymbol{u}$ | 位移向量 |
| $\boldsymbol{\varepsilon}$ | 小应变张量 |
| $\boldsymbol{\varepsilon}_{th}$ | 热应变 |
| $\boldsymbol{\sigma}$ | Cauchy 应力 |
| $E$ | Young's modulus |
| $\nu$ | Poisson's ratio |
| $\alpha$ | 热膨胀系数 |
| $\sigma_y$ | 屈服应力 |
| $H$ | 等向硬化模量 |
| $\bar{\varepsilon}_p$ | 等效塑性应变 |
| $T_{ref}$ | 应力自由参考温度 |

---

## 4. 几何、网格与坐标体系

### 4.1 网格输入

当前代码通过：

```python
raw_points, cells, selected_cells = read_tet4_inp(args.inp, args.max_cells)
points = raw_points * args.mesh_length_scale
mesh = Mesh(points, cells, ele_type="TET4")
```

读取 Abaqus `.inp` 中的 TET4 四面体网格。`mesh_length_scale` 用于统一单位，例如将 mm 坐标缩放到 m。

### 4.2 构建方向与打印平面

用户通过：

```bash
--build-axis x|y|z
--base-side min|max
```

定义构建方向。若：

```bash
--build-axis x --base-side min
```

则表示以 $x_{min}$ 面为基底，沿 $+x$ 方向逐层铺粉/打印；打印平面为 $yz$ 平面。

构建方向符号定义为：

$$
s_b =
\begin{cases}
+1, & \text{base-side = min} \\
-1, & \text{base-side = max}
\end{cases}
$$

计算单元质心在构建方向上的坐标：

$$
x_b^e = \frac{1}{N_e}\sum_{a \in e} x_{b,a}
$$

其中 $x_b$ 为构建轴坐标，$N_e$ 为单元节点数。

---

## 5. 层激活与路径生成方法

## 5.1 有效打印区域

如果网格中包含基板或支撑结构，则实际打印区域不直接等于整体网格范围，而是从基板/支撑之后开始：

$$
\Omega_{part} = \Omega \setminus (\Omega_{substrate} \cup \Omega_{support})
$$

构建方向上的打印起点为：

$$
x_{part,min} = x_{min} + t_{substrate} + t_{support}
$$

当 `base-side = min` 时，层从 $x_{part,min}$ 向 $x_{part,max}$ 推进。

## 5.2 层厚控制

如果提供：

```bash
--layer-thickness h_L
```

则层数由几何厚度自动推导：

$$
N_L = \left\lceil \frac{L_b}{h_L} \right\rceil
$$

其中：

$$
L_b = |x_{part,max} - x_{part,min}|
$$

第 $i$ 层的激活前沿为：

$$
x_f^{(i)} =
\begin{cases}
x_{part,min} + \min((i+1)h_L, L_b), & s_b = +1 \\
x_{part,max} - \min((i+1)h_L, L_b), & s_b = -1
\end{cases}
$$

如果没有提供 `layer_thickness`，则使用 `layers` 将打印厚度均分：

$$
x_f^{(i)} = x_{part,min} + \frac{i+1}{N_L}L_b
$$

## 5.3 单元激活判据

单元是否被激活由前沿位置决定：

$$
a_e^n =
\begin{cases}
1, & s_b(x_f^n - x_b^e) \geq -\delta \\
0, & \text{otherwise}
\end{cases}
$$

基板和支撑结构始终视为 active：

$$
a_e^n = a_{part,e}^n \lor a_{substrate,e} \lor a_{support,e}
$$

在代码中，cell 级 active 会被映射到积分点：

$$
a_{q}^n = \operatorname{expand}(a_e^n)
$$

## 5.4 扫描平面内局部坐标

对于每一层，代码建立扫描方向 $\boldsymbol{e}_s$ 和 hatch 方向 $\boldsymbol{e}_h$。

未旋转时：

$$
\boldsymbol{e}_{s0} = \text{scan-axis unit vector}
$$

$$
\boldsymbol{e}_{h0} = \text{hatch-axis unit vector}
$$

层间旋转角为：

$$
\theta_i = i \cdot \theta_{rot}
$$

则第 $i$ 层的局部方向为：

$$
\boldsymbol{e}_s^{(i)} = \cos\theta_i \boldsymbol{e}_{s0} + \sin\theta_i \boldsymbol{e}_{h0}
$$

$$
\boldsymbol{e}_h^{(i)} = -\sin\theta_i \boldsymbol{e}_{s0} + \cos\theta_i \boldsymbol{e}_{h0}
$$

这使代码可以支持 $0^\circ$、$90^\circ$、$67^\circ$ 等层间旋转扫描。

## 5.5 Hatch spacing 推导

若提供：

```bash
--hatch-spacing h_s
```

则 hatch 偏移由打印矩形投影到 $\boldsymbol{e}_h$ 方向后生成。

先计算打印矩形四个角点相对中心在 hatch 方向上的投影：

$$
\xi_h^c = (\boldsymbol{x}_c - \boldsymbol{x}_{center}) \cdot \boldsymbol{e}_h
$$

得到范围：

$$
\xi_{h,min} = \min_c \xi_h^c, \quad \xi_{h,max} = \max_c \xi_h^c
$$

hatch 数量：

$$
N_h = \left\lfloor \frac{\xi_{h,max}-\xi_{h,min}}{h_s} \right\rfloor + 1
$$

第 $j$ 条 hatch 的偏移量：

$$
\xi_h^{(j)} = \xi_{h,min} + jh_s
$$

若最后一条 hatch 与边界距离过大，则补充边界 hatch。

## 5.6 扫描线裁剪

一条扫描线写为：

$$
\boldsymbol{x}(s) = \boldsymbol{x}_{center} + \xi_h^{(j)}\boldsymbol{e}_h + s\boldsymbol{e}_s
$$

代码将该直线裁剪到打印平面矩形边界内。对每个坐标轴 $m$，满足：

$$
x_{m,min} \leq x_m(s) \leq x_{m,max}
$$

即：

$$
s \in \left[\frac{x_{m,min}-b_m}{e_{s,m}}, \frac{x_{m,max}-b_m}{e_{s,m}}\right]
$$

所有轴约束取交集得到：

$$
s_{start} = \max_m s_{m,min}, \quad s_{end} = \min_m s_{m,max}
$$

若：

$$
s_{start} > s_{end}
$$

则该扫描线与打印区域无交集，跳过。

## 5.7 扫描步长与速度

若启用：

```bash
--auto-scan-steps-from-speed
```

则每条扫描线的步数为：

$$
N_s = \left\lceil \frac{L_s}{v_s \Delta t} \right\rceil + 1
$$

其中：

$$
L_s = |s_{end} - s_{start}|
$$

第 $k$ 个扫描采样点：

$$
\lambda_k = \frac{k}{N_s-1}
$$

$$
s_k = s_{start} + \lambda_k(s_{end}-s_{start})
$$

激光中心：

$$
\boldsymbol{x}_{laser}^{n} = \boldsymbol{x}_{center} + \xi_h^{(j)}\boldsymbol{e}_h + s_k\boldsymbol{e}_s
$$

构建方向坐标强制设置为当前层前沿：

$$
x_{laser,b}^{n} = x_f^{(i)}
$$

## 5.8 蛇形扫描

当 `serpentine = True` 时，奇数 hatch 的起终点反转：

$$
(s_{start}, s_{end}) \leftarrow (s_{end}, s_{start})
$$

该设置减少空行程，能够更接近常见 LPBF raster 工艺。

## 5.9 Jump / dwell / recoat / cooling 状态

代码中的 `StepState` 包含：

```text
global_step, mode, layer_idx, hatch_idx, scan_idx,
laser_center, laser_power, laser_switch, dt,
scan_frac, hatch_frac, front_coord, layer_frac
```

其中 `mode` 取值包括：

| mode | 含义 | 激光 |
|---|---|---|
| scan | 有效扫描 | on |
| jump | hatch 间空行程 | off |
| hatch_dwell | hatch 间停留 | off |
| layer_dwell | 层间停留 | off |
| recoat | 铺粉等待 | off |
| cooling | 打印后冷却 | off |
| release | 释放基底约束后求解 | off |

jump 时间由两条 hatch 之间的距离和 jump speed 推导：

$$
t_{jump} = \frac{\|\boldsymbol{x}_{end} - \boldsymbol{x}_{start}\|}{v_{jump}}
$$

jump 子步数：

$$
N_{jump} = \left\lceil \frac{t_{jump}}{\Delta t} \right\rceil
$$

每个 jump 子步长：

$$
\Delta t_{jump} = \frac{t_{jump}}{N_{jump}}
$$

---

## 6. 热传导控制方程

## 6.1 强形式

当前模型求解瞬态热传导方程：

$$
\rho(\phi,T) C_p^{eff}(\phi,T)\frac{\partial T}{\partial t}
= \nabla \cdot \left(k(\phi,T)\nabla T\right) + Q_{laser} - Q_{front}
$$

其中 $\phi$ 是积分点材料状态：

$$
\phi \in \{void, powder, solid, mushy, liquid, substrate, support\}
$$

## 6.2 隐式时间离散

代码采用后向 Euler 格式：

$$
\rho C_p^{eff}\frac{T^n - T^{n-1}}{\Delta t}
- \nabla \cdot (k\nabla T^n)
- Q_{laser}^n
+ Q_{front}^{n-1}
= 0
$$

其中对流/辐射边界中的温度在表面项里直接使用当前未知温度 $T$，而移动前沿体积散热项使用 $T_{old}$ 近似。

## 6.3 弱形式

令试函数为 $\delta T$，热问题弱式为：

$$
\int_{\Omega}
\rho C_p^{eff}\frac{T^n-T^{n-1}}{\Delta t}\delta T\,d\Omega
+
\int_{\Omega}
k\nabla T^n\cdot \nabla \delta T\,d\Omega
-
\int_{\Omega}
Q_{laser}\delta T\,d\Omega
+
\int_{\Omega}
Q_{front}\delta T\,d\Omega
-
\int_{\Gamma_N}
q_{surf}\delta T\,d\Gamma
=0
$$

代码中：

```python
get_tensor_map() -> conductivity * T_grad
get_mass_map()   -> rho * cp_eff * (T - T_old) / dt - q_vol + q_front_loss
get_surface_maps() -> convection + radiation boundary flux
```

---

## 7. 激光体热源模型

## 7.1 吸收功率

有效吸收功率为：

$$
P_{eff} = \eta P
$$

其中 $\eta$ 为 absorptivity。

## 7.2 横向高斯分布

在打印平面内，定义激光中心坐标为 $\boldsymbol{x}_l$，积分点坐标为 $\boldsymbol{x}$。若打印平面轴为 $a_0,a_1$，则：

$$
r^2 = (x_{a_0}-x_{l,a_0})^2 + (x_{a_1}-x_{l,a_1})^2
$$

横向高斯项：

$$
G(r) = \exp\left(-\frac{2r^2}{r_b^2}\right)
$$

## 7.3 构建方向深度衰减

构建方向深度定义为：

$$
d = s_b(x_{l,b} - x_b)
$$

只有当积分点位于当前激光作用面以下时才吸收能量：

$$
D(d) =
\begin{cases}
\exp\left(-\frac{d}{d_s}\right), & d\ge 0 \\
0, & d<0
\end{cases}
$$

## 7.4 体热源表达式

当前代码采用：

$$
Q_{laser}(\boldsymbol{x},t) =
\frac{2P_{eff}}{\pi r_b^2 d_s}
\exp\left(-\frac{2r^2}{r_b^2}\right)
\exp\left(-\frac{d}{d_s}\right)
S_{laser}a_q
$$

其中：

| 项 | 含义 |
|---|---|
| $S_{laser}$ | 激光开关，scan 时为 1，jump/dwell/recoat/cooling 时为 0 |
| $a_q$ | 积分点 active 标志 |
| $d_s$ | source depth |

归一化系数：

$$
\frac{2}{\pi r_b^2 d_s}
$$

对应横向高斯积分：

$$
\int_{0}^{\infty}2\pi r\exp\left(-\frac{2r^2}{r_b^2}\right)dr
= \frac{\pi r_b^2}{2}
$$

以及一侧指数深度积分：

$$
\int_0^\infty \exp\left(-\frac{d}{d_s}\right)dd = d_s
$$

因此体积分近似为吸收功率 $P_{eff}$。

---

## 8. 热边界条件与移动前沿散热

## 8.1 初始条件

初始温度：

$$
T(\boldsymbol{x},0) = T_{init}
$$

其中：

$$
T_{init} =
\begin{cases}
T_{preheat}, & \text{if preheat-temperature is provided} \\
T_{ambient}, & \text{otherwise}
\end{cases}
$$

## 8.2 底面边界

若：

```bash
--bottom-thermal-bc fixed
```

则底面为 Dirichlet 边界：

$$
T = T_{bottom}
$$

若：

```bash
--bottom-thermal-bc convection
```

则底面也被加入 surface maps，采用对流/辐射边界。

## 8.3 对流与辐射边界

表面热通量：

$$
q_{conv}=h(T_{amb}-T)
$$

$$
q_{rad}=\epsilon\sigma_{SB}(T_{amb}^4-T^4)
$$

合并为：

$$
q_{surf}=q_{conv}+q_{rad}
$$

代码中使用负号返回给 JAX-FEM 的 surface map：

```python
return -np.array([q_conv + q_rad])
```

该符号约定对应弱式右端/残差方向，具体正负号以 JAX-FEM 的 Neumann 实现为准。

## 8.4 移动前沿体积等效散热

真实逐层打印中，当前构建前沿是内部移动界面，静态边界选择函数无法直接捕捉。代码使用一个体积带状散热近似：

$$
B_f(d) =
\begin{cases}
\exp\left[-\left(\frac{d}{l_f}\right)^2\right]a_q, & d\ge 0 \\
0, & d<0
\end{cases}
$$

对流部分：

$$
Q_{front,conv} = \frac{h_f}{l_f}(T_{old}-T_{amb})B_f(d)
$$

若开启前沿辐射：

$$
Q_{front,rad} = \frac{\epsilon\sigma_{SB}}{l_f}(T_{old}^4-T_{amb}^4)B_f(d)
$$

总前沿散热：

$$
Q_{front}=Q_{front,conv}+Q_{front,rad}
$$

---

## 9. 相变与潜热模型

## 9.1 材料状态编码

当前代码使用如下状态码：

| 代码 | 状态 | 含义 |
|---|---|---|
| 0 | void | 未激活弱材料或空域 |
| 1 | powder | 已铺粉但未熔化粉末 |
| 2 | solid | 已凝固实体 |
| 3 | mushy | 固液混合区 |
| 4 | liquid | 液态熔池 |
| 5 | substrate | 基板 |
| 6 | support | 支撑 |

## 9.2 激活状态转换

当单元第一次被层前沿激活时：

$$
void \rightarrow powder
$$

该转换只表示材料铺粉或参与热计算，不代表已经形成承载残余应力的固体。

## 9.3 温度驱动相态转换

当提供 $T_s,T_l$ 且 $T_l>T_s$ 时，非基板/非支撑的积分点满足：

$$
T \ge T_l \Rightarrow \phi = liquid
$$

$$
T_s \le T < T_l \Rightarrow \phi = mushy
$$

$$
T < T_s \text{ 且之前为 liquid/mushy } \Rightarrow \phi = solid
$$

重熔时：

$$
solid \rightarrow mushy/liquid
$$

重新凝固时：

$$
mushy/liquid \rightarrow solid
$$

并写入新的应力自由参考温度 $T_{ref}$。

## 9.4 表观比热潜热法

在糊状区内加入潜热项：

$$
C_p^{eff}=C_p + C_p^{latent}
$$

其中：

$$
C_p^{latent}=
\begin{cases}
\frac{L}{T_l-T_s}, & T_s \le T \le T_l \\
0, & \text{otherwise}
\end{cases}
$$

当前代码只在 active 积分点上施加潜热修正：

$$
C_p^{latent}=0 \quad \text{if } a_q=0
$$

---

## 10. 粉末/固体/液体热物性传递

## 10.1 温度表格插值

材料表格统一使用：

```csv
T,value
300,15.0
500,18.0
800,22.0
```

对于表格数据，当前使用一维线性插值：

$$
property(T)=\operatorname{interp}(T;T_i,p_i)
$$

支持的热物性表包括：

```text
k_table_solid
cp_table_solid
k_table_powder
cp_table_powder
k_table_liquid
cp_table_liquid
```

若没有提供表格，则使用命令行给出的常数。

## 10.2 固体物性

$$
\rho_s = \rho_{solid}
$$

$$
C_{p,s}=C_{p,solid}(T)
$$

$$
k_s=k_{solid}(T)
$$

## 10.3 粉末物性

$$
\rho_p = \rho_{powder}
$$

$$
C_{p,p}=C_{p,powder}(T)
$$

$$
k_p=k_{powder}(T)
$$

## 10.4 液体物性

$$
\rho_l = \rho_{liquid}
$$

$$
C_{p,l}=C_{p,liquid}(T)
$$

$$
k_l=k_{liquid}(T)
$$

若液体参数缺失，则默认退化为固体参数。

## 10.5 糊状区混合规则

定义液相分数近似：

$$
f_l = \operatorname{clip}\left(\frac{T-T_s}{T_l-T_s},0,1\right)
$$

糊状区密度：

$$
\rho_m = (1-f_l)\rho_s + f_l\rho_l
$$

糊状区比热：

$$
C_{p,m} = (1-f_l)C_{p,s}+f_lC_{p,l}
$$

糊状区导热系数：

$$
k_m = (1-f_l)k_s+f_lk_l
$$

## 10.6 Void 与 inactive 区域

对于 `powder-mode = void`，未激活区域采用弱材料：

$$
\rho_{void}=\rho_s f_{thermal}^{inactive}
$$

$$
k_{void}=k_s f_{thermal}^{inactive}
$$

对于 `powder-mode = powder`，未激活区域采用粉末物性。

最终积分点物性选择可写为：

$$
(\rho,C_p,k)_q =
\begin{cases}
(\rho_{void},C_{p,void},k_{void}), & \phi=void \\
(\rho_p,C_{p,p},k_p), & \phi=powder \\
(\rho_s,C_{p,s},k_s), & \phi=solid/substrate/support \\
(\rho_m,C_{p,m},k_m), & \phi=mushy \\
(\rho_l,C_{p,l},k_l), & \phi=liquid
\end{cases}
$$

---

## 11. 力学控制方程

## 11.1 准静态平衡方程

当前力学问题为：

$$
-\nabla \cdot \boldsymbol{\sigma} = \boldsymbol{0}
\quad \text{in } \Omega
$$

边界条件为：

$$
\boldsymbol{u}=\boldsymbol{u}_D \quad \text{on } \Gamma_D
$$

$$
\boldsymbol{\sigma}\boldsymbol{n}=\boldsymbol{0} \quad \text{on } \Gamma_N
$$

## 11.2 弱形式

令虚位移为 $\delta\boldsymbol{u}$，弱形式为：

$$
\int_{\Omega}\boldsymbol{\sigma}:\nabla\delta\boldsymbol{u}\,d\Omega = 0
$$

代码中 `ThermoMechanical.get_tensor_map()` 返回应力张量映射，由 JAX-FEM 组装刚度残差。

## 11.3 小应变张量

位移梯度为：

$$
\nabla\boldsymbol{u}
$$

小应变：

$$
\boldsymbol{\varepsilon}=\frac{1}{2}\left(\nabla\boldsymbol{u}+\nabla\boldsymbol{u}^T\right)
$$

## 11.4 热应变

当前代码的关键升级是使用凝固参考温度 $T_{ref}$，而不是铺粉激活温度作为应力自由温度。

$$
\Delta T_q = T_q - T_{ref,q}
$$

热应变为：

$$
\boldsymbol{\varepsilon}_{th}=\alpha(T)\Delta T_q\boldsymbol{I}
$$

只有 solid/substrate/support 状态具有热膨胀系数：

$$
\alpha_q =
\begin{cases}
\alpha(T), & \phi \in \{solid,substrate,support\} \\
0, & \phi \in \{void,powder,mushy,liquid\}
\end{cases}
$$

## 11.5 线弹性应力

弹性应变：

$$
\boldsymbol{\varepsilon}_e = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{th}
$$

Lamé 参数：

$$
\mu = \frac{E}{2(1+\nu)}
$$

$$
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

线弹性应力：

$$
\boldsymbol{\sigma}_{trial}
= \lambda\operatorname{tr}(\boldsymbol{\varepsilon}_e)\boldsymbol{I}
+2\mu\boldsymbol{\varepsilon}_e
$$

## 11.6 相态力学缩放

为避免粉末、液态和未激活区域产生不合理刚度，代码引入力学 active factor：

$$
f_m(\phi)=
\begin{cases}
1, & \phi\in\{solid,substrate,support\} \\
f_{mushy}, & \phi=mushy \\
f_{liquid}, & \phi=liquid \\
f_{inactive}, & \phi\in\{void,powder\}
\end{cases}
$$

最终应力：

$$
\boldsymbol{\sigma}=f_m(\phi)\boldsymbol{\sigma}_{model}
$$

其中 $f_{mushy}$、$f_{liquid}$、$f_{inactive}$ 分别对应：

```bash
--mushy-mechanics-factor
--liquid-mechanics-factor
--inactive-mechanics-factor
```

---

## 12. 简化 J2 塑性模型

当：

```bash
--mechanics-model j2_plastic
```

代码使用简化的 J2 偏应力缩放近似。

## 12.1 偏应力

静水应力部分：

$$
\boldsymbol{\sigma}_{hydro}
=\frac{1}{3}\operatorname{tr}(\boldsymbol{\sigma}_{trial})\boldsymbol{I}
$$

偏应力：

$$
\boldsymbol{s}=\boldsymbol{\sigma}_{trial}-\boldsymbol{\sigma}_{hydro}
$$

等效应力：

$$
\sigma_{eq}=\sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}
$$

## 12.2 当前屈服应力

$$
\sigma_Y = \sigma_y(T) + H(T)\bar{\varepsilon}_p^{old}
$$

## 12.3 应力缩放

代码采用：

$$
scale = \min\left(1,\frac{\sigma_Y}{\sigma_{eq}}\right)
$$

并得到：

$$
\boldsymbol{\sigma}=\boldsymbol{\sigma}_{hydro}+scale\cdot\boldsymbol{s}
$$

该方法可防止应力超过屈服面，但不是严格完整的径向返回算法。若后续做定量残余应力预测，建议升级为完整 return mapping，并维护塑性应变张量历史。

## 12.4 等效塑性应变更新

代码中近似更新：

$$
\Delta\bar{\varepsilon}_p=
\frac{\max(\sigma_{eq}-\sigma_Y,0)}{3\mu+H}
$$

$$
\bar{\varepsilon}_p^{new}=\bar{\varepsilon}_p^{old}+a_m\Delta\bar{\varepsilon}_p
$$

其中 $a_m$ 表示该积分点是否足够 active。

## 12.5 重熔塑性历史重置

若：

```bash
--reset-plastic-on-melt
```

当固体进入 mushy/liquid 状态时：

$$
\bar{\varepsilon}_p \leftarrow 0
$$

该设置表示重熔区域发生应力和塑性历史释放。

---

## 13. 凝固参考温度与残余应力数据传递

## 13.1 为什么不用 activation temperature

铺粉激活只代表材料加入计算域，不代表该材料已经熔化—凝固并形成承载残余应力的实体。因此，若使用：

$$
\Delta T = T - T_{activation}
$$

会导致刚铺上的粉末在尚未熔化前就参与固态热应变计算，残余应力可能偏高。

## 13.2 当前方法：solidification-based reference temperature

当积分点从 mushy/liquid 冷却到 $T<T_s$ 时，判定为新凝固：

$$
\phi^{old}\in\{mushy,liquid\},\quad T<T_s
\Rightarrow \phi^{new}=solid
$$

此时写入：

$$
T_{ref}^{new}=T^n
$$

后续热应变使用：

$$
\Delta T = T^n - T_{ref}
$$

当该点再次重熔并重新凝固时，$T_{ref}$ 会被覆盖，实现重熔后的应力自由温度更新。

---

## 14. 热-力数据传递方法

当前时间步 $n$ 的数据传递顺序如下。

### 14.1 StepState 到热源参数

每个时间步从 `StepState` 读取：

```text
laser_center
laser_power
laser_switch
dt
front_coord
mode
```

有效激光功率：

$$
P_{eff}=\eta P_{state}
$$

### 14.2 cell active 到 quadrature active

cell 级激活：

$$
a_e^n = compute\_active\_cell(...)
$$

历史不可逆激活：

$$
a_e^n = a_e^{n-1} \lor a_{raw,e}^n
$$

积分点激活：

$$
a_q^n = make\_quad\_scalar(a_e^n)
$$

### 14.3 温度 DOF 到积分点

节点温度 $T_{node}$ 通过单元形函数插值到积分点：

$$
T_q = \sum_a N_a(\xi_q)T_a
$$

代码中通过：

```python
convert_from_dof_to_quad(T_old)
convert_from_dof_to_quad(T_new)
```

完成。

### 14.4 热物性映射

输入：

```text
T_old_quad
active_quad
phase_quad
material tables
```

输出：

```text
rho_quad
cp_quad
conductivity_quad
latent_cp_quad
```

即：

$$
(T_q^{old}, a_q, \phi_q) \rightarrow \rho_q,C_{p,q},k_q,C_{p,q}^{latent}
$$

### 14.5 热场求解

热问题输入：

```text
T_old
dt
laser_center
P_eff
beam_radius
source_depth
laser_switch
active_quad
rho_quad
cp_quad
conductivity_quad
latent_cp_quad
```

求解得到：

$$
T^n = solver(thermal)
$$

### 14.6 相态和参考温度更新

用 $T_q^n$ 更新：

```text
phase_quad
T_ref_quad
eqp_quad
newly_solidified_quad
entered_melted_quad
```

数学上：

$$
(T_q^n,\phi_q^{n-1},T_{ref,q}^{n-1},\bar{\varepsilon}_{p,q}^{n-1})
\rightarrow
(\phi_q^n,T_{ref,q}^n,\bar{\varepsilon}_{p,q}^{*})
$$

### 14.7 温度到力学增量

当前力学使用：

$$
\Delta T_q^n = (T_q^n-T_{ref,q}^n)a_q^n
$$

该量作为 `dT_quad` 进入力学问题。

### 14.8 力学物性映射

输入：

```text
T_quad
active_quad
phase_quad
E table
alpha table
poisson table
yield table
hardening table
```

输出：

```text
active_factor_quad
E_quad
alpha_quad
poisson_quad
yield_quad
hardening_quad
```

即：

$$
(T_q^n,a_q^n,\phi_q^n)\rightarrow f_{m,q},E_q,\alpha_q,\nu_q,\sigma_{y,q},H_q
$$

### 14.9 力学求解与后处理

若满足：

```python
state.global_step % mechanics_every == 0
```

则求解位移：

$$
\boldsymbol{u}^n = solver(mechanics)
$$

然后在积分点计算：

```text
stress_quad
vm_quad
eqp_quad
```

---

## 15. 位移梯度、应力和 von Mises 后处理

## 15.1 位移梯度

对每个单元和积分点：

$$
\nabla \boldsymbol{u}_q = \sum_a \boldsymbol{u}_a \otimes \nabla N_a(\xi_q)
$$

代码通过：

```python
u_grads = take(sol, cells) * shape_grads
```

计算。

## 15.2 应力分量输出

代码保存以下积分点应力分量：

```text
stress_quad_xx
stress_quad_yy
stress_quad_zz
stress_quad_xy
stress_quad_yz
stress_quad_xz
```

若一个单元有多个积分点，会加上 quad 编号。

## 15.3 von Mises 应力

对于三维应力张量：

$$
\sigma_{vm}=\sqrt{
\frac{1}{2}\left[(\sigma_{xx}-\sigma_{yy})^2+(\sigma_{yy}-\sigma_{zz})^2+(\sigma_{zz}-\sigma_{xx})^2\right]
+3(\sigma_{xy}^2+\sigma_{yz}^2+\sigma_{xz}^2)
}
$$

该值保存为：

```text
vm_quad
```

---

## 16. 输出数据字段

每个 VTK 文件保存以下数据。

## 16.1 点数据 point data

| 字段 | 含义 |
|---|---|
| `T` | 节点温度 |
| `u` | 节点位移 |

## 16.2 单元数据 cell data

| 字段 | 含义 |
|---|---|
| `active` | 单元是否已激活 |
| `layer_id` | 单元所属层编号，基板/支撑为 0 |
| `activation_step` | 单元首次激活步 |
| `activation_temperature` | 单元首次激活时温度，仅用于记录 |
| `solidification_temperature` | 单元首次/最近凝固温度，来自积分点 $T_{ref}$ 平均值 |
| `solidification_step` | 单元凝固步 |
| `material_state` | 材料状态编码 |
| `dT` | 用于力学热应变的 $T-T_{ref}$ 单元平均值 |
| `eq_plastic_strain` | 等效塑性应变 |
| `max_temperature_history` | 历史最高温度 |
| `mechanics_valid` | 当前 VTK 是否包含当前步力学结果 |
| `mechanics_source_step` | 力学结果来源步 |
| `mode_id` | 当前工艺模式编码 |
| `stress_quad_*` | 积分点应力分量 |
| `vm_quad` | 积分点 von Mises 应力 |

## 16.3 路径输出

`path_used.csv` 保存：

```text
step,mode,layer,hatch,scan,x,y,z,front_coord,power,laser_on,dt,scan_frac,hatch_frac
```

该文件用于复查：

1. 构建方向是否正确；
2. 扫描是否位于指定平面；
3. 层厚和 hatch spacing 是否生效；
4. jump/recoat/cooling 是否进入热积分；
5. 外部路径和自动 raster 结果是否可复现。

---

## 17. 主循环算法流程

```text
1. 读取命令行参数和配置文件
2. 读取 inp 四面体网格
3. 缩放坐标，确定 pmin/pmax
4. 根据 build-axis 和 base-side 定义底面、外露面、侧壁
5. 扣除 substrate/support 后得到 part build box
6. 若提供 layer-thickness，自动推导 layers
7. 生成 StepState 序列：scan/jump/dwell/recoat/cooling
8. 写出 path_used.csv
9. 初始化：T_old, u_guess, eqp_quad, phase_quad, T_ref_quad
10. 对每个 StepState:
    10.1 根据 front_coord 更新 active_cell
    10.2 active_cell -> active_quad
    10.3 VOID + active -> POWDER
    10.4 T_old -> T_old_quad
    10.5 根据 T_old_quad + phase_quad 计算 rho/cp/k/latent_cp
    10.6 组装热问题参数并求解 T_new
    10.7 T_new -> T_quad
    10.8 更新 phase_quad, T_ref_quad, eqp_quad
    10.9 dT_quad = (T_quad - T_ref_quad) * active_quad
    10.10 根据 T_quad + phase_quad 计算力学材料参数
    10.11 若达到 mechanics_every，求解 u
    10.12 计算 stress_quad, vm_quad, eqp_quad
    10.13 按输出频率保存 VTK
    10.14 更新 T_old = T_new
11. 若 release-after-cooling，改用 anchor 约束求解释放变形
12. 输出 release.vtu
```

---

## 18. 特定材料铺粉模拟需要提供的数据

若要针对 Ti-6Al-4V、IN625、不锈钢、铝合金或金属玻璃合金等特定材料进行铺粉模拟，建议提供以下数据。

## 18.1 热学参数

| 状态 | 必需参数 | 推荐形式 |
|---|---|---|
| 粉末 | $\rho_p$, $C_{p,p}$, $k_p$ | 常数或 T-value 表 |
| 固体 | $\rho_s$, $C_{p,s}$, $k_s$ | T-value 表优先 |
| 液体 | $\rho_l$, $C_{p,l}$, $k_l$ | 常数或 T-value 表 |
| 相变 | $T_s$, $T_l$, $L$ | 常数 |
| 表面 | emissivity | 常数/校准参数 |
| 热源 | absorptivity | 常数/校准参数 |

## 18.2 力学参数

| 参数 | 推荐形式 |
|---|---|
| $E(T)$ | T-value 表 |
| $\nu(T)$ | 常数或 T-value 表 |
| $\alpha(T)$ | T-value 表 |
| $\sigma_y(T)$ | T-value 表，J2 塑性必须提供 |
| $H(T)$ | 常数或 T-value 表 |

## 18.3 粉末等效参数说明

粉末态导热系数不能直接使用实体导热系数。实际 LPBF 粉末床导热通常显著低于实体材料，受以下因素影响：

1. 粉末粒径分布；
2. 球形度；
3. 堆积密度；
4. 气氛导热；
5. 氧化膜；
6. 温度；
7. 粉末重复使用次数。

因此，推荐将 `conductivity_powder` 作为实验校准参数或基于公开数据给定范围。

## 18.4 推荐材料配置文件结构

```yaml
rho_solid: 4430
rho_powder: 2500
rho_liquid: 4100
cp_solid: 560
cp_powder: 560
cp_liquid: 750
conductivity_solid: 7.0
conductivity_powder: 0.5
conductivity_liquid: 25.0
solidus_temperature: 1878
liquidus_temperature: 1928
latent_heat: 2.86e5
emissivity: 0.35
absorptivity: 0.4
young: 1.1e11
poisson: 0.34
alpha: 9.0e-6
```

若有温度表，则用：

```yaml
k_table_solid: material/ti64_k_solid.csv
cp_table_solid: material/ti64_cp_solid.csv
k_table_powder: material/ti64_k_powder.csv
cp_table_powder: material/ti64_cp_powder.csv
k_table_liquid: material/ti64_k_liquid.csv
cp_table_liquid: material/ti64_cp_liquid.csv
E_table: material/ti64_E.csv
alpha_table: material/ti64_alpha.csv
yield_table: material/ti64_yield.csv
hardening_table: material/ti64_hardening.csv
```

---

## 19. 当前模型的适用范围

当前代码适合用于：

1. 单道/多道扫描热历史预测；
2. 多层打印热积累分析；
3. 不同 layer thickness、hatch spacing、scan rotation 对温度场影响的比较；
4. 基于等效粉末床的熔池区域近似识别；
5. 热应变驱动的应力/变形趋势分析；
6. 神经算子或机器学习代理模型的数据生成；
7. 工艺参数敏感性研究。

---

## 20. 当前模型的局限性

当前模型尚未考虑：

1. 熔池流动；
2. Marangoni 对流；
3. 蒸发反冲压力；
4. keyhole 模式；
5. 飞溅；
6. 粉末颗粒级接触和辐射；
7. 动态自由表面；
8. 实际铺粉刮刀/滚轮过程；
9. 各向异性材料组织演化；
10. 完整的有限应变塑性；
11. 严格的 J2 径向返回塑性积分；
12. 热-力双向耦合；
13. 残余应力实验标定。

因此，当前结果更适合作为 **工程近似模型、趋势分析模型和机器学习数据生成器**，而不宜直接作为工业级残余应力定量预测模型。

---

## 21. 建议的验证指标

建议每次仿真输出或后处理以下指标：

| 类别 | 指标 |
|---|---|
| 热场 | $T_{max}$、$T_{min}$、平均温度、冷却速率 |
| 熔池 | 熔池宽度、深度、长度、mushy/liquid 体积分数 |
| 相变 | 首次熔化步、凝固步、重熔次数 |
| 应力 | 最大 von Mises、$\sigma_{xx}$、$\sigma_{yy}$、$\sigma_{zz}$ |
| 变形 | 最大位移、顶面翘曲、释放后变形 |
| 工艺 | 路径长度、实际扫描速度、jump 时间、recoat 时间 |
| 数值 | 牛顿迭代次数、异常温度、非物理解 |

---

## 22. 可用于论文方法章节的简述

本研究建立了一种面向粉末床增材制造的三维瞬态热-力耦合有限元模型。热分析采用隐式时间离散的瞬态热传导方程，并引入基于扫描路径的移动体积高斯热源。材料域根据构建方向和层厚进行逐层激活，扫描路径在打印平面内由 hatch spacing、扫描速度、蛇形策略及层间旋转角共同确定。为近似粉末、熔池和凝固实体的差异，模型在积分点层面引入 void、powder、solid、mushy、liquid、substrate 和 support 等材料状态，并根据状态选择粉末态、固态和液态热物性。相变潜热通过表观比热法加入热方程。热场求解完成后，积分点温度驱动相态更新；当材料由熔融/糊状区冷却为固态时，写入凝固参考温度，作为后续热应变计算的应力自由温度。力学分析采用准静态平衡方程，热应变由当前温度与凝固参考温度之差确定，并根据材料状态缩放粉末、糊状区和液态材料的承载能力。模型输出温度场、位移场、材料状态、凝固历史、等效塑性应变、应力分量及 von Mises 应力，可用于研究扫描策略、层厚、hatch spacing 和材料热物性对温度演化与残余应力趋势的影响。

---

## 23. 下一步建议

1. **增加 melt pool metrics 自动后处理**
   从 `material_state == liquid/mushy` 的积分点自动估算熔池宽度、深度和长度。

2. **增加 cooling rate 输出**
   保存：

   $$
   \dot{T} = \frac{T^n-T^{n-1}}{\Delta t}
   $$

3. **增加 remelt count**
   记录每个积分点进入 liquid/mushy 的次数，用于分析层间重熔。

4. **升级 J2 塑性**
   引入塑性应变张量：

   $$
   \boldsymbol{\varepsilon}_p
   $$

   并实现完整 radial return mapping。

5. **增加材料配置模板库**
   为 Ti64、IN625、316L、不锈钢、铝合金和金属玻璃合金建立标准 YAML/CSV 参数集。

6. **接入神经算子数据生成流程**
   将输入参数：

   ```text
   P, v, h_L, h_s, rotation, k_powder, absorptivity, emissivity
   ```

   映射到输出场：

   ```text
   T(x,t), material_state(x,t), sigma_vm(x,t), u(x,t)
   ```

---

## 24. 结论

当前代码已经形成了一个可扩展的 3D 粉末打印热-力耦合模拟框架。其核心数据链路为：

```text
扫描路径 StepState
    -> 层激活 active_cell / active_quad
    -> 热物性 rho/cp/k/latent_cp
    -> 瞬态热传导 T_new
    -> 相态更新 phase_quad
    -> 凝固参考温度 T_ref_quad
    -> 热应变 dT_quad
    -> 力学材料参数 E/alpha/nu/yield/hardening
    -> 准静态位移 u
    -> 应力 stress_quad / von Mises
    -> VTK + path_used.csv 输出
```

该框架已经具备支撑论文实验的基础，特别适合作为后续神经算子加速、多材料参数校准、扫描策略优化和金属玻璃成型能力数据驱动预测模块的前端物理仿真器。
