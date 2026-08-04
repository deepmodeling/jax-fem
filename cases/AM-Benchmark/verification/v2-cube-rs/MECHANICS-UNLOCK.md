# V2 力学解锁 — D-V2-22 因果证伪与 D-V2-19-R1 采纳

2026-08-04。本文件记录 V2 多轨力学从"四臂全灭"到"两臂干净收敛"的完整证据链。
配套登记见 `inputs/deviations.yaml` 的 D-V2-19(R1)与 D-V2-22(resolved)。

> **纪律**:全程零标定,没有任何参数向 XRD / ABAQUS / 高温计数值回调;
> 未修改任何共享求解器代码;所有改动都在 `v2-cube-rs/` 自有输入内。

## 1. 结论摘要

| 项 | 结论 |
|---|---|
| 不收敛的病因 | **不是** J-C 近零切线奇异(D-V2-22),而是**高温端切线比 H/E**(D-V2-19-R1) |
| 采纳的修复 | 正则化硬化由"固定加法 H_reg = 1e7 Pa"改为"逐段最小塑性切线 H ≥ 0.01·E(T)" |
| 主臂 / 括号臂 | offset(0.2 % 偏移约定)为主臂,cap 为 RS 敏感度括号(yuyao 决策,IET-5) |
| 共享代码 | **未改动**;`flat` 臂已证明 `flow_curve_table` 消费路径无罪 |
| 对产生残余应力那一段材料本构的扰动 | 在实测工作应变区(eqp ≤ 0.02)**恰好为零**,仅固相线行改变 |

## 2. 为什么 D-V2-22 的因果被推翻

A 相四臂(asis / offset / cap / asis + 求解器参数)**全部**失败,且**全部停在同一处**
(ledger = 12, global_step = 10):

| 臂 | 近零切线 H1/E(室温) | 结果 | ledger | Newton 不收敛次数 |
|---|---|---|---|---|
| asis | 2.98 | FAIL | 12 | 5 |
| offset | 0.23 | FAIL | 12 | 5 |
| cap | 0.10 | FAIL | 12 | 5 |
| asis + max-iter 200 / max-cuts 6 | 2.98 | FAIL | 12 | 8 |

近零切线被从"比弹性还硬"一路括到"十分之一弹性"——**跨度 30 倍,收敛行为纹丝不动**。

原因在生成器里一眼可见,并已逐点核验(`scratch/verify_claim.py`):
J-C 的软化因子 `soft(T)` 在 Tm 处恰好归零,而它乘的正是三臂**唯一不同**的硬化项
`B·(…)^n`。于是:

```
每个温度行上三臂之间的最大相对差
  294.15 K … 1423.15 K   29 % – 38 %     三臂互不相同
  1563.15 K(固相线)      0.0000 %        三臂逐点完全相同  <<<
```

**三臂唯一相同的一行,正是失败发生的那一行。** 试验梯从未变动过失败的那一段,
所以它当然测不出差别——这不是"近零切线无害"的证据,而是"这把刀没切在病灶上"。

D-V2-22 的观测本身仍然成立(室温首段 H = 178.7 GPa > E = 171 GPa 不可辩护,
如印刷值在 eps_p = 0.2 处给出 1746 MPa 真应力,高于任何实测 IN625 流动应力),
只是它**不是**这次不收敛的原因。该主张已在登记里显式撤回。

## 3. 第一刀:消费路径无罪

上一轮的 `v2_T1_coarse` 对照看似指向 flow curve,但它一次换掉三样东西
(全温区常数 490 MPa 屈服、常数 1 GPa 硬化、以及 `yield_table + hardening_table`
这条**另一条代码路径**)。两个二分臂把它们分开:

| 臂 | 构造 | ledger | Newton 不收敛 |
|---|---|---|---|
| `flat` | T1 占位对,但经 `flow_curve_table` 交付 | 152 | 0 |
| `hotfloor` | offset 近零端 + T1 的高温地板 | 165 | 0 |

`flat` 通过 ⇒ **`flow_curve_table` 消费路径(共享求解器代码)无罪**,病因在表值。
按既定约束,共享代码一行未改。

## 4. 第二刀:屈服轴 vs 硬化轴(单变量)

`hotfloor` 一次改了两样(高温屈服量级、高温硬化模量),故再切一刀。
所有臂与失败的 offset 臂**逐字相同,只动一个旋钮**(`vtmp/v_hotreg.sh`):

| 臂 | 改动 | 固相线 H1/E | ledger | Newton 不收敛 | 判读 |
|---|---|---|---|---|---|
| offset(基线) | — | 1.6e-4 | 12 | 5 | FAIL |
| `hy10` | 地板 1e6 → 1e7 Pa | 1.6e-4 | 20 | 3 | 靠反复回切爬行,未恢复健康 |
| `hy100` | 地板 1e6 → 1e8 Pa | 0.00 | 12 | 5 | **FAIL,与基线逐位同签名** |
| `htan` | H_reg 1e7 → 1e9 Pa | 1.6e-2 | 72 | **0** | 健康 |

> `hy10` / `htan` 因内存告急被我按精确 PID 于 1200 s 上限前终止(rc = 143),
> 不是自然结束;两者判读在终止时均已确立(见 ledger 与不收敛次数)。
> `hy100` 是自然失败(rc = 1)。

**`hy100` 是最锋利的一条证据**:它在屈服轴上改动最大(1 → 100 MPa),却失败得最彻底。
因为把地板抬到 1e8 Pa 会把 1423 K 与 1563 K **两行**的首段一起压平,
使 H1/E 从 1.6e-4 掉到 0.00,把零切线带从一行扩成两行。
**该臂里屈服量级与切线比朝相反方向移动,而结果跟着切线走,不跟着屈服走。**

十个臂无一反例:凡越过 ledger = 12 的臂(`flat` / `hotfloor` / `htan`)高温切线比都是
1.6e-2;凡停在那里的臂高温切线比都是 1.6e-4 或更低。

## 5. 采纳的修复(D-V2-19-R1)

病态量本来就是**比值** H/E,而不是硬化模量的绝对值:

- 固相线行:H_reg = 1e7 Pa 对 E = 61.6 GPa ⇒ H/E = 1.6e-4,近乎理想塑性,
  且因为 J-C 软化在整个近固相线带上坍塌,这一状态是**成千上万个积分点同时**发生的;
- 室温:同一个 1e7 Pa 完全无关痛痒,因为 J-C 自带 35–40 GPa 切线。

一个用**绝对模量**表达的正则化项不可能同时在温度区间两端都正确;把它**系到 E(T)** 就可以。
故 R1 采纳:

```
sigma 逐段满足   dsigma/deps_p  >=  0.01 * E(T)
```

生成:`model/make_flow_curve_variants.py --arm offset --name offset_mt --min-tangent-frac 0.01`。

### 为什么用"下限"而不是 `htan` 的"加法项"

`htan` 是诊断臂,不可采纳:加法 1e9 Pa 会把室温 `sigma(eps_p = 0.2)` 从 1391 抬到
1589 MPa(+14 %),那是在动**产生残余应力的那一段**材料本构。
逐段**最小切线**只在 J-C 已经软化坍塌的地方起作用。逐点量化(`scratch/perturb.py`):

在**完整运行实测**的塑性应变处逐行核验(`scratch/perturb_at_peak.py`)。
完整运行(240 步 + 冷却)的 `eq_plastic_strain`:**峰值 0.0233,p99 = 6.5e-4**。
(此前记录的 0.004–0.005 只属于 900 s 截断筛选运行,不能外推到完整运行,已更正。)

```
offset -> offset_mt   相对扰动 (%)
     T(K)   @p99=6.5e-4   @峰值=0.0233
   294.15         0.000          0.000
     ...           ...            ...      <- 294 K … 1273 K 全部恰好为 0.000
  1273.15         0.000          0.000
  1423.15         0.000          0.414
  1563.15        39.103       1144.222     <- 固相线行,本来就是正则化,不是材料数据
```

即:**在实测塑性应变范围内,1273 K 及以下所有温度行的流动应力逐字节未变**
(已用 CSV 字面量逐格点核验,`scratch/verify_exact.py`:84 个 `eps_p ≤ 0.02` 的格点中
78 个完全相同,6 个不同全部落在固相线行);1423 K 行最大偏 0.41 %;
其余改动全部落在固相线那一行——而那一行从 D-V2-19 起就已声明是正则化而非材料数据。

`0.01` 是对 H/E 的数值适定性下限:比病态值 1.6e-4 高一个量级,比表中最小的真实
J-C 切线比(1423 K 处 0.081)低一个量级。它不向任何实测量拟合。

## 6. 解锁证据:两臂完整运行 + RS 敏感度括号

`model/runs/v_fcfull.sh offset_mt cap_mt`,coarse 网格(30000 单元),
240 个扫描步 + 40 个冷却步跑完:

| 臂 | 结果 | ledger | Newton 不收敛 | rc | 能量台账最大相对失衡 |
|---|---|---|---|---|---|
| `offset_mt`(主臂) | **COMPLETE** | 240 | 2 | 0 | 4.022e-06 |
| `cap_mt`(括号臂) | **COMPLETE** | 240 | 6 | 0 | 4.022e-06 |

这是 V2 力学的**第一次完整收敛运行**。对比:同一网格、同一载荷、同一求解器参数下,
`asis / offset / cap / asis_iter / hy100` 五个臂全部死在 global_step = 10。

> 诚实注记:两臂**并非全程零失败**。offset_mt 有 2 次、cap_mt 有 6 次 Newton
> 不收敛,都发生在后段(塑性累积与熔化material 最多的阶段),都被回切子步吸收后恢复,
> 运行以 rc = 0 与台账 `complete: true` 正常结束。cap 臂更吃力与它近零端更软
> (H1/E = 0.10 对 offset 的 0.23)一致。**尚未证明**在 Balbaa 平价功率(220 W)
> 下也能这样收敛——本次是 D-V2-20 的 50 W 探针载荷。

### RS 敏感度括号(D-V2-22 要求如实上报的量)

两臂用同一网格、同一路径、同一求解器参数,只有 flow curve 不同,故可**逐单元**相减,
无需插值。件内顶层 10000 个单元,末帧 `step_000239_cooling.vtu`:

| 量 | 主臂 RMS | 括号臂 RMS | 最大差 | 相对 RMS 差 |
|---|---|---|---|---|
| sigma_xx | 41.63 MPa | 40.84 MPa | 12.47 MPa | **1.96 %** |
| sigma_yy | 8.30 MPa | 8.06 MPa | 4.05 MPa | **3.14 %** |
| sigma_zz | 7.13 MPa | 6.95 MPa | 3.97 MPa | **2.64 %** |
| von Mises | 41.73 MPa | 40.96 MPa | 14.35 MPa | **1.95 %** |
| eqp | 2.058e-3 | 2.077e-3 | 3.23e-4 | 1.02 % |

**括号宽度约 2 – 3 %。** 这是个有分量的结论:offset 与 cap 在室温近零切线上相差
30 倍(H1/E = 0.23 对 0.10),而它们对件内残余应力的影响只有 2–3 %。
也就是说,**D-V2-22 的读法选择不是 RS 的主要不确定度来源**——这对后续 RS 闸门
(IET-9)是个好消息,该项不确定度远小于典型的 ABAQUS-vs-XRD 差距。
按已确认决策,此差异如实上报,不回调。

## 7. 附带查清、但**不是**本次病因的两件事

1. **塑性应变端点钳制**:`jax_fem/materials/j2.py::_interpolate_clamped` 对 eps_p 轴做端点钳制,
   越过表末端(eps_p = 0.2)后切线**恰好为 0**(`_plastic_increment_from_curve` 的
   `beyond_root = (target - stresses[-1]) / 3mu` 分支)。作为候选病因被**实测排除**
   而非论证排除:完整运行里 eqp 峰值 0.0233、p99 = 6.5e-4,越过 0.2 的单元数为
   **0 / 30000**,本载荷下根本够不着。但在 Balbaa 平价功率(220 W)下高温带大得多,
   届时会变成实打实的问题——已登记。
2. **`max_temperature_history` 是单元量**:vtu 里该场最大值 1537.7 K,低于固相线;
   而同一帧的节点温度 `T` 最大 2467 K。本构在积分点上取值,不在单元均值上,
   读该场判断"有没有到固相线"会得出错误结论。

## 7. 主线可复用的教训

主线 AMB2018-01 不用 Johnson-Cook,所以 D-V2-22 那个具体陷阱不迁移。但两条一般性教训迁移:

1. 任何表格化流动曲线,若**首段比 E 还硬**,就会产生这次的签名
   (巨大 force_ratio + 位移修正塌缩);
2. 更危险的一条(D-V2-19-R1):任何**温度相关**流动曲线,若随温度软化到地板而 E(T)
   不同步坍塌,就会让 H/E 在**一整带单元上同时**趋零;而用**绝对值**表达的硬化正则化项
   会随着这一带变大而悄悄失效——失效时没有任何报错,只有 Newton 停滞。

建议在消费 `flow_curve_table` 的地方加两条装载期断言:逐行 `H1 < E`,以及
`min(H/E) > 阈值`。两条都能在**装载时**而不是 Newton 停滞时抓住这两类问题。

## 8. 复现

```bash
# 生成三个 A 相臂(不带新参数时与 2026-08-03 已提交表逐字节相同)
python model/make_flow_curve_variants.py --arm all

# 生成 D-V2-19 单变量梯
python model/make_flow_curve_variants.py --arm offset --name hy10  --floor-pa 1e7
python model/make_flow_curve_variants.py --arm offset --name hy100 --floor-pa 1e8
python model/make_flow_curve_variants.py --arm offset --name htan  --h-reg-pa 1e9

# 生成采纳的生产臂
python model/make_flow_curve_variants.py --arm offset --name offset_mt --min-tangent-frac 0.01
python model/make_flow_curve_variants.py --arm cap    --name cap_mt    --min-tangent-frac 0.01

# 运行(WSL conda jax-fem-env,CPU)。协议脚本已原样归档在 model/runs/:
bash model/runs/v_hotreg.sh                    # D-V2-19 单变量梯
bash model/runs/v_fcfull.sh offset_mt cap_mt   # 采纳臂 + 括号臂完整运行
bash model/runs/v_sub6.sh                      # D-V2-07 基板探针(串行)
```

`model/runs/` 里的脚本是**本次实际运行的原件**,未改写路径。归档的动机是:
2026-08-03 的 `v_fcladder.sh` / `v_bisect.sh` / `v_sub5.sh` 只存在于仓库外的
`~/work/159/vtmp`,本次会话不得不从 `output/*/used_config.json` 反推那一长串 runner
参数,才能保证新臂与失败臂**逐字可比**。这一步不必再重复第二次。

各臂之间唯一的差异是 `--config`(即 `flow_curve_table`)与输出目录;
`v_hotreg.sh` / `v_fcfull.sh` 的 `run_case()` 与 `v_bisect.sh` 的逐字相同,
这是"单变量"这个说法能成立的前提。已核验:`v2_material_config_fc_offset_mt.json`
与基准 `v2_material_config.json` 除 `_comment` 与 `flow_curve_table` 外字段完全一致。
