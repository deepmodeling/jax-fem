#!/usr/bin/env python3
"""V1 三角对照图 (Fig A: 绝对尺寸;Fig B: 相对误差)。

数据源(禁止在此处修改数值,只做绘图):
  我方   -> 各运行 v1_meltpool_metrics.json,汇总于 ../RESULTS.md
  NIST   -> ../inputs/nist-meltpool.json (Lane2020 Table 4 参考值 + Table 5 扩展不确定度)
  Balbaa -> ../inputs/balbaa-model.json reported_validation_results (Fig 8-10 数字化)
不确定度约定:宽/深用 u_mean(均值标准不确定度);长用 U(k=2) 扩展不确定度
(Lane2020 Table 5 只对长度给出完整不确定度预算)。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ---- 中文字体:借用 Windows 侧字体(WSL 无 CJK 字体) ----
for cand in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf",
             "/mnt/c/Windows/Fonts/msyhl.ttc"):
    if Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- 校验过的分类调色板前三槽(light mode) ----
C_OURS, C_NIST, C_BALBAA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

CASES = ["A\n150 W\n400 mm/s", "B\n195 W\n800 mm/s", "C\n195 W\n1200 mm/s"]
OURS = {"width": [164.4, 141.0, 106.2], "depth": [114.9, 90.6, 60.0],
        "length": [525.0, 575.0, 570.0]}
NIST = {"width": [171.0, 133.0, 100.0], "depth": [151.0, 91.0, 60.0],
        "length": [659.0, 780.0, 754.0]}
NIST_U = {"width": [0.82, 0.50, 0.48], "depth": [5.75, 0.52, 0.16],
          "length": [79.84, 87.39, 95.85]}
BALBAA = {"width": [None, 131.0, None], "depth": [None, 44.0, None],
          "length": [None, 378.0, None]}
TITLES = {"width": "熔池宽度", "depth": "熔池深度(基板面以下)", "length": "熔池长度"}
UNOTE = {"width": "误差棒:u_mean", "depth": "误差棒:u_mean", "length": "误差棒:U(k=2)"}

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)


def style(ax):
    ax.set_facecolor(SURF)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


# ================= Fig A: 绝对尺寸 =================
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), facecolor=SURF)
x = np.arange(3)
w = 0.26
handles = None
for ax, key in zip(axes, ("width", "depth", "length")):
    style(ax)
    b1 = ax.bar(x - w, OURS[key], w * 0.92, color=C_OURS, label="我方求解器", zorder=3)
    b2 = ax.bar(x, NIST[key], w * 0.92, color=C_NIST, label="NIST 实测", zorder=3,
                yerr=NIST_U[key],
                error_kw=dict(ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4))
    bx = [i + w for i, v in zip(x, BALBAA[key]) if v is not None]
    bv = [v for v in BALBAA[key] if v is not None]
    b3 = ax.bar(bx, bv, w * 0.92, color=C_BALBAA, label="Balbaa 2022 (ABAQUS)", zorder=3)
    handles = [b1, b2, b3]
    ax.set_ylim(0, max(NIST[key]) * 1.30)
    # 选择性直接标注:仅工况 B(三方齐备的对照点)
    # NIST 标注抬到误差棒之上,避免压线
    for xi, v, pad in ((1 - w, OURS[key][1], 0.0), (1, NIST[key][1], NIST_U[key][1]),
                       (1 + w, BALBAA[key][1], 0.0)):
        ax.text(xi, v + pad + ax.get_ylim()[1] * 0.02, f"{v:g}", ha="center",
                va="bottom", fontsize=9, color=INK2, zorder=5)
    for xi in (0, 2):
        ax.text(xi + w, ax.get_ylim()[1] * 0.02, "未发表", ha="center", va="bottom",
                fontsize=7.5, color=MUTED, rotation=90, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(CASES, fontsize=8.5, color=INK2)
    ax.set_title(TITLES[key], fontsize=11, color=INK, pad=8)
    ax.set_ylabel("µm", fontsize=9, color=MUTED)
    ax.text(0.02, 0.97, UNOTE[key], transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=MUTED)

fig.suptitle("V1 单道熔池三角对照:我方 vs NIST AMB2018-02 实测 vs Balbaa 2022 已发表预测",
             fontsize=12.5, color=INK, y=0.985)
fig.legend(handles=handles, labels=["我方求解器", "NIST 实测", "Balbaa 2022 (ABAQUS)"],
           frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.2,
           ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.945))
fig.text(0.5, 0.008,
         "IN625 裸板单道,熔池边界=固相线 1290 °C(两参考文献共同约定);零标定——"
         "无任何参数向实测或已发表值回调。Balbaa 仅在工况 B 发表了预测值;标注数值仅示工况 B。",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=[0, 0.035, 1, 0.895], w_pad=2.4)
fa = OUT / "v1_triangle_absolute.png"
fig.savefig(fa, dpi=200, facecolor=SURF)
print(f"wrote {fa}")

# ================= Fig B: 工况 B 相对误差 =================
fig2, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURF)
style(ax)
q = ["熔池宽度", "熔池深度", "熔池长度"]
ours_err = [6.0, -0.5, -26.3]
balb_err = [-1.5, -51.6, -51.5]
y = np.arange(3)
h = 0.32
ax.barh(y + h / 2 + 0.02, ours_err, h, color=C_OURS, label="我方求解器", zorder=3)
ax.barh(y - h / 2 - 0.02, balb_err, h, color=C_BALBAA,
        label="Balbaa 2022 (ABAQUS)", zorder=3)
for yi, v in zip(y + h / 2 + 0.02, ours_err):
    ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"{v:+.1f}%", va="center",
            ha="left" if v >= 0 else "right", fontsize=9, color=INK2, zorder=5)
for yi, v in zip(y - h / 2 - 0.02, balb_err):
    ax.text(v - 1.2, yi, f"{v:+.1f}%", va="center", ha="right",
            fontsize=9, color=INK2, zorder=5)
ax.axvline(0, color=BASE, lw=1.4, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(q, fontsize=10, color=INK)
ax.set_xlabel("相对 NIST 实测的偏差 (%)  ← 低估    高估 →", fontsize=9, color=MUTED)
ax.set_xlim(-62, 20)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.yaxis.grid(False)
ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2,
          handlelength=1.2)
ax.set_title("工况 B(195 W / 800 mm/s)相对实测偏差:我方 vs 已发表 ABAQUS 结果",
             fontsize=11.5, color=INK, pad=10)
fig2.text(0.5, 0.01,
          "深度:我方 −0.5%(达测量精度)对 已发表 −51.6%;长度两者均低估,"
          "原因为熔池内流动(Marangoni)未建模,已登记。",
          ha="center", fontsize=8, color=MUTED)
fig2.tight_layout(rect=[0, 0.05, 1, 1])
fb = OUT / "v1_triangle_relative_caseB.png"
fig2.savefig(fb, dpi=200, facecolor=SURF)
print(f"wrote {fb}")
