#!/usr/bin/env python3
"""V2 多道扫描路径(Balbaa 多道模型,Sec 2.6.2 / Table 3)。

蛇形往返,hatch 沿 y,扫描沿 x。默认工况 = 他的高温计验证点
220 W / 650 mm/s / hatch 0.12 mm(Sec 3.3)。

--tracks 限制道数:基板离散敏感性研究(D-V2-07)只需要少数几道,
结论不依赖整层保真度,因为三个基板变体共用同一条路径。
整层(33 道)留给热门的高温计对照。

条带宽度未给(D-V2-09):4 mm 域小于典型 EOS 条带(5-10 mm),按单条带
处理,即纯蛇形。CSV 列与 runner --path-file 一致(m/s/W)。
"""
import argparse
from pathlib import Path

AREA_X = 4.0e-3
AREA_Y = 4.0e-3

ap = argparse.ArgumentParser()
ap.add_argument("--power", type=float, default=220.0)
ap.add_argument("--speed", type=float, default=0.650)
ap.add_argument("--hatch", type=float, default=0.12e-3)
ap.add_argument("--tracks", type=int, default=0, help="0 = 整层")
ap.add_argument("--sample-step", type=float, default=50.0e-6)
ap.add_argument("--jump-speed", type=float, default=5.0)
ap.add_argument("--z", type=float, default=440.0e-6, help="粉层顶面")
ap.add_argument("--margin", type=float, default=0.1e-3, help="道端距域边留白")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()

n_full = int(round((AREA_Y - 2 * args.margin) / args.hatch)) + 1
n = n_full if args.tracks <= 0 else min(args.tracks, n_full)
# 少道数时居中放置,避免贴边效应污染基板研究
y0 = (AREA_Y - (n - 1) * args.hatch) / 2.0
x_lo, x_hi = args.margin, AREA_X - args.margin

rows = []
t = 0.0
prev = None
for i in range(n):
    y = y0 + i * args.hatch
    a, b = (x_lo, x_hi) if i % 2 == 0 else (x_hi, x_lo)
    if prev is not None:
        jump = abs(y - prev[1]) + abs(a - prev[0])
        t += max(jump / args.jump_speed, 1.0e-5)
        rows.append((t, a, y, 0.0, 0))
    length = abs(b - a)
    nseg = max(int(round(length / args.sample_step)), 1)
    for s in range(1, nseg + 1):
        frac = s / nseg
        t += (length / nseg) / args.speed
        rows.append((t, a + frac * (b - a), y, args.power, 1))
    prev = (b, y)

args.output.parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "w") as f:
    f.write("time,x,y,z,power,laser_on,layer,hatch,mode,front_coord\n")
    for k, (tt, x, y, p, on) in enumerate(rows):
        f.write(f"{tt:.9f},{x:.9e},{y:.9e},{args.z:.9e},{p},{on},"
                f"1,{k},scan,{args.z:.9e}\n")

print(f"wrote {args.output}: {len(rows)} rows, {n}/{n_full} tracks, "
      f"P={args.power} W, v={args.speed*1e3:.0f} mm/s, hatch={args.hatch*1e6:.0f} um, "
      f"t_end={t*1e3:.3f} ms")
