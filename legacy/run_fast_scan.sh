#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 维护入口 1/2：快速扫描（fast scan）—— 全件 91 层，约 1 小时
#
# 层聚合 flash 模式：无激光逐道扫描，每个 1mm 宏观层整层以松弛温度热激活，
# 保留 10s 层间 recoat 冷却 + 600s 终冷 + release。用于全件应力/变形形态、
# 参数标定扫描（k_s / T_relax）与快速迭代。物理内核 = v05（增量 J2 + ε_p）。
#
# 用法:  bash 159_local/run_fast_scan.sh [附加参数覆盖]
# 输出:  output/fastscan_flash_v05_full91_<时间戳>/
# 详见:  159_local/README.md 与 159_local/v04/CALIBRATION_KNOBS.md
# ============================================================================

exec bash "$(dirname "$0")/v05/run_fastscan_flash_v05.sh" "$@"
