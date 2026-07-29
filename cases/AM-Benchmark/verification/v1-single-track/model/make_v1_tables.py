#!/usr/bin/env python3
"""Generate V1 material property tables (Balbaa2022 parity, IN625).

Provenance: inputs/balbaa-model.json (Balbaa2022 Table 1) + registered
reconstructions in inputs/deviations.yaml:
  - D-V1-01: bulk ks/cp linear between the printed range endpoints,
    anchored at (293.15 K, low end) and (solidus 1563 K, high end).
  - D-V1-08: N2 conductivity linear between printed endpoints with the
    same anchors (0.02604 W/mK at ~300 K matches standard N2 at 300 K).
  - Powder-bed conductivity: Sih-Barlow, Balbaa Eq 12-13, with the
    ln(ks/kg) term restored (dropped by PDF text extraction; standard
    Sih & Barlow form). Emissivity of powder bed from Eq 9-11.
  - D-V1-15: cp_powder = (1-phi)*cp_s AS PRINTED (Eq 14); combined with
    rho_powder = (1-phi)*rho_s this double-counts porosity in the
    volumetric heat capacity. Reproduced as printed, flagged.

Zero-calibration: no constant here is tuned; every number traces to the
paper or a registered deviation.
"""
import csv
import math
from pathlib import Path

OUT = Path(__file__).parent / "tables"
OUT.mkdir(exist_ok=True)

# Balbaa2022 Table 1 (inputs/balbaa-model.json)
T_LO, T_SOLIDUS = 293.15, 1563.0
KS_LO, KS_HI = 10.1, 31.6          # W/mK
CP_LO, CP_HI = 419.0, 657.0        # J/kgK
KG_LO, KG_HI = 0.02604, 0.0947     # W/mK (N2)
PHI = 0.4
DP = 27.0e-6                       # m
ES_BULK = 0.4                      # bulk emissivity
SIGMA_SB = 5.670374419e-8

def lin(T, lo, hi, Tlo=T_LO, Thi=T_SOLIDUS):
    T = min(max(T, Tlo), Thi)
    return lo + (hi - lo) * (T - Tlo) / (Thi - Tlo)

# Powder-bed emissivity, Balbaa Eq 9-11 (Sih-Barlow radiation emissivity)
AH = 0.908 * PHI**2 / (1.908 * PHI**2 - 2 * PHI + 1)
x = 3.082 * ((1 - PHI) / PHI) ** 2
EH = (ES_BULK * (2 + x)) / (ES_BULK * (1 + x) + 1)
E_PB = AH * EH + (1 - AH) * ES_BULK

def k_powder_bed(T):
    """Sih-Barlow Eq 12 with Eq 13 radiation term."""
    kg = lin(T, KG_LO, KG_HI)
    ks = lin(T, KS_LO, KS_HI)
    kr = 4.0 * E_PB * SIGMA_SB * T**3 * DP / (1.0 - 0.132 * E_PB)
    s = math.sqrt(1.0 - PHI)
    r = kg / ks
    term_a = (1.0 - s) * (1.0 + PHI * kr / kg)
    term_b = s * ((2.0 / (1.0 - r)) * ((1.0 / (1.0 - r)) * math.log(1.0 / r) - 1.0) + kr / kg)
    return kg * (term_a + term_b)

GRID = [293.15, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1563.0]

# Balbaa2022 Fig 1 powder curve, digitized 2026-07-29 from a 400 dpi render
# (T in C on the axis -> K here). D-V1-20: the printed Sih-Barlow equations
# (Eq 12-13, ln-term restored) give ~half these values; the figure reflects
# the model as run, so it is the primary k_powder source. +/-0.15 W/mK
# read-off uncertainty.
FIG1_POWDER = [
    (293.15, 0.55), (473.15, 0.62), (673.15, 0.78), (873.15, 0.92),
    (1073.15, 1.15), (1273.15, 1.32), (1473.15, 1.48), (1523.15, 1.55),
]

def write(name, rows, unit, source):
    path = OUT / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "value", "source"])
        for T, v in rows:
            w.writerow([f"{T:.2f}", f"{v:.6g}", source])
    print(f"{path.name}: {len(rows)} rows, {rows[0][1]:.4g}..{rows[-1][1]:.4g} {unit}")

write("k_solid.csv", [(T, lin(T, KS_LO, KS_HI)) for T in GRID], "W/mK",
      "Balbaa2022 Table1 range endpoints; linear reconstruction D-V1-01")
write("cp_solid.csv", [(T, lin(T, CP_LO, CP_HI)) for T in GRID], "J/kgK",
      "Balbaa2022 Table1 range endpoints; linear reconstruction D-V1-01")
write("k_powder.csv", FIG1_POWDER, "W/mK",
      "Balbaa2022 Fig 1 powder curve digitized (primary, D-V1-20)")
write("k_powder_sihbarlow.csv", [(T, k_powder_bed(T)) for T in GRID], "W/mK",
      "Sih-Barlow Eq12-13 as printed (ln restored); sensitivity variant, D-V1-20")
write("cp_powder.csv", [(T, (1 - PHI) * lin(T, CP_LO, CP_HI)) for T in GRID], "J/kgK",
      "Balbaa2022 Eq14 as printed ((1-phi)*cp_s); D-V1-15 double-count flag")

print(f"\nderived constants:")
print(f"  powder-bed emissivity e_pb = {E_PB:.4f} (AH={AH:.4f}, eH={EH:.4f})")
print(f"  rho_solid endpoints as printed: 8453 (RT) .. 7925 (solidus) kg/m3")
print(f"  rho_powder = (1-phi)*rho_s = {0.6*8453:.1f} kg/m3 (at RT anchor)")
print(f"  k_powder(293)={k_powder_bed(293.15):.4f}, k_powder(1563)={k_powder_bed(1563.0):.4f} W/mK")
print(f"  solid/powder k ratio at 1000 K: {lin(1000, KS_LO, KS_HI)/k_powder_bed(1000):.1f}x (Balbaa Fig 1: ~20x)")
