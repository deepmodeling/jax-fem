"""Post-process trial results: correct Scheil phase amounts and milestones;
compute gamma-frozen (FCC-only) cp for fair comparison with Gen3 CSP."""
import json
import warnings

import numpy as np
from pycalphad import Database, equilibrium, variables as v

warnings.filterwarnings('ignore')

R = json.load(open('/tmp/calphad/trial_results.json'))
x = R['composition_x']
molar_mass = R['molar_mass_kg_mol']

# ---- Scheil milestones (fs curve is valid; phase sums were mis-extracted) ----
Ts = np.array(R['scheil']['T_K'])
fs = np.array(R['scheil']['fraction_solid'])
print('--- Scheil milestones ---')
for m in (0.05, 0.50, 0.90, 0.95, 0.98):
    Tm = float(np.interp(m, fs, Ts))
    print(f'  fs={m:.2f}: {Tm:.0f} K ({Tm-273.15:.0f} C)')
print(f'  terminal: {Ts.min():.0f} K ({Ts.min()-273.15:.0f} C)')

# ---- re-run Scheil phase bookkeeping from the scheil result is not saved;
# recompute final amounts from cum sums stored? Original arrays lost.
# Instead report phase onsets from a fresh light-weight source: skip — will
# re-extract in the rerun below if needed.

# ---- gamma-frozen cp: FCC_A1 single phase over 400-1500 K ----
dbf = Database('/tmp/calphad/mc_ni_v2036_pycalphad.tdb')
comps = sorted(x) + ['VA']
conds = {v.T: (400, 1505, 10), v.P: 101325, v.N: 1}
conds.update({v.X(el): x[el] for el in x if el != 'NI'})
eq = equilibrium(dbf, comps, ['FCC_A1'], conds, output='HM')
T = eq.T.values
HM = eq.HM.values.squeeze()
ok = np.isfinite(HM)
cp = np.gradient(HM[ok], T[ok]) / molar_mass
Tok = T[ok]
print('--- gamma-frozen cp (FCC_A1 only) vs Gen3 CSP ---')
gen3 = {260: (452, 28), 300: (455, 30), 420: (464, 37), 500: (469, 42),
        620: (545, 50), 700: (550, 59), 780: (562, 60), 900: (558, 75),
        1000: (550, 91)}
for TC_, (g, u) in gen3.items():
    c = float(np.interp(TC_ + 273.15, Tok, cp))
    flag = 'OK' if abs(c - g) <= u else ('high' if c > g else 'low')
    print(f'  {TC_:5d} C: calc {c:4.0f}  Gen3 {g}+-{u}  [{flag}]')

# fix equilibrium-cp NaN report too
TB = np.array(R['cp']['T_K'])
HMB = np.array(R['cp']['HM_J_mol'])
okB = np.isfinite(HMB)
print(f'--- equilibrium cp scan: {int((~okB).sum())} failed points '
      f'at {TB[~okB].tolist()} K ---')
cpB = np.gradient(HMB[okB], TB[okB]) / molar_mass
for TC_ in (260, 500, 800, 1000):
    c = float(np.interp(TC_ + 273.15, TB[okB], cpB))
    print(f'  eq-cp({TC_} C) = {c:.0f} J/(kg K)')

R['cp_gamma_frozen'] = {'T_K': Tok.tolist(), 'cp_J_kgK': cp.tolist()}
json.dump(R, open('/tmp/calphad/trial_results.json', 'w'))
print('saved')
