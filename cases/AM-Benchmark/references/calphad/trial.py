"""CALPHAD trial: IN625 (AMB2018-01 mill cert) with mc_ni_v2036 + pycalphad.

Outputs: equilibrium solidus/liquidus, latent heat, Scheil path, cp(T).
Gates:  Special Metals melting range 1288-1349 C (1561-1622 K),
        Ghosh 2018 CALPHAD solidus 1587 K, Gen3 CSP cp(T).
"""
import json
import time
import warnings

import numpy as np
from pycalphad import Database, equilibrium, variables as v

warnings.filterwarnings('ignore')

DB = '/tmp/calphad/mc_ni_v2036_pycalphad.tdb'
OUT = '/tmp/calphad/trial_results.json'

dbf = Database(DB)

# AMB2018-01 IN625 mill cert (EOS lot M421601, ICP wt%); traces Co/Mn/Ta/P/S/O/N dropped
wt = {'CR': 20.61, 'MO': 8.82, 'NB': 3.97, 'FE': 0.81,
      'TI': 0.39, 'AL': 0.30, 'SI': 0.18, 'C': 0.02}
M = {'NI': 58.6934, 'CR': 51.9961, 'MO': 95.95, 'NB': 92.90637,
     'FE': 55.845, 'TI': 47.867, 'AL': 26.9815, 'SI': 28.085, 'C': 12.011}
wt['NI'] = 100.0 - sum(wt.values())
moles = {el: wt[el] / M[el] for el in wt}
tot = sum(moles.values())
x = {el: moles[el] / tot for el in moles}
molar_mass = sum(x[el] * M[el] for el in x) * 1e-3  # kg/mol

comps = sorted(x) + ['VA']
melt_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'DELTA', 'LAVES', 'LAV_C14',
               'M23C6', 'M6C', 'GRAPHITE']
solid_extra = ['GAMMA_PRIME', 'GAMMA_DP', 'SIGMA', 'MU_PHASE', 'P_PHASE',
               'NI2CR', 'HCP_A3', 'M7C3', 'M12C', 'KSI_CARBIDE']
all_phases = melt_phases + [p for p in solid_extra if p in dbf.phases]
conds_x = {v.X(el): x[el] for el in x if el != 'NI'}

results = {'composition_wt': wt, 'composition_x': x, 'molar_mass_kg_mol': molar_mass}


def liquid_fraction(eq):
    ph = eq.Phase.values.squeeze()
    npf = np.nan_to_num(eq.NP.values.squeeze())
    return np.array([npf[i][ph[i] == 'LIQUID'].sum() for i in range(ph.shape[0])])


# ---------- Stage 1: melting window ----------
print('[stage 1] equilibrium scan 1450-1690 K step 2 K ...', flush=True)
t0 = time.time()
eqA = equilibrium(dbf, comps, melt_phases,
                  {v.T: (1450, 1690, 2), v.P: 101325, v.N: 1, **conds_x},
                  output='HM')
TA = eqA.T.values
fliq = liquid_fraction(eqA)
HMA = eqA.HM.values.squeeze()
print(f'  done in {time.time()-t0:.0f}s', flush=True)

solid_mask = fliq < 1e-4
liq_mask = fliq > 1 - 1e-4
T_solidus = TA[solid_mask].max() if solid_mask.any() else None
T_liquidus = TA[~liq_mask].max() + 2 if (~liq_mask).any() else None

# latent heat: gap between linear extrapolations of solid/liquid HM branches
Ls = None
if T_solidus and T_liquidus:
    ps = np.polyfit(TA[solid_mask][-10:], HMA[solid_mask][-10:], 1)
    pl = np.polyfit(TA[liq_mask][:10], HMA[liq_mask][:10], 1)
    Tm = 0.5 * (T_solidus + T_liquidus)
    Ls = (np.polyval(pl, Tm) - np.polyval(ps, Tm)) / molar_mass  # J/kg
results['equilibrium'] = {
    'T_solidus_K': float(T_solidus) if T_solidus else None,
    'T_liquidus_K': float(T_liquidus) if T_liquidus else None,
    'latent_heat_J_kg': float(Ls) if Ls else None,
    'T_K': TA.tolist(), 'f_liquid': fliq.tolist(), 'HM_J_mol': HMA.tolist(),
}
print(f'  equilibrium solidus  = {T_solidus:.0f} K ({T_solidus-273.15:.0f} C)', flush=True)
print(f'  equilibrium liquidus = {T_liquidus:.0f} K ({T_liquidus-273.15:.0f} C)', flush=True)
print(f'  latent heat          = {Ls/1e3:.0f} kJ/kg', flush=True)
json.dump(results, open(OUT, 'w'))

# ---------- Stage 2: Scheil ----------
print('[stage 2] Scheil solidification ...', flush=True)
t0 = time.time()
from scheil import simulate_scheil_solidification
sol = simulate_scheil_solidification(
    dbf, comps, melt_phases, conds_x, float(T_liquidus) + 15.0,
    step_temperature=1.0)
Tsch = np.array(sol.temperatures)
fsol = 1.0 - np.array(sol.fraction_liquid)
print(f'  done in {time.time()-t0:.0f}s, {len(Tsch)} steps', flush=True)
sch_sol_98 = float(np.interp(0.98, fsol, Tsch))
sch_sol_end = float(Tsch.min())
cum = {p: float(np.nansum(a)) for p, a in sol.cum_phase_amounts.items()}
results['scheil'] = {
    'T_K': Tsch.tolist(), 'fraction_solid': fsol.tolist(),
    'T_at_fs098_K': sch_sol_98, 'T_terminal_K': sch_sol_end,
    'cum_phase_amounts': cum,
}
print(f'  Scheil fs=0.98 at {sch_sol_98:.0f} K ({sch_sol_98-273.15:.0f} C); '
      f'terminal {sch_sol_end:.0f} K ({sch_sol_end-273.15:.0f} C)', flush=True)
print(f'  solid phases formed: { {p: round(f,4) for p, f in cum.items() if f > 1e-4} }', flush=True)
json.dump(results, open(OUT, 'w'))

# ---------- Stage 3: solid-state cp ----------
print('[stage 3] equilibrium scan 400-1450 K step 15 K for cp ...', flush=True)
t0 = time.time()
eqB = equilibrium(dbf, comps, all_phases,
                  {v.T: (400, 1455, 15), v.P: 101325, v.N: 1, **conds_x},
                  output='HM')
TB = eqB.T.values
HMB = eqB.HM.values.squeeze()
cp_mass = np.gradient(HMB, TB) / molar_mass  # J/(kg K)
print(f'  done in {time.time()-t0:.0f}s', flush=True)
results['cp'] = {'T_K': TB.tolist(), 'HM_J_mol': HMB.tolist(),
                 'cp_J_kgK': cp_mass.tolist()}
for TC_ in (260, 400, 600, 800, 1000):
    print(f'  cp({TC_} C) = {np.interp(TC_+273.15, TB, cp_mass):.0f} J/(kg K)', flush=True)
json.dump(results, open(OUT, 'w'))
print('ALL DONE', flush=True)
