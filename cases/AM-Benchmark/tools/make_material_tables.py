"""Main-case material tables + config generator (L0 wiring).

Every row cites its archived source. L0-provisional items are labeled
PROVISIONAL-L0 and must be replaced before L1 conclusions:
  - alpha constant (Special Metals RT mean CTE)
  - yield/hardening ramps (mill-cert RT anchor, linear decay)
  - k_powder linear between the two Zhang endpoints, flat extrapolation
  - linear mushy latent release (Scheil fs(T) integration is a D.6 item
    deferred to L1; runner supports linear solidus-liquidus only)
Sources: Kaschnitz IJT 2019 (k, rho), CALPHAD D-08 gamma-frozen (cp),
Zhang 2019 (k_powder, phi_void), D-12 (emissivity), Special Metals /
mill cert (E, nu, yield anchor), D-07/A.3 (temperatures).
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
OUT = os.path.join(CASE, 'derived', 'material')
os.makedirs(OUT, exist_ok=True)

KJ = json.load(open(os.path.join(
    CASE, 'references', 'material',
    'Kaschnitz2019_IJT_tables2-3_thermophysical.json')))
TRIAL = json.load(open(os.path.join(
    CASE, 'references', 'calphad', 'trial_results.json')))


def write_csv(name, rows):
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['T', 'value', 'source'])
        w.writerows(rows)
    print(f'{name}: {len(rows)} rows')
    return path


# ---- k_solid(T): Kaschnitz IJT Tables 2+3 ----
k_rows = []
for r in KJ['table2_low_temperature'] + KJ['table3_high_temperature']:
    if 'k_W_mK' in r:
        src = 'Kaschnitz2019 IJT Table 2/3'
        if r.get('k_W_mK_interextrapolated'):
            src += ' (inter/extrapolated flag)'
        k_rows.append([round(r['T_C'] + 273.15, 2), r['k_W_mK'], src])
k_solid = write_csv('k_solid.csv', k_rows)

# ---- cp_solid(T): D-08 gamma-frozen CALPHAD, 260 C bump smoothed ----
cp = TRIAL['cp_gamma_frozen']
cp_rows = []
prev = None
for T, c in zip(cp['T_K'], cp['cp_J_kgK']):
    if 510.0 <= T <= 550.0:      # registered smoothing of the 260 C bump
        continue
    cp_rows.append([round(T, 2), round(c, 1),
                    'D-08 gamma-frozen CALPHAD (mc_ni_v2036); 510-550 K '
                    'bump excluded per D.6 caveat 3'])
# flat extension down to preheat range (variation < 3 % over 293-400 K)
cp_rows.insert(0, [293.15, cp_rows[0][1],
                   'flat extension of the 400 K gamma-frozen value; '
                   'PROVISIONAL-L0'])
cp_solid = write_csv('cp_solid.csv', cp_rows)

# ---- k_powder(T): Zhang 2019 measured endpoints ----
kp_rows = [
    [373.15, 0.65, 'Zhang2019 measured (100 C)'],
    [773.15, 1.02, 'Zhang2019 measured (500 C)'],
    [1552.0, 1.02, 'flat extrapolation above 500 C; PROVISIONAL-L0 '
                   '(registered E-table caveat)'],
]
k_powder = write_csv('k_powder.csv', kp_rows)

# ---- mechanics tables ----
E_rows = [
    [294.15, 207.5e9, 'Special Metals bulletin (21 C)'],
    [1144.15, 147.5e9, 'Special Metals bulletin (871 C)'],
    [1552.0, 20.0e9, 'ramp to weak solid at solidus; PROVISIONAL-L0'],
]
E_table = write_csv('E_table.csv', E_rows)
alpha_rows = [
    [293.15, 1.28e-5, 'Special Metals RT mean CTE; constant '
                      'PROVISIONAL-L0 (bulletin curve at L1)'],
    [1552.0, 1.28e-5, 'constant PROVISIONAL-L0'],
]
alpha_table = write_csv('alpha_table.csv', alpha_rows)
poisson_rows = [
    [294.15, 0.278, 'Special Metals bulletin (21 C)'],
    [1144.15, 0.336, 'Special Metals bulletin (871 C)'],
]
poisson_table = write_csv('poisson_table.csv', poisson_rows)
yield_rows = [
    [293.15, 630e6, 'mill cert M421601 laser-processed Rp0.2; anchor'],
    [1046.15, 300e6, 'linear decay; PROVISIONAL-L0 (D-11 MaCTO tables '
                     'replace this)'],
    [1273.15, 50e6, 'PROVISIONAL-L0'],
    [1552.0, 5e6, 'floor at solidus; PROVISIONAL-L0'],
]
yield_table = write_csv('yield_table.csv', yield_rows)
hard_rows = [
    [293.15, 2.0e9, 'PROVISIONAL-L0 linear hardening'],
    [1552.0, 0.1e9, 'PROVISIONAL-L0'],
]
hardening_table = write_csv('hardening_table.csv', hard_rows)

# ---- config ----
RHO_S = 8453.0                    # Kaschnitz IJT Table 3, 20 C
PHI_VOID_MID = 0.48               # Zhang mid; phi_pack = 0.52 (D-03)
config = {
    '_comment': 'AMB2018-01 main-case material config, L0 wiring. '
                'Sources per table row; PROVISIONAL-L0 labels mark items '
                'that must be replaced before L1 conclusions. Flash '
                'deposition (laser power 0, activation reset at liquidus) '
                'per D-02; energy-input fidelity is the L1 topic.',
    'material_name': 'IN625 AMB2018-01 (D-01/D-08 genealogy)',
    'units': 'SI',
    'rho_solid': RHO_S,
    'rho_powder': round((1 - PHI_VOID_MID) * RHO_S, 1),
    'rho_liquid': 7914.0,
    'k_table_solid': k_solid,
    'cp_table_solid': cp_solid,
    'k_table_powder': k_powder,
    'cp_table_powder': cp_solid,
    'conductivity_liquid': 32.3,
    'cp_liquid': 663.0,
    '_liquid_note': 'Kaschnitz 1290 C values held constant into the melt; '
                    'PROVISIONAL-L0',
    'solidus_temperature': 1552.0,
    'liquidus_temperature': 1630.0,
    'latent_heat': 259000.0,
    '_melting_note': 'D-08 equilibrium window + latent heat; linear mushy '
                     'release (Scheil fs(T) deferred to L1, deviation '
                     'registered)',
    'absorptivity': 0.62,
    'emissivity': 0.46,
    '_emissivity_note': 'D-12 powder mid-value (surface is powder-covered '
                        'under D-04); L1 runs the [0.24, 0.68] extremes',
    'convection_h': 20.0,
    '_convection_note': 'V1 precedent constant; PROVISIONAL-L0',
    'E_table': E_table,
    'alpha_table': alpha_table,
    'poisson_table': poisson_table,
    'yield_table': yield_table,
    'hardening_table': hardening_table,
    'mechanics_model': 'j2_plastic',
    'powder_mode': 'powder',
}
cfg = os.path.join(OUT, 'amb_material_config_L0.json')
with open(cfg, 'w') as f:
    json.dump(config, f, indent=1)
    f.write('\n')
print(f'wrote {cfg}')
