"""D-12 emissivity bracket generator (Sih & Barlow 1995, verified original).

Source of the model: Sih & Barlow, "Emissivity of Powder Beds", SFF Symposium
1995, pp. 402-409 (archived references/docs/SihBarlow1995_emissivity_SFF.pdf).
Formula chain verified against the original AND re-derived from its Eqs 5-11:

    eps_pb = A_H * eps_H + (1 - A_H) * eps_s                      (Eq 4)
    A_H    = 0.908 phi^2 / (1.908 phi^2 - 2 phi + 1)              (Eq 7)
    eps_H  = eps_s (2 + 3.082 x) / (eps_s (1 + 3.082 x) + 1)      (Eq 12)
    x      = ((1 - phi) / phi)^2

phi is the FRACTIONAL POROSITY (void fraction; original: "phi = 1 - p").
NOTE the ledger notation collision: D-03's phi is the PACKING fraction —
here it is phi_void. Balbaa 2022 Eq 11 renders the denominator's "+1" as
"-1", which is unphysical (eps_H > 1); the original says +1.

Inputs: eps_s bracket [0.12 (Kieruj polished, Zhang's value), 0.50 (oxidized
envelope)]; phi_void = Zhang 2019 measured powder porosity 0.403-0.557.
"""
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)

EPS_S = [0.12, 0.50]
PHI_VOID = [0.403, 0.557]


def sih_barlow_eps_powder(eps_s, phi):
    x = ((1.0 - phi) / phi) ** 2
    a_h = 0.908 * phi ** 2 / (1.908 * phi ** 2 - 2.0 * phi + 1.0)
    eps_h = eps_s * (2.0 + 3.082 * x) / (eps_s * (1.0 + 3.082 * x) + 1.0)
    return a_h * eps_h + (1.0 - a_h) * eps_s, a_h, eps_h


corners = []
for es, ph in itertools.product(EPS_S, PHI_VOID):
    ep, ah, eh = sih_barlow_eps_powder(es, ph)
    corners.append({'eps_solid': es, 'phi_void': ph,
                    'A_H': round(ah, 4), 'eps_hole': round(eh, 4),
                    'eps_powder': round(ep, 4)})
    assert 0.0 < eh <= 1.0 and 0.0 < ep < 1.0

lo = min(c['eps_powder'] for c in corners)
hi = max(c['eps_powder'] for c in corners)

out = {
    'schema_version': 'ambench.emissivity-bracket/1',
    'decision': 'D-12 (2026-07-30): formula route approved; ranges revised '
                'per user review (phi definition confirmed = porosity; '
                'Balbaa Eq-11 sign error corrected against the original)',
    'model_source': {
        'citation': 'Sih & Barlow, Emissivity of Powder Beds, SFF 1995',
        'archive': 'references/docs/SihBarlow1995_emissivity_SFF.pdf',
        'verification': 'formulas transcribed from the original and '
                        're-derived from its Eqs 5-11 (3.082 = 4*0.514*3/2)',
    },
    'inputs': {
        'eps_solid_bracket': EPS_S,
        'eps_solid_provenance': 'lower: Kieruj 2016 polished (Zhang 2019 '
                                'adopted); upper: oxidized envelope '
                                '(Sensors 2024, chamber ~0.5 % O2)',
        'phi_void_bracket': PHI_VOID,
        'phi_void_provenance': 'Zhang 2019 measured powder porosity '
                               '40.3-55.7 % (100-500 C)',
        'notation': 'phi_void = porosity (Sih-Barlow phi); the D-03 packing '
                    'fraction is phi_pack = 1 - phi_void',
    },
    'corners': corners,
    'eps_powder_interval': [round(lo, 3), round(hi, 3)],
    'observation': 'the interval is dominated by eps_solid; phi_void moves '
                   'eps_powder by only 0.01-0.06 at fixed eps_solid',
    'sensitivity_plan': 'two L1 thermal runs at the all-low and all-high '
                        'extremes (D-12); freeze mid-values if interpass '
                        'delta-T < 2 K, else propagate as uncertainty band',
}

dst = os.path.join(CASE, 'derived', 'emissivity-bracket.json')
with open(dst, 'w') as f:
    json.dump(out, f, indent=1)
    f.write('\n')
for c in corners:
    print(c)
print(f'eps_powder interval: [{lo:.3f}, {hi:.3f}]')
print(f'wrote {dst}')
