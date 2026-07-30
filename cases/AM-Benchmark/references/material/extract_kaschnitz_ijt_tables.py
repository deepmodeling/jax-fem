"""Transcribe Kaschnitz, Kaschnitz & Heugenhauser (2019), Int. J. Thermophys.
40:27, Tables 2-3 (IN625 thermophysical data) into machine-readable JSON.

Parses the PDF text stream directly (no hand typing). Rows are sequences of
numeric tokens following the table header; the 'a' suffix marks values the
paper labels "Inter/extrapolated".

Source PDF: ../docs/Kaschnitz2019_IN625-resistivity-conductivity_IJT-40-27.pdf
"""
import hashlib
import json
import os
import re

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, '..', 'docs',
                   'Kaschnitz2019_IN625-resistivity-conductivity_IJT-40-27.pdf')
OUT = os.path.join(HERE, 'Kaschnitz2019_IJT_tables2-3_thermophysical.json')

doc = pymupdf.open(PDF)


def parse_rows(lines, ncols):
    """Group a flat token stream into rows of ncols; first token is T (int)."""
    rows = []
    i = 0
    while i < len(lines):
        tok = re.sub(r'\s+', '', lines[i].replace('−', '-'))
        if re.fullmatch(r'-?\d+', tok):
            T = int(tok)
            vals, flags, j = [], [], i + 1
            while j < len(lines) and len(vals) < ncols - 1:
                v = re.sub(r'\s+', '', lines[j].replace('−', '-'))
                m = re.fullmatch(r'(-?\d+\.?\d*)(a?)', v)
                if not m:
                    break
                vals.append(float(m.group(1)))
                flags.append(bool(m.group(2)))
                j += 1
            rows.append((T, vals, flags))
            i = j
        else:
            i += 1
    return rows


def clean_lines(text, start_marker, stop_marker=None):
    i = text.find(start_marker)
    seg = text[i + len(start_marker):]
    if stop_marker:
        j = seg.find(stop_marker)
        if j >= 0:
            seg = seg[:j]
    return [ln.strip() for ln in seg.splitlines() if ln.strip()]


# Table 2 (p5): T, cp, rho, alpha, k — variable trailing columns (early rows
# lack rho/alpha/k) and negative integer temperatures are indistinguishable
# from values by shape. Anchor rows on the known T sequence -170..10 step 10.
def parse_table2(lines):
    toks = [re.sub(r'\s+', '', l.replace('−', '-')) for l in lines]
    temps = [str(t) for t in range(-170, 20, 10)]
    anchors, ti = [], 0
    for k, tok in enumerate(toks):
        if ti < len(temps) and tok == temps[ti]:
            anchors.append(k)
            ti += 1
    assert ti == len(temps), f'found {ti}/{len(temps)} Table-2 temperatures'
    rows = []
    for n, a in enumerate(anchors):
        end = anchors[n + 1] if n + 1 < len(anchors) else len(toks)
        vals = [float(v) for v in toks[a + 1:end]
                if re.fullmatch(r'-?\d+\.?\d*', v)]
        rows.append((int(toks[a]), vals, [False] * len(vals)))
    return rows


t2_lines = clean_lines(doc[4].get_text(), 'conductivity \n(W·K−1·m−1)')
t2 = parse_table2(t2_lines)

# Table 3 (p6): T, cp, rho, alpha, k, rho_el — full 5 values per row.
t3_lines = clean_lines(doc[5].get_text(), 'resistivity \n(μΩ·m)')
t3 = parse_rows(t3_lines, 6)

COLS2 = ['cp_J_gK', 'density_kg_m3', 'diffusivity_mm2_s', 'k_W_mK']
COLS3 = COLS2 + ['resistivity_uOhm_m']


def rows_to_records(rows, cols):
    recs = []
    for T, vals, flags in rows:
        # early Table-2 rows have fewer values; align from the left (cp first)
        rec = {'T_C': T}
        for name, v, fl in zip(cols, vals, flags):
            rec[name] = v
            if fl:
                rec[name + '_interextrapolated'] = True
        recs.append(rec)
    return recs


payload = {
    'source': {
        'citation': 'E. Kaschnitz, L. Kaschnitz, S. Heugenhauser, '
                    'Int. J. Thermophys. 40:27 (2019)',
        'doi': '10.1007/s10765-019-2490-8',
        'pdf': os.path.basename(PDF),
        'pdf_sha256': hashlib.sha256(open(PDF, 'rb').read()).hexdigest(),
        'acquired': 'user-supplied 2026-07-30',
    },
    'material': 'IN625 (wrought, see paper Sec. 2.1 for specimen provenance)',
    'methods': {
        'cp': 'DSC (2x NETZSCH DSC 404), EN 821-3, 10 K/min, argon; expanded '
              'uncertainty +-3 % (low/medium T), +-5 % above 1100 C',
        'density': 'rho0 by Archimedean balance at RT; rho(T) = rho0/(1+dl/l0)^3 '
                   'with dilatometry per DIN 51045-1 (details in Heugenhauser & '
                   'Kaschnitz, HTHP 48, the companion density paper)',
        'diffusivity': 'laser flash (2x NETZSCH LFA 427), EN 821-2, corrected '
                       'for thermal expansion; +-4 % up to 1000 C, +-5 % above',
        'k': 'calculated: k = a0*rho0*cp/(1+dl/l0)',
        'resistivity': 'millisecond pulse heating, corrected for thermal '
                       'expansion (true geometry)',
    },
    'notes': [
        'cp kink 500-620 C on heating AND cooling: precipitate '
        'formation/dissolution (matches the Gen3 CSP cp jump 580-620 C)',
        'values flagged _interextrapolated are printed with superscript a '
        '("Inter/extrapolated") in the paper',
        'solidus quoted by the paper: 1295 C (dilatometry range endpoint)',
        'liquid/mushy density is NOT in this paper - only in HTHP 48 '
        '(ref [12], still not acquired)',
    ],
    'table2_low_temperature': rows_to_records(t2, COLS2),
    'table3_high_temperature': rows_to_records(t3, COLS3),
}

with open(OUT, 'w') as f:
    json.dump(payload, f, indent=1)
    f.write('\n')

n2, n3 = len(payload['table2_low_temperature']), len(payload['table3_high_temperature'])
print(f'table2: {n2} rows ({payload["table2_low_temperature"][0]["T_C"]} to '
      f'{payload["table2_low_temperature"][-1]["T_C"]} C)')
print(f'table3: {n3} rows ({payload["table3_high_temperature"][0]["T_C"]} to '
      f'{payload["table3_high_temperature"][-1]["T_C"]} C)')
# spot checks against visually-read values
t3r = {r['T_C']: r for r in payload['table3_high_temperature']}
assert t3r[20]['density_kg_m3'] == 8453 and t3r[20]['cp_J_gK'] == 0.419
assert t3r[1250]['k_W_mK'] == 31.6 and t3r[1290]['density_kg_m3'] == 7914
assert t3r[600]['cp_J_gK'] == 0.575 and t3r[600].get('k_W_mK_interextrapolated')
t2r = {r['T_C']: r for r in payload['table2_low_temperature']}
assert t2r[-170] == {'T_C': -170, 'cp_J_gK': 0.244}
assert t2r[-150]['density_kg_m3'] == 8501 and 'diffusivity_mm2_s' not in t2r[-150]
assert t2r[-120]['diffusivity_mm2_s'] == 2.62 and t2r[-120]['k_W_mK'] == 7.37
assert t2r[10]['density_kg_m3'] == 8456 and t2r[10]['k_W_mK'] == 9.95
print('spot checks OK')
