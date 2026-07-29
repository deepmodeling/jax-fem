"""Survey top-level commands in the TDB, then write a pycalphad-compatible copy.

Only strips commands pycalphad's grammar does not recognize (MatCalc metadata);
never touches ELEMENT/SPECIES/PHASE/CONSTITUENT/FUNCTION/PARAMETER/TYPE_DEFINITION.
"""
import re
from collections import Counter

SRC = '/tmp/calphad/mc_ni_v2036_utf8.tdb'
DST = '/tmp/calphad/mc_ni_v2036_pycalphad.tdb'

# pycalphad-recognized TDB commands (io/tdb.py grammar)
SUPPORTED_PREFIXES = (
    'ELEMENT', 'SPECIES', 'TYPE_DEF', 'FUNCTION', 'DEFINE_SYSTEM_DEFAULT',
    'DEFAULT_COMMAND', 'DATABASE_INFO', 'VERSION_DATE', 'REFERENCE_FILE',
    'ADD_REFERENCES', 'LIST_OF_REFERENCES', 'TEMPERATURE_LIMITS', 'PHASE',
    'CONSTITUENT', 'ASSESSED_SYSTEMS', 'PARAMETER',
)

# pycalphad TDB_PARAM_TYPES (io/tdb_keywords.py, v0.11.2)
SUPPORTED_PARAM_TYPES = {
    'G', 'L', 'TC', 'NT', 'BMAGN', 'GD', 'THETA',
    'V0', 'VA', 'VC', 'VK', 'VISC', 'ELRS', 'THCD', 'SIGM', 'XI',
    'MQ', 'MF', 'DQ', 'DF', 'VS',
}

txt = open(SRC).read()

# Work line-wise: keep comment lines ($...) as-is, group non-comment text into
# '!'-terminated commands.
out_parts = []
survey = Counter()
dropped = Counter()

# Split the whole file into segments on '!', but preserve comments.
# Simpler robust approach: remove comment lines first for command detection,
# but reconstruct output from commands only (comments are not needed by pycalphad).
lines = [l for l in txt.splitlines() if not l.lstrip().startswith('$')]
body = '\n'.join(lines)
commands = body.split('!')

repairs = 0
param_survey = Counter()
for cmd in commands:
    # upstream typo: temperature limit '6000.00.00' (2 occurrences, comments aside)
    if '6000.00.00' in cmd:
        cmd = cmd.replace('6000.00.00', '6000.00')
        repairs += 1
    tok = cmd.split()
    if not tok:
        continue
    kw = tok[0].upper()
    # MatCalc abbreviations -> full keywords pycalphad expects
    if kw in ('CONST', 'PARAM'):
        full = {'CONST': 'CONSTITUENT', 'PARAM': 'PARAMETER'}[kw]
        cmd = re.sub(r'^(\s*)' + kw, r'\1' + full, cmd, count=1,
                     flags=re.IGNORECASE)
        kw = full
    survey[kw] += 1
    if not any(kw.startswith(p) for p in SUPPORTED_PREFIXES):
        dropped[kw] += 1
        continue
    if kw.startswith('PARAM'):
        m = re.match(r'\s*PARAM\w*\s+([A-Z0-9]+)\s*\(', cmd, re.IGNORECASE)
        ptype = m.group(1).upper() if m else '???'
        param_survey[ptype] += 1
        if ptype not in SUPPORTED_PARAM_TYPES:
            dropped[f'PARAMETER:{ptype}'] += 1
            continue
    out_parts.append(cmd.rstrip() + ' !')

print('--- parameter types ---')
for k, n in sorted(param_survey.items()):
    print(f'  {k}: {n}')

print('--- command survey ---')
for k, n in sorted(survey.items()):
    print(f'  {k}: {n}')
print('--- dropped ---')
for k, n in sorted(dropped.items()):
    print(f'  {k}: {n}')

with open(DST, 'w') as f:
    f.write('\n'.join(out_parts) + '\n')
print(f'typo repairs (6000.00.00 -> 6000.00): {repairs}')
print(f'wrote {DST}')
