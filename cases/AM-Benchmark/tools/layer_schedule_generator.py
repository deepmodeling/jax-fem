"""AMB2018-01 layer-clock / scan-schedule generator (decision D-10).

CASE-LOCAL SAMPLE TOOLING - not part of the jax_fem_am package.

Reconstructs the per-layer clock of the AMB2018-01 build bottom-up from the
transcribed JRES timing tables (inputs/scan-timing.json) and the STL-verified
plan geometry. Everything geometry-driven is COMPUTED; the only inferred
elements (registered in the output) are:

  I1. machine dead time (end-of-layer dwell + recoat + overheads)
      = 52 s - computed legs-band plate active time, assumed layer-invariant;
  I2. ridge band (layers 601-624) follows the same infill/contour rules as
      the published bands (its timing was never published);
  I3. inter-part travel uses the midpoint of the published 0.307-0.363 s;
      contour-to-contour travel uses the midpoint of 15.5-25.5 ms.

Closed loop: the computed legs-band plate scan time is reported against the
published "~26 s for all four parts". The residual is REPORTED, never scaled
away (constitution: no calibration).

Line counts are derived as ceil(width/hatch) except where the paper prints
measured per-leg counts (legs band, Table 5) - per the Table-5 misprint
verdict in scan-timing.json.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
ST = json.load(open(os.path.join(CASE, 'inputs', 'scan-timing.json')))

G = ST['global']
SPEED = G['infill_speed_mm_s']            # 800 mm/s
HATCH = G['hatch_spacing_mm']             # 0.10 mm
EDGE = G['infill_edge_offset_mm']         # 0.03 mm per end
TOTAL_LAYERS = G['total_layers']          # 624 (conflict B1: Phan says 625)

ODD = ST['infill_odd_layers_table4']
EVEN = ST['infill_even_layers_table5']
CON = ST['contour_timing_table3']
IPL = ST['inter_part_and_layer']

OFF_LINE_END = ODD['within_feature_laser_off_at_part_ends_ms'][0] / 1e3   # 0.48 ms
OFF_HOP_ODD = ODD['between_features_laser_off_ms'][0] / 1e3               # 1.66 ms
OFF_LINE_EVEN = EVEN['within_feature_line_to_line_laser_off_ms'][0] / 1e3  # 0.48 ms
OFF_HOP_EVEN = EVEN['between_features_laser_off_ms'][0] / 1e3             # 1.78 ms
INTER_PART = sum(IPL['time_between_parts_s']) / 2                          # I3
INTER_CONTOUR = sum(CON['inter_leg_travel_ms']) / 2 / 1e3                  # I3
N_PARTS = 4
LAYER_TIME_LEGS = IPL['layer_time_legs_s']                                 # 52 s
PUBLISHED_SCAN = IPL['scan_time_all_four_parts_s']                         # ~26 s

# ---------------- plan geometry (verified, scan-timing.json + STL) ----------
Y_DEPTH = 5.0
TIP_X_EDGE, TIP_X_MID = 72.55, 75.00     # 45-degree point in plan
LEG_X0 = ST['feature_id_map']            # left edges at z=2.5
LEG_W = {'thick': 5.0, 'thin': 0.5, 'med': 2.5}
LEG_TYPE = {k: ('thick' if k in ('L1', 'L4', 'L7', 'L10') else
                'thin' if k in ('L2', 'L5', 'L8', 'L11') else 'med')
            for k in [f'L{i}' for i in range(1, 13)]}
BASE_X0 = 56.0
RIDGE_X0 = [0.5 + 7.0 * i for i in range(11)]
RIDGE_XMAX = 71.0                        # STL: ridges stop at x = 71.0 (I2)


def part_x_max(y):
    """Right boundary of the part at depth y (the 45-degree tip)."""
    return TIP_X_EDGE + (TIP_X_MID - TIP_X_EDGE) * (1.0 - abs(y - 2.5) / 2.5)


def tip_y_half(x):
    """Half-depth of the part at x inside the tip region."""
    if x <= TIP_X_EDGE:
        return 2.5
    return 2.5 * (TIP_X_MID - x) / (TIP_X_MID - TIP_X_EDGE)


def overhang_growth(layer):
    """Linear interpolation factor 0..1 across the overhang band (published rule)."""
    return (layer - 250) / 100.0


def features_at(layer, band):
    """[(x0, x1_of_y)] per feature, left-to-right, for odd-layer row sweeps.
    x1 may be callable(y) for tip-clipped features."""
    if band in ('legs', 'overhang'):
        g = 0.0 if band == 'legs' else overhang_growth(layer)
        feats = []
        for k in [f'L{i}' for i in range(1, 13)]:
            w0 = LEG_W[LEG_TYPE[k]]
            grow = {'thick': 1.98, 'thin': 1.98, 'med': 1.98}[LEG_TYPE[k]] * g
            x0 = LEG_X0[k] - grow / 2
            feats.append((x0, x0 + w0 + grow))
        bx0 = BASE_X0 - 1.98 * g          # base grows on its left side
        feats.append((bx0, part_x_max))
        return feats
    if band == 'bridge':
        return [(0.0, part_x_max)]
    if band == 'ridges':                  # I2
        return [(x0, min(x0 + 1.0, RIDGE_XMAX)) for x0 in RIDGE_X0]
    raise ValueError(band)


def odd_layer_active(layer, band):
    """x-parallel rows, hopping feature to feature along constant y."""
    feats = features_at(layer, band)
    n_rows = 49                            # published lines_per_feature (odd)
    on = off = 0.0
    for j in range(n_rows):
        y = EDGE + j * HATCH
        for (x0, x1) in feats:
            xr = x1(y) if callable(x1) else x1
            seg = max(0.0, (xr - x0) - 2 * EDGE)
            on += seg / SPEED
        off += OFF_LINE_END + OFF_HOP_ODD * (len(feats) - 1)
    return on, off


# published per-leg even-layer line counts (Table 5, legs band only)
EVEN_LEG_LINES = {'L1': 49, 'L10': 49, 'L4': 50, 'L7': 50,
                  'L2': 4, 'L8': 4, 'L11': 4, 'L5': 5,
                  'L3': 24, 'L9': 24, 'L12': 24, 'L6': 25}


def even_layer_active(layer, band):
    """y-parallel lines, each feature completed before moving on."""
    feats = features_at(layer, band)
    keys = ([f'L{i}' for i in range(1, 13)] + ['base']
            if band in ('legs', 'overhang') else None)
    on = off = 0.0
    for fi, (x0, x1) in enumerate(feats):
        tip_clipped = callable(x1)
        xr = x1(2.5) if tip_clipped else x1
        width = xr - x0
        if band == 'legs' and keys and keys[fi] != 'base':
            n_lines = EVEN_LEG_LINES[keys[fi]]        # published counts
            xs = [x0 + EDGE + i * HATCH for i in range(n_lines)]
        else:
            n_lines = math.ceil(width / HATCH)        # misprint verdict rule
            xs = [x0 + EDGE + i * HATCH for i in range(n_lines)]
        for x in xs:
            depth = 2 * tip_y_half(x) if tip_clipped else Y_DEPTH
            seg = max(0.0, depth - 2 * EDGE)
            on += seg / SPEED
            off += OFF_LINE_EVEN
        off += OFF_HOP_EVEN if fi < len(feats) - 1 else 0.0
    return on, off


def contour_time(layer, band):
    """Per-part contour laser-on + inter-contour travel (Table 3)."""
    def interp(pair, g):
        return pair[0] + (pair[1] - pair[0]) * g
    if band == 'legs':
        c = CON['layers_1_250']
        on = (4 * c['L1_L4_L7_L10_ms'] + 4 * c['L2_L5_L8_L11_ms'] +
              4 * c['L3_L6_L9_L12_ms'] + c['base_ms']) / 1e3
        travel = 12 * INTER_CONTOUR
    elif band == 'overhang':
        g = overhang_growth(layer)
        c = CON['layers_251_350']
        on = (4 * interp(c['L1_L4_L7_L10_ms'], g) +
              4 * interp(c['L2_L5_L8_L11_ms'], g) +
              4 * interp(c['L3_L6_L9_L12_ms'], g) +
              interp(c['base_ms'], g)) / 1e3
        travel = 12 * INTER_CONTOUR
    elif band == 'bridge':
        on = CON['layers_351_600']['bridge_ms'] / 1e3
        travel = 0.0
    else:                                  # ridges, I2: perimeter/contour speed
        speed_c = G['contour_speed_mm_s']
        on = sum(2 * ((min(x0 + 1.0, RIDGE_XMAX) - x0) + Y_DEPTH) / speed_c
                 for x0 in RIDGE_X0)
        travel = 10 * INTER_CONTOUR
    return on + travel


def band_of(layer):
    for name, b in ST['layer_bands'].items():
        if name.startswith('_'):
            continue
        lo, hi = b['layers']
        if lo <= layer <= hi:
            return name
    raise ValueError(layer)


def main():
    layers = []
    for layer in range(1, TOTAL_LAYERS + 1):
        band = band_of(layer)
        parity = 'odd' if layer % 2 == 1 else 'even'
        on, off = (odd_layer_active if parity == 'odd'
                   else even_layer_active)(layer, band)
        part_active = on + off + contour_time(layer, band)
        plate_active = N_PARTS * part_active + (N_PARTS - 1) * INTER_PART
        layers.append({'layer': layer, 'band': band, 'parity': parity,
                       'infill_on_s': round(on, 4), 'infill_off_s': round(off, 4),
                       'part_active_s': round(part_active, 4),
                       'plate_active_s': round(plate_active, 4)})

    legs_active = [r['plate_active_s'] for r in layers if r['band'] == 'legs']
    legs_mean = sum(legs_active) / len(legs_active)
    dead_time = LAYER_TIME_LEGS - legs_mean                       # I1
    for r in layers:
        r['layer_time_s'] = round(r['plate_active_s'] + dead_time, 3)
        if r['band'] == 'ridges':
            r['source_class'] = 'inferred'

    closed_loop = {
        'computed_legs_plate_scan_s': {
            'mean': round(legs_mean, 3),
            'odd': round(sum(r['plate_active_s'] for r in layers
                             if r['band'] == 'legs' and r['parity'] == 'odd')
                         / len([r for r in layers if r['band'] == 'legs'
                                and r['parity'] == 'odd']), 3),
            'even': round(sum(r['plate_active_s'] for r in layers
                              if r['band'] == 'legs' and r['parity'] == 'even')
                          / len([r for r in layers if r['band'] == 'legs'
                                 and r['parity'] == 'even']), 3)},
        'published_plate_scan_s': PUBLISHED_SCAN,
        'residual_vs_published': round(legs_mean - PUBLISHED_SCAN, 3),
        'residual_relative': round((legs_mean - PUBLISHED_SCAN)
                                   / PUBLISHED_SCAN, 4),
        'dead_time_s_I1': round(dead_time, 3),
        'note': 'residual reported, not scaled away (constitution). '
                'layer_time = plate_active + dead_time reproduces 52 s '
                'on the legs band by construction of I1.'}

    # D-02 aggregation, N = 10 physical layers per computational layer
    N = 10
    agg = []
    for k in range(0, TOTAL_LAYERS, N):
        chunk = layers[k:k + N]
        agg.append({'computational_layer': k // N + 1,
                    'physical_layers': [chunk[0]['layer'], chunk[-1]['layer']],
                    'bands': sorted({r['band'] for r in chunk}),
                    'time_s': round(sum(r['layer_time_s'] for r in chunk), 3)})

    out = {
        'schema_version': 'ambench.layer-schedule/1',
        'decision': 'D-10 (2026-07-30, sample scope): case-local tooling',
        'inputs': {'scan_timing': 'inputs/scan-timing.json',
                   'geometry': 'plan geometry constants verified against '
                               'AMB2018_01_Part.STL (see scan-timing.json)'},
        'inferred_elements': {
            'I1': f'dead time {dead_time:.3f} s = 52 - computed legs active, '
                  'assumed layer-invariant',
            'I2': 'ridge band 601-624 uses the same infill/contour rules; '
                  'timing never published',
            'I3': 'inter-part 0.335 s and inter-contour 20.5 ms are midpoints '
                  'of published ranges'},
        'conflict_B1_note': '624 layers per JRES Table 2; Phan says 625 and '
                            'the STL height implies 625. If layer 625 exists '
                            'it is one more ridge layer (~same clock as 624).',
        'closed_loop': closed_loop,
        'total_build_time_s': round(sum(r['layer_time_s'] for r in layers), 1),
        'computational_layers_N10': agg,
        'layers': layers}

    os.makedirs(os.path.join(CASE, 'derived'), exist_ok=True)
    dst = os.path.join(CASE, 'derived', 'layer-schedule.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')

    print(f'closed loop: computed legs plate scan '
          f'{closed_loop["computed_legs_plate_scan_s"]["mean"]} s '
          f'(odd {closed_loop["computed_legs_plate_scan_s"]["odd"]} / '
          f'even {closed_loop["computed_legs_plate_scan_s"]["even"]}) '
          f'vs published ~{PUBLISHED_SCAN} s '
          f'-> residual {closed_loop["residual_relative"]*100:+.1f} %')
    print(f'dead time (I1): {dead_time:.2f} s')
    for b in ('legs', 'overhang', 'bridge', 'ridges'):
        ts = [r['layer_time_s'] for r in layers if r['band'] == b]
        print(f'  {b:9s}: {len(ts):3d} layers, layer time '
              f'{min(ts):.1f} - {max(ts):.1f} s')
    print(f'total build: {out["total_build_time_s"]/3600:.2f} h; '
          f'{len(agg)} computational layers (N=10)')
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
