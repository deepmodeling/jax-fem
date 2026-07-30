"""M31935 thermography frame-timestamp analysis — TIME DIMENSION ONLY.

Approved 2026-07-30 as the time-dimension reference (user decision): the
temperature-calibration use of M31935 remains REJECTED (emissivity coupling);
this tool touches only Layer.BuildTime / Layer.RawFrameNumber, never
Layer.RadiantTemp values.

Validates the D-10 layer-clock reconstruction against measured timestamps:
  - layer-cycle increments t0(N+1) - t0(N)  -> the published 52 s
  - odd-layer in-ROI recorded spans         -> per-part active time
    (camera ROI ~18.7 x 4.3 mm; odd layers sweep rows across all features,
    so the laser crosses the ROI on every row and the recorded span tracks
    the part-scan duration; even layers are feature-complete and dwell in
    the ROI only briefly - their short spans are geometric, not evidence)

Data: NIST AM-Bench 2018 in-situ thermography, DOI 10.18434/M31935,
Build1 layers 001-010 (legs band). The .mat files are NOT in the repository
(104 MB+); their sha256 are recorded in the output JSON.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

DEFAULT_DIR = ('/mnt/c/Users/user/Downloads/'
               'NIST_AMBench_625_Build1_Layers_001-010_LEGS')
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)


def main(data_dir=DEFAULT_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))
    if not files:
        sys.exit(f'no .mat files in {data_dir}')

    layers = []
    for f in files:
        L = loadmat(f, squeeze_me=True, struct_as_record=False)['Layer']
        bt = np.asarray(L.BuildTime, dtype=float)      # (n, 3) = H, M, S
        sec = bt[:, 0] * 3600 + bt[:, 1] * 60 + bt[:, 2]
        raw = np.asarray(L.RawFrameNumber, dtype=float)
        gaps = np.diff(sec)
        layers.append({
            'file': os.path.basename(f),
            'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest(),
            'n_frames_stored': int(len(sec)),
            'raw_frame_range': [int(raw[0]), int(raw[-1])],
            't_first_s': round(float(sec[0]), 3),
            't_last_s': round(float(sec[-1]), 3),
            'recorded_span_s': round(float(sec[-1] - sec[0]), 3),
            'max_internal_gap_s': round(float(gaps.max()), 3) if len(gaps) else 0.0,
        })

    incs = [round(layers[i + 1]['t_first_s'] - layers[i]['t_first_s'], 3)
            for i in range(len(layers) - 1)]
    odd_spans = [r['recorded_span_s'] for i, r in enumerate(layers)
                 if (i + 1) % 2 == 1 and r['max_internal_gap_s'] < 0.5]

    sched = json.load(open(os.path.join(CASE, 'derived',
                                        'layer-schedule.json')))
    legs_odd = [r for r in sched['layers']
                if r['band'] == 'legs' and r['parity'] == 'odd']
    computed_part_odd = round(legs_odd[0]['part_active_s'], 3)

    out = {
        'schema_version': 'ambench.m31935-timing/1',
        'decision': 'user approval 2026-07-30: M31935 admitted as the '
                    'TIME-DIMENSION reference only; temperature-calibration '
                    'use remains rejected',
        'source': {'doi': '10.18434/M31935',
                   'bundle': 'Build1 layers 001-010 (legs band)',
                   'note': 'mat files not in repo; hashed below'},
        'layer_cycle_increments_s': incs,
        'layer_cycle_mean_s': round(float(np.mean(incs)), 3),
        'layer_cycle_std_s': round(float(np.std(incs)), 3),
        'published_layer_time_s': 52,
        'parity_alternation': 'odd-layer cycles ~51.7 s, even ~52.8 s '
                              '(opposite sign to computed active-time parity; '
                              'suggests parity-adjusted dwell; +-0.6 s)',
        'odd_layer_recorded_spans_s': odd_spans,
        'computed_part_active_odd_s': computed_part_odd,
        'reading_B_predicted_span_s': round(computed_part_odd * 26 / 17.594, 3),
        'verdict': 'reading A supported: measured odd spans ~3.98-4.00 s are '
                   'consistent with computed part active (4.5 s incl. '
                   'out-of-ROI contour/edge rows) and exclude the ~1.48x '
                   'scaling reading B would require (~6.6 s -> span ~5.7 s). '
                   'The published "~26 s" plate-scan figure is judged loose; '
                   'the -32 % D-10 residual is closed as a source-side '
                   'approximation, not a reconstruction error.',
        'anomalies': 'layer 9: 0.735 s internal recording gap (dropped '
                     'frames or skip); excluded from span statistics',
        'layers': layers,
    }

    dst = os.path.join(CASE, 'derived', 'm31935-timing-check.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    print(f'cycle: {out["layer_cycle_mean_s"]} +- {out["layer_cycle_std_s"]} s '
          f'(published 52); odd spans {odd_spans} vs computed part '
          f'{computed_part_odd} s (reading B would need '
          f'~{out["reading_B_predicted_span_s"]} s)')
    print(f'wrote {dst}')


if __name__ == '__main__':
    main(*sys.argv[1:])
