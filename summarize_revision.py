"""Aggregate completed checks without rerunning the engine or hiding invalids."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'revision_results/verified'


def read(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


def main():
    for mode in ('grid', 'hardware', 'reference51', 'resources'):
        assert json.loads((OUT/f'{mode}_status.json').read_text())['completed']
        settings = json.loads((OUT/f'{mode}_settings.json').read_text())
        assert settings['numpy'] == '2.2.3' and settings['scipy'] == '1.15.2'
    rows = read('grid_summary.csv')
    points = read('grid_per_field.csv')
    # Add the dense-grid endpoints to the independent shifted interior grid.
    # All signals were already propagated at these fields; do not interpolate
    # or extrapolate missing field samples to produce these comparisons.
    shifted = []
    for row in rows:
        if row['test_grid'] != 'shifted400':
            continue
        key = ('method', 'scale', 'calibration_points', 'inverse', 'dt_us')
        candidates = [p for p in points if all(p[k] == row[k] for k in key)]
        selected = [p for p in candidates if p['test_grid'] == 'shifted400' or
                    (p['test_grid'] == 'dense401' and abs(float(p['field_uT'])) == .2)]
        assert len(selected) == 402
        valid = [p for p in selected if p['valid'] == 'True']
        errors = np.array([abs(float(p['error_nT'])) for p in valid])
        r = dict(row)
        r.update(test_grid='shifted402_with_endpoints', test_points=402,
                 invalid_count=402-len(valid), invalid_fraction=(402-len(valid))/402,
                 max_bias_nT=errors.max() if len(valid) == 402 else np.nan,
                 valid_max_bias_nT=errors.max(), valid_median_abs_bias_nT=np.median(errors),
                 valid_p95_abs_bias_nT=np.quantile(errors, .95),
                 peak_valid_field_uT=valid[int(np.argmax(errors))]['field_uT'])
        shifted.append(r)
    write('shifted_endpoint_summary.csv', shifted)
    all_rows = rows+shifted
    def pick(method, scale, grid, ncal, inverse='inverse_pchip', dt='.0005'):
        return next(r for r in all_rows if r['method'] == method and float(r['scale']) == scale
                    and r['test_grid'] == grid and int(r['calibration_points']) == ncal
                    and r['inverse'] == inverse and float(r['dt_us']) == float(dt))
    checks = []
    for method in ('hard', 'bb1', 'stirap', 'oc'):
        for scale in (.9, 1.1, 1.5):
            base = pick(method, scale, 'original25', 41)
            dense = pick(method, scale, 'dense401', 161)
            test_delta = max(abs(float(pick(method, scale, grid, 41)['max_bias_nT']) -
                                 float(base['max_bias_nT']))
                             for grid in ('dense101', 'dense401', 'shifted402_with_endpoints'))
            cal_delta = abs(float(pick(method, scale, 'dense401', 41)['max_bias_nT']) -
                            float(dense['max_bias_nT']))
            inverse_delta = max(abs(float(pick(method, scale, grid, n, 'forward_pchip_root')['max_bias_nT']) -
                                    float(pick(method, scale, grid, n)['max_bias_nT']))
                                for grid in ('original25', 'dense101', 'dense401', 'shifted402_with_endpoints')
                                for n in (41, 81, 161))
            dense_shift_delta = abs(float(pick(method, scale, 'shifted402_with_endpoints', 161)['max_bias_nT']) -
                                    float(dense['max_bias_nT']))
            checks.append(dict(method=method, scale=scale, original25_cal41_nT=float(base['max_bias_nT']),
                               dense401_cal161_nT=float(dense['max_bias_nT']),
                               max_test_grid_change_cal41_nT=test_delta,
                               calibration41_to161_dense401_change_nT=cal_delta,
                               max_alternative_inverse_change_nT=inverse_delta,
                               dense_vs_shifted_cal161_change_nT=dense_shift_delta))
    write('key_comparisons.csv', checks)
    with (ROOT/'reference_results/result1_common_amplitude_sweep.csv').open() as f:
        archived = list(csv.DictReader(f))
    reproduction = []
    for r in checks:
        ref = next(a for a in archived if a['method'] == r['method'] and
                   abs(float(a['amplitude_scale'])-r['scale']) < 1e-12)
        delta = abs(r['original25_cal41_nT']-float(ref['max_abs_bias_nT']))
        reproduction.append(dict(method=r['method'], scale=r['scale'], difference_from_reference_nT=delta))
        assert delta < 1e-5, (r, ref)
    write('baseline_reproduction.csv', reproduction)
    oc = [pick('oc', 1.1, 'dense401', 161, dt=dt) for dt in ('.0005', '.00025', '.000125')]
    summary = dict(selective_cases=len(checks),
                   max_test_grid_change_cal41_nT=max(r['max_test_grid_change_cal41_nT'] for r in checks),
                   max_calibration41_to161_dense401_change_nT=max(r['calibration41_to161_dense401_change_nT'] for r in checks),
                   max_alternative_inverse_change_nT=max(r['max_alternative_inverse_change_nT'] for r in checks),
                   max_dense_vs_shifted_cal161_change_nT=max(r['dense_vs_shifted_cal161_change_nT'] for r in checks),
                   oc_plus10_dense_cal161_by_dt=[dict(dt_us=float(r['dt_us']), bias_nT=float(r['max_bias_nT'])) for r in oc],
                   invalids_in_tested_selective_rows=sum(int(r['invalid_count']) for r in rows),
                   note='Observed changes for selected cases, not certified error bounds.')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    # Selected short table for the main six-page manuscript.
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Method & $s$ & Original & Refined & $\Delta_{\rm inv}$ \\', r'\midrule']
    for method, scale, label in [('bb1', .9, 'BB1'), ('oc', 1.1, 'OC'),
                                  ('bb1', 1.5, 'BB1'), ('stirap', 1.5, r'\ST'), ('oc', 1.5, 'OC')]:
        r = next(r for r in checks if r['method'] == method and r['scale'] == scale)
        lines.append(f"{label} & {scale:.1f} & {r['original25_cal41_nT']:.5f} & "
                     f"{r['dense401_cal161_nT']:.5f} & {r['max_alternative_inverse_change_nT']:.5f}"+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'manuscript/tables/revision_checks.tex').write_text('\n'.join(lines)+'\n')
    # Manifest covers numerical inputs and all final validation outputs.
    manifest_files = list(OUT.glob('*')) + [ROOT/'run_analysis.py', ROOT/'validate_revision.py', ROOT/'summarize_revision.py']
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in manifest_files if p.is_file() and p.name != 'manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
