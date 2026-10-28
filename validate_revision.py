"""Targeted review checks; preserve reference_results and do not run the full script.

Load the definitions verbatim from run_analysis.py using AST selection, excluding
all top-level simulation, optimizer and output side effects. The input hash is
recorded. This checks the existing engine; it is not an independent engine.
The DOP853 mode uses analytic envelopes, not splitting segments or drive matrices.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, replace
import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent
FUNCTIONS = {'shape', '_const', 'segments', 'pulse_time', '_step_unitaries',
             '_drive_unitaries', 'evolve', 'probs_at', 'noise_grid', 'probabilities',
             'response', 'calibrate', 'max_bias', 'sensitivity', 'analytic_plan'}
ASSIGNMENTS = {'WAIT', 'FIELDS', 'METHODS', 'TAU', 'GAMMA', 'A_HF', 'ZD', 'PSI0',
               'PHI_BB1', 'BB1', 'HONEST', '_CAL', 'OC_AMPS_MHZ', 'OC_PHASES',
               'OC_T', 'CFG'}


def load_engine():
    source = (ROOT / 'run_analysis.py').read_text(encoding='utf-8')
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in {'Cfg', 'Env', 'Ctl'}:
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in ASSIGNMENTS for t in node.targets):
            selected.append(node)
    # Dataclasses require a registered module name; use this module's globals.
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'run_analysis.py', 'exec'), globals())
    return hashlib.sha256(source.encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def emit(msg):
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), msg, flush=True)


def make_inverse(grid, signals, kind):
    delta = np.diff(signals)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError('Nonmonotonic nominal calibration')
    order = np.argsort(signals)
    if kind == 'inverse_pchip':
        return PchipInterpolator(signals[order], grid[order], extrapolate=False)
    forward = PchipInterpolator(grid, signals, extrapolate=False)
    lo, hi = signals.min(), signals.max()
    def inverse(values):
        return np.array([brentq(lambda b: float(forward(b)) - float(v), grid[0],
                         grid[-1], xtol=1e-13) if lo <= v <= hi else np.nan
                         for v in np.atleast_1d(values)])
    return inverse


def summarize(method, scale, grid_name, ncal, inverse_name, dt, grid, signals, inv, scenario):
    estimates = np.asarray(inv(signals))
    valid = np.isfinite(estimates)
    errors = 1000 * (estimates - grid)
    abs_valid = np.abs(errors[valid])
    peak = np.flatnonzero(valid)[np.argmax(abs_valid)] if valid.any() else None
    row = dict(scenario=scenario, method=method, scale=scale, test_grid=grid_name,
               test_points=len(grid), calibration_points=ncal, inverse=inverse_name,
               dt_us=dt, invalid_count=int((~valid).sum()),
               invalid_fraction=float((~valid).mean()),
               max_bias_nT=float(abs_valid.max()) if valid.all() else np.nan,
               valid_max_bias_nT=float(abs_valid.max()) if valid.any() else np.nan,
               valid_median_abs_bias_nT=float(np.median(abs_valid)) if valid.any() else np.nan,
               valid_p95_abs_bias_nT=float(np.quantile(abs_valid, .95)) if valid.any() else np.nan,
               peak_valid_field_uT=float(grid[peak]) if peak is not None else np.nan)
    points = [dict(**{k: row[k] for k in ('scenario', 'method', 'scale', 'test_grid',
                                        'calibration_points', 'inverse', 'dt_us')},
                   field_uT=float(b), signal=float(v), estimate_uT=float(e),
                   valid=bool(ok), error_nT=float(err))
              for b, v, e, ok, err in zip(grid, signals, estimates, valid, errors)]
    return row, points


def grid_checks(out):
    # Union permits exact comparisons of nested 25/101/401 grids; shifted 400
    # points are cell midpoints independent of the original test locations.
    grids = {'original25': np.linspace(-.2, .2, 25),
             'dense101': np.linspace(-.2, .2, 101),
             'dense401': np.linspace(-.2, .2, 401),
             'shifted400': -.2 + (np.arange(400)+.5)*(.4/400)}
    fields = np.unique(np.concatenate(list(grids.values())))
    indices = {k: np.searchsorted(fields, v) for k, v in grids.items()}
    calibrations = {n: np.linspace(-.3, .3, n) for n in (41, 81, 161)}
    cal_union = np.unique(np.concatenate(list(calibrations.values())))
    rows, points = [], []
    cases = [('hard', (.9, 1.1, 1.5)), ('bb1', (.9, 1.1, 1.5)),
             ('stirap', (.9, 1.1, 1.5)), ('oc', (.9, 1.1, 1.5))]
    for method, scales in cases:
        emit(f'Grid check {method}: nominal calibration')
        nominal = response(method, cal_union, cfg=CFG)
        for scale in scales:
            emit(f'Grid check {method} s={scale}: {len(fields)} independent/union fields')
            disturbed = response(method, fields, ctl=Ctl(scale, scale), cfg=CFG)
            for ncal, cg in calibrations.items():
                cs = nominal[np.searchsorted(cal_union, cg)]
                for inverse_name in ('inverse_pchip', 'forward_pchip_root'):
                    inv = make_inverse(cg, cs, inverse_name)
                    for name, grid in grids.items():
                        row, ps = summarize(method, scale, name, ncal, inverse_name,
                                            CFG.dt, grid, disturbed[indices[name]], inv, 'selective')
                        rows.append(row); points.extend(ps)
            write_csv(out/'grid_summary.csv', rows)
            write_csv(out/'grid_per_field.csv', points)
    # Dedicated propagation check uses the same refined 161-point calibration
    # and dense 401-point test grid at each step, not just the original grid.
    for dt in (.00025, .000125):
        cfg = replace(CFG, dt=dt)
        grid, cg = grids['dense401'], calibrations[161]
        emit(f'OC +10% time-step check dt={dt} us')
        cs = response('oc', cg, cfg=cfg)
        attacked = response('oc', grid, ctl=Ctl(1.1, 1.1), cfg=cfg)
        inv = make_inverse(cg, cs, 'inverse_pchip')
        row, ps = summarize('oc', 1.1, 'dense401', 161, 'inverse_pchip', dt,
                            grid, attacked, inv, 'selective')
        rows.append(row); points.extend(ps)
        write_csv(out/'grid_summary.csv', rows); write_csv(out/'grid_per_field.csv', points)


def hardware_checks(out):
    nuclear = tuple((A_HF*m/GAMMA, 1/3) for m in (-1, 0, 1))
    cfg = replace(CFG, cross_df=280.)
    env = Env(nuclear=nuclear)
    grid = np.linspace(-.2, .2, 25)
    cg = np.linspace(-.3, .3, 41)
    rows, points = [], []
    for method, scales in [('bb1', (.9, 1.5)), ('stirap', (.9,)), ('oc', (.9,))]:
        emit(f'Per-field 5 mT mixture: {method} calibration')
        inv = make_inverse(cg, response(method, cg, cfg=cfg, env=env), 'inverse_pchip')
        for scale in scales:
            signal = response(method, grid, ctl=Ctl(scale, scale), cfg=cfg, env=env)
            row, ps = summarize(method, scale, 'original25', 41, 'inverse_pchip', cfg.dt,
                                grid, signal, inv, '5mT_equal_mixture')
            rows.append(row); points.extend(ps)
            write_csv(out/'validity_summary.csv', rows); write_csv(out/'validity_per_field.csv', points)
            emit(f'{method} s={scale}: {row["invalid_count"]}/25 invalid')


def dop853_probabilities(fields, ctl, cfg, env, max_step, rtol=1e-10, atol=1e-12):
    """Independent analytic-envelope evolution, batching hyperfine/noise states.

    No call to segments(), evolve(), or _drive_unitaries(). t_eval retains
    only segment endpoints. Field and quasistatic averaging match evaluation.
    """
    fields = np.atleast_1d(fields)
    offsets, weights = noise_grid(env)
    xs = (fields[:, None]+offsets).ravel()
    gx = GAMMA*xs
    stats = []
    def propagate(plan, state, absolute_start):
        for duration, envelope in plan:
            origin = absolute_start
            def rhs(t, y):
                A, B = envelope(t)
                phase = TAU*cfg.cross_df*(origin+t)
                aa = (A+B*np.exp(1j*phase))/2
                bb = (B+A*np.exp(-1j*phase))/2
                v = y.reshape(-1, 3)
                dy = np.empty_like(v)
                dy[:, 0] = -1j*(gx*v[:, 0] + np.conj(bb)*v[:, 1])
                dy[:, 1] = -1j*(bb*v[:, 0] + aa*v[:, 2])
                dy[:, 2] = -1j*(-gx*v[:, 2] + np.conj(aa)*v[:, 1])
                return dy.ravel()
            started = time.time()
            sol = solve_ivp(rhs, (0., duration), state.ravel(), method='DOP853',
                            rtol=rtol, atol=atol, max_step=max_step, t_eval=[duration])
            if not sol.success:
                raise RuntimeError(sol.message)
            state = sol.y[:, -1].reshape(-1, 3)
            stats.append(dict(duration_us=duration, nfev=sol.nfev,
                              seconds=time.time()-started,
                              max_norm_error=float(np.max(np.abs(np.sum(abs(state)**2, axis=1)-1)))))
            absolute_start += duration
            emit(f'DOP853 segment {duration:g} us complete; nfev={sol.nfev}')
        return state, absolute_start
    state = np.broadcast_to(PSI0, (len(xs), 3)).copy()
    state, t = propagate(analytic_plan('stirap', ctl, cfg), state, 0.)
    state *= np.exp(-1j*GAMMA*xs[:, None]*ZD*WAIT)
    raw = []
    for phase in (np.pi/2, -np.pi/2):
        final, _ = propagate(analytic_plan('stirap', ctl, cfg, phase, True), state.copy(), t+WAIT)
        raw.append(abs(final[:, 1])**2)
    component = np.stack(raw, axis=1).reshape(len(fields), len(offsets), 2)
    return np.einsum('fnq,n->fq', component, weights), component, stats


def reference_check(out):
    cfg = replace(CFG, cross_df=2856.)
    off = -A_HF/GAMMA
    env = Env(nuclear=((0., .95), (off, .025), (2*off, .025)))
    grid = np.linspace(-.2, .2, 25); cg = np.linspace(-.3, .3, 41)
    emit('51 mT mixture: split calibration and peak identification')
    nominal = probabilities('stirap', cg, cfg=cfg, env=env)
    inv = make_inverse(cg, nominal@np.array([1., -1.]), 'inverse_pchip')
    attacked = probabilities('stirap', grid, ctl=Ctl(1.5, 1.5), cfg=cfg, env=env)
    row, points = summarize('stirap', 1.5, 'original25', 41, 'inverse_pchip', cfg.dt,
                            grid, attacked@np.array([1., -1.]), inv, '51mT_95pct')
    write_csv(out/'reference51_split_summary.csv', [row])
    write_csv(out/'reference51_split_per_field.csv', points)
    peak = row['peak_valid_field_uT']
    idx = int(np.flatnonzero(grid == peak)[0])
    estimate = float(inv(attacked[idx]@np.array([1., -1.])))
    k = int(np.searchsorted(cg, estimate))-1
    start = max(0, min(k-1, len(cg)-4))
    local_fields = cg[start:start+4]
    maxstep = 1/(12*cfg.cross_df)
    # Honest local nodes provide both slopes for the interior inverse interval.
    # This is local verification, not a full independent 41-node calibration.
    emit(f'51 mT independent nominal nodes {local_fields.tolist()}')
    direct_nominal, raw_nominal, stats1 = dop853_probabilities(local_fields, HONEST, cfg, env, maxstep)
    np.savez_compressed(out/'reference51_nominal.npz', fields=local_fields,
                        probabilities=direct_nominal, component_probabilities=raw_nominal)
    (out/'reference51_nominal_solver.json').write_text(json.dumps(stats1, indent=2))
    emit(f'51 mT independent disturbed population average at b={peak:g} uT')
    direct, raw_attacked, stats2 = dop853_probabilities([peak], Ctl(1.5, 1.5), cfg, env, maxstep)
    np.savez_compressed(out/'reference51_disturbed.npz', fields=[peak],
                        probabilities=direct, component_probabilities=raw_attacked)
    (out/'reference51_disturbed_solver.json').write_text(json.dumps(stats2, indent=2))
    local_inv = make_inverse(local_fields, direct_nominal@np.array([1., -1.]), 'inverse_pchip')
    direct_est = float(local_inv(direct[0]@np.array([1., -1.])))
    split_local = nominal[start:start+4]
    result = dict(field_uT=peak, split_max_bias_nT=row['max_bias_nT'],
                  independent_local_bias_nT=1000*abs(direct_est-peak),
                  inferred_field_difference_nT=1000*abs(direct_est-estimate),
                  max_disturbed_probability_difference=float(np.max(abs(direct[0]-attacked[idx]))),
                  max_nominal_probability_difference=float(np.max(abs(direct_nominal-split_local))),
                  max_step_us=maxstep, rtol=1e-10, atol=1e-12,
                  noise_nodes_per_manifold=21, nuclear_populations=[.95,.025,.025],
                  calibration_scope='four independent nominal nodes around the inferred field',
                  scope='one original-grid peak at s=1.5, not a continuous-field or full-grid certificate')
    result['passes_selected_checks'] = bool(result['inferred_field_difference_nT'] < .1 and
        result['max_disturbed_probability_difference'] < 1e-5 and
        result['max_nominal_probability_difference'] < 1e-5)
    (out/'reference51_check.json').write_text(json.dumps(result, indent=2))
    emit(str(result))


def energy_checks(out):
    rows = []
    for method in METHODS:
        segs = segments(method, HONEST, CFG)
        energy = 2*sum(float(np.sum((abs(s['A'])**2+abs(s['B'])**2)*s['dt'])) for s in segs)
        rows.append(dict(method=method, control_time_us=pulse_time(method, CFG),
                         integrated_drive_rad2_per_us=energy,
                         equivalent_peak_time_us=energy/CFG.peak**2,
                         nuclear_preparation_cost='not modeled'))
    write_csv(out/'drive_resources.csv', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['grid', 'hardware', 'reference51', 'resources'])
    parser.add_argument('--output', default='revision_results')
    args = parser.parse_args()
    sha = load_engine()
    out = ROOT/args.output; out.mkdir(parents=True, exist_ok=True)
    settings = dict(mode=args.mode, started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), source_sha256=sha,
                    reference_commit='d093f5151b385b3c8d22f2175c88fd668016e3e0',
                    source_file='run_analysis.py', selected_definitions=sorted(FUNCTIONS))
    (out/f'{args.mode}_settings.json').write_text(json.dumps(settings, indent=2))
    start = time.time()
    try:
        {'grid': grid_checks, 'hardware': hardware_checks,
         'reference51': reference_check, 'resources': energy_checks}[args.mode](out)
    except Exception as exc:
        (out/f'{args.mode}_status.json').write_text(json.dumps(dict(completed=False,
            error=repr(exc), elapsed_seconds=time.time()-start), indent=2))
        raise
    (out/f'{args.mode}_status.json').write_text(json.dumps(dict(completed=True,
        elapsed_seconds=time.time()-start), indent=2))
    emit(f'{args.mode} completed in {time.time()-start:.1f} s')


if __name__ == '__main__':
    main()
