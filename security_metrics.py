"""Frozen-calibration threshold metrics for selected common-gain cases.
Run: python security_metrics.py
Outputs are deterministic sampled-field fractions, not attack probabilities.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nv_security_engine import CFG, Env, Ctl, calibrate, response

METHODS = ['hard', 'bb1', 'stirap', 'oc']
LABELS = ['Hard', 'BB1', 'Half-STIRAP', 'Stored OC']

def summarize(errors, thresholds):
    valid = np.isfinite(errors)
    n = len(errors)
    return [dict(threshold_nT=float(t), n_fields=n, n_valid=int(valid.sum()),
                 n_invalid=int((~valid).sum()), invalid_fraction=float((~valid).mean()),
                 n_exceed_valid=int((valid & (np.abs(errors) > t)).sum()),
                 fraction_valid_above_threshold=float((valid & (np.abs(errors) > t)).mean()))
            for t in thresholds]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scales', nargs='+', type=float, default=[0.6,0.9,1.1,1.5])
    ap.add_argument('--thresholds', nargs='+', type=float, default=[1,5,10,50])
    ap.add_argument('--points', type=int, default=25)
    ap.add_argument('--output', type=Path, default=Path('security_results'))
    args = ap.parse_args()
    if args.points < 2 or any(s <= 0 for s in args.scales) or any(t < 0 for t in args.thresholds):
        ap.error('Use at least two fields, positive scales, and nonnegative thresholds.')
    out = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    fields = np.linspace(-.2,.2,args.points)
    env = Env()  # Selective drive, one nuclear state; T2*=20 us; 21 noise nodes.
    rows, summaries, checks = [], [], []
    for method in METHODS:
        print(f'Calibrating {method}...', flush=True)
        inv = calibrate(method, cfg=CFG, env=env)
        if inv is None:
            raise RuntimeError(f'{method}: nominal calibration is not monotonic.')
        nominal_est = inv(response(method,fields,6.,Ctl(),CFG,env))
        nominal_error = 1000*(nominal_est-fields)
        checks.append(dict(method=method, nominal_max_bias_nT=float(np.max(np.abs(nominal_error)))))
        if not np.isfinite(nominal_error).all() or np.max(np.abs(nominal_error)) > .1:
            raise RuntimeError(f'{method}: nominal calibration check exceeded 0.1 nT.')
        for scale in args.scales:
            print(f'  {method}: scale={scale:g}', flush=True)
            signal = response(method, fields, 6., Ctl(scale,scale), CFG, env)
            estimate = inv(signal)
            errors = 1000*(estimate-fields)
            for b,s,e,err in zip(fields,signal,estimate,errors):
                rows.append(dict(method=method, amplitude_scale=scale, true_field_uT=b,
                                 signal=s, estimated_field_uT=e, error_nT=err,
                                 valid=bool(np.isfinite(err))))
            for row in summarize(errors,args.thresholds):
                summaries.append(dict(method=method,amplitude_scale=scale,**row))
            # Save incremental outputs so partial runs are recoverable.
            pd.DataFrame(rows).to_csv(out/'per_field_errors.csv',index=False)
            pd.DataFrame(summaries).to_csv(out/'threshold_metrics.csv',index=False)
    summary = pd.DataFrame(summaries)
    pd.DataFrame(checks).to_csv(out/'nominal_checks.csv',index=False)
    maxima = pd.DataFrame(rows).groupby(['method','amplitude_scale']).agg(
        max_abs_valid_bias_nT=('error_nT',lambda x: np.abs(x).max()),
        n_valid=('valid','sum'),n_fields=('valid','size')).reset_index()
    maxima['max_bias_nT'] = maxima.max_abs_valid_bias_nT.where(maxima.n_valid == maxima.n_fields)
    maxima.to_csv(out/'maximum_bias_checks.csv',index=False)
    fig, axes = plt.subplots(1,len(args.thresholds),figsize=(3.2*len(args.thresholds),2.8),squeeze=False,layout='constrained')
    for ax,t in zip(axes[0],args.thresholds):
        grid=summary[summary.threshold_nT==t].pivot(index='method',columns='amplitude_scale',values='fraction_valid_above_threshold').reindex(index=METHODS,columns=args.scales).to_numpy()
        im=ax.imshow(grid,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=range(len(args.scales)),xticklabels=[f'{100*(s-1):+.0f}%' for s in args.scales],yticks=range(4),yticklabels=LABELS,title=f'Error > {t:g} nT',xlabel='Common amplitude change')
        for i in range(4):
            for j in range(len(args.scales)):
                ax.text(j,i,f'{grid[i,j]:.0%}',ha='center',va='center',color='white' if grid[i,j]>.55 else 'black')
    fig.colorbar(im,ax=list(axes[0]),label='Fraction of all tested fields',shrink=.85)
    fig.savefig(out/'threshold_fractions.pdf',bbox_inches='tight')
    fig.savefig(out/'threshold_fractions.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    for t in args.thresholds:
        table=summary[summary.threshold_nT==t].pivot(index='method',columns='amplitude_scale',values='fraction_valid_above_threshold').reindex(index=METHODS,columns=args.scales)
        lines = [r'\begin{tabular}{l'+'r'*len(args.scales)+'}', r'\toprule',
                 'Method & '+' & '.join(f'{100*(s-1):+.0f}\\%' for s in args.scales)+r' \\', r'\midrule']
        for method,label in zip(METHODS,LABELS):
            lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in table.loc[method])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (out/f'threshold_{t:g}_nT.tex').write_text('\n'.join(lines))
    settings=dict(scenario='selective drive, single nuclear state', fields_uT=fields.tolist(),
                  scales=args.scales,thresholds_nT=args.thresholds,strict_inequality=True,
                  denominator='all tested fields including invalid readings',
                  invalid_handling='excluded from valid-output exceedances; reported separately; no alarm assumed',
                  wait_us=6.,T2_DQ_us=env.T2_DQ,noise_nodes=env.nodes,dt_us=CFG.dt,
                  source_commit='d093f5151b385b3c8d22f2175c88fd668016e3e0',
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('nv_security_engine.py')]})
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    print(f'Done. Outputs: {out.resolve()}')

if __name__ == '__main__':
    main()
