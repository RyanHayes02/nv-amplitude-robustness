"""Run the NV control benchmark and save results. See README.md for model and scope."""

# %% Analysis block 1
import time, os, zipfile, json, sys, platform, hashlib, traceback
from pathlib import Path
from datetime import datetime, timezone
try:
    from IPython.display import display, FileLink
except ImportError:
    display = print
    FileLink = str
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = str(Path('paper_results') / RUN_ID)
os.makedirs(OUT, exist_ok=True)
# Restore streams on a repeated Run All, then capture this run's printed output.
if hasattr(sys.stdout, '_nv_original'):
    sys.stdout = sys.stdout._nv_original
if hasattr(sys.stderr, '_nv_original'):
    sys.stderr = sys.stderr._nv_original
RUN_LOG = open(Path(OUT)/'run_log.txt', 'w', buffering=1, encoding='utf-8')
class Tee:
    def __init__(self, stream, log): self._nv_original, self.log = stream, log
    def write(self, text):
        self._nv_original.write(text); self.log.write(text)
        return len(text)
    def flush(self): self._nv_original.flush(); self.log.flush()
    def __getattr__(self, name): return getattr(self._nv_original, name)
sys.stdout = Tee(sys.stdout, RUN_LOG)
sys.stderr = Tee(sys.stderr, RUN_LOG)
print('Run:', RUN_ID, '| results:', Path(OUT).resolve())
WAIT = 6.0                                   # us, microwave-free wait
FIELDS = np.linspace(-0.2, 0.2, 25)          # uT, residual fields tested
METHODS = ['hard', 'bb1', 'stirap', 'oc']
LABEL = {'hard': 'Hard', 'bb1': 'BB1', 'stirap': 'Half-STIRAP', 'oc': 'Optimal control (time-matched)'}
COLOR = {'hard': 'tab:blue', 'bb1': 'tab:orange', 'stirap': 'tab:green', 'oc': 'tab:purple'}
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25})
T_START = time.time()
def elapsed(): return f'{(time.time()-T_START)/60:.1f} min'
print('ready')

plt.close('all')
_original_show = getattr(plt.show, '_nv_original', plt.show)
FIGURE_COUNTER = 0
def saved_show(*args, **kwargs):
    global FIGURE_COUNTER
    for number in plt.get_fignums():
        fig=plt.figure(number)
        if not getattr(fig, '_nv_saved', False):
            FIGURE_COUNTER+=1
            fig.savefig(f'{OUT}/figure_{FIGURE_COUNTER:02d}.png',dpi=200,bbox_inches='tight')
            fig.savefig(f'{OUT}/figure_{FIGURE_COUNTER:02d}.pdf',bbox_inches='tight')
            fig._nv_saved=True
    return _original_show(*args, **kwargs)
saved_show._nv_original=_original_show
plt.show=saved_show


# %% Analysis block 3
TAU = 2*np.pi
GAMMA = TAU*0.028            # rad/us per uT
A_HF = TAU*(-2.16)           # 14N hyperfine, rad/us
ZD = np.array([1.0, 0.0, -1.0])
PSI0 = np.array([0, 1, 0], complex)
PHI_BB1 = np.arccos(-0.25)
BB1 = ((np.pi, 0.0), (np.pi, PHI_BB1), (2*np.pi, 3*PHI_BB1), (np.pi, PHI_BB1))

@dataclass(frozen=True)
class Cfg:
    peak: float = TAU*10.0       # combined peak Rabi scale, rad/us
    half_time: float = 1.2       # us
    arp_time: float = 1.0        # us
    chirp_span: float = TAU*25.0 # rad/us
    dt: float = 0.0005           # us
    cross_df: float = 0.0        # MHz tone separation for cross-driving (0 = selective drive)
    cross_res: float = 0.05      # 2*pi*df*dt on the sub-step grid
    oc_pulse: tuple = None       # (amplitudes rad/us, phases, duration us)

@dataclass(frozen=True)
class Env:
    T2_DQ: float = 20.0
    nodes: int = 21
    nuclear: tuple = ((0.0, 1.0),)   # (field offset uT, weight) per nuclear manifold

@dataclass(frozen=True)
class Ctl:
    sA: float = 1.0
    sB: float = 1.0

HONEST = Ctl()

def shape(s):
    R = np.where(s < .2, np.sin(np.pi*s/.4)**2, np.where(s > .8, np.cos(np.pi*(s-.8)/.4)**2, 1.0))
    u = np.clip((s-.2)/.6, 0, 1)
    th = np.pi/8*(1-np.cos(np.pi*u))
    return R*np.cos(th), R*np.sin(th)

def _const(T, a, b, cfg):
    n = max(int(np.ceil(T/cfg.dt)), 1)
    return dict(dt=np.full(n, T/n), A=np.full(n, a, complex), B=np.full(n, b, complex))

def segments(method, ctl, cfg, phase_B=0.0, analysis=False):
    eB = np.exp(1j*phase_B)
    if method in ('hard', 'bb1', 'oc'):
        if method == 'oc':
            amps, phs, T = cfg.oc_pulse
            plan = [(a, p, T/len(amps)) for a, p in zip(amps, phs)]
        else:
            steps = ((np.pi, 0.0),) if method == 'hard' else BB1
            plan = [(cfg.peak, p, ang/cfg.peak) for ang, p in steps]
        return [_const(T, a/np.sqrt(2)*ctl.sA*np.exp(1j*p), a/np.sqrt(2)*ctl.sB*np.exp(1j*p)*eB, cfg)
                for a, p, T in plan]
    Ts, Th = cfg.arp_time, cfg.half_time
    def arp(rev):
        n = int(np.ceil(Ts/cfg.dt)); t = (np.arange(n)+.5)*Ts/n
        ph = (-1 if rev else 1)*cfg.chirp_span*(t*t/Ts - t)
        return dict(dt=np.full(n, Ts/n), A=np.zeros(n, complex),
                    B=cfg.peak*ctl.sB*np.sin(np.pi*t/Ts)*np.exp(1j*ph)*eB)
    def half(rev):
        n = int(np.ceil(Th/cfg.dt)); t = (np.arange(n)+.5)*Th/n
        a, b = shape((Th-t)/Th if rev else t/Th)
        return dict(dt=np.full(n, Th/n), A=cfg.peak*ctl.sA*a+0j, B=cfg.peak*ctl.sB*b*eB)
    return [half(True), arp(True)] if analysis else [arp(False), half(False)]

def pulse_time(method, cfg):
    if method == 'stirap': return 2*(cfg.arp_time+cfg.half_time)
    if method == 'oc': return 2*cfg.oc_pulse[2]
    return 2*(np.pi if method == 'hard' else sum(a for a, _ in BB1))/cfg.peak

def _step_unitaries(dt, A, B, cfg, t0):
    H = np.zeros((len(dt), 3, 3), complex)
    H[:, 1, 0] = B/2; H[:, 0, 1] = np.conj(B)/2
    H[:, 1, 2] = A/2; H[:, 2, 1] = np.conj(A)/2
    if cfg.cross_df:
        # Frame U = exp(i(w_B P+ + w_A P-)t), D = w_B - w_A: off-resonant partner couplings
        D = TAU*cfg.cross_df
        t = t0 + np.cumsum(dt) - dt/2
        cA = A/2*np.exp(-1j*D*t); cB = B/2*np.exp(1j*D*t)
        H[:, 1, 0] += cA; H[:, 0, 1] += np.conj(cA)
        H[:, 1, 2] += cB; H[:, 2, 1] += np.conj(cB)
    E, V = np.linalg.eigh(H)
    return (V*np.exp(-1j*E*dt[:, None])[:, None, :]) @ np.conj(np.swapaxes(V, 1, 2))

def _drive_unitaries(seg, cfg, t0):
    dt, A, B = seg['dt'], seg['A'], seg['B']
    if not cfg.cross_df:
        return _step_unitaries(dt, A, B, cfg, t0)
    M = max(int(np.ceil(dt.max()*TAU*abs(cfg.cross_df)/cfg.cross_res)), 1)
    # Limit peak memory: at 51 mT, refinement creates many fast sub-steps.
    blocks = np.empty((len(dt), 3, 3), complex)
    edges = np.concatenate([[0.], np.cumsum(dt)])
    for start in range(0, len(dt), 128):
        stop = min(start+128, len(dt))
        U = _step_unitaries(np.repeat(dt[start:stop]/M, M),
                            np.repeat(A[start:stop], M), np.repeat(B[start:stop], M),
                            cfg, t0+edges[start]).reshape(stop-start, M, 3, 3)
        part = U[:, 0].copy()
        for j in range(1, M): part = U[:, j] @ part
        blocks[start:stop] = part
    return blocks

def evolve(states, x, segs, cfg, t0=0.0):
    gx = GAMMA*np.asarray(x)[:, None]*ZD
    t = t0
    for seg in segs:
        U = _drive_unitaries(seg, cfg, t)
        for k, dtk in enumerate(seg['dt']):
            h = np.exp(-1j*gx*dtk/2)
            states = h*((h*states) @ U[k].T)
        t += seg['dt'].sum()
    return states, t

def probs_at(method, x, wait, ctl=HONEST, cfg=Cfg()):
    x = np.asarray(x, float)
    st = np.broadcast_to(PSI0, (len(x), 3)).copy()
    st, t = evolve(st, x, segments(method, ctl, cfg), cfg)
    st = st*np.exp(-1j*GAMMA*x[:, None]*ZD*wait)
    out = []
    for ph in (np.pi/2, -np.pi/2):
        s2, _ = evolve(st, x, segments(method, ctl, cfg, ph, analysis=True), cfg, t+wait)
        out.append(np.abs(s2[:, 1])**2)
    return np.stack(out, 1)

def noise_grid(env):
    if np.isinf(env.T2_DQ):
        z, w = np.array([0.]), np.array([1.])
    else:
        z, w = np.polynomial.hermite_e.hermegauss(env.nodes); w = w/w.sum()
        z = z/(np.sqrt(2)*GAMMA*env.T2_DQ)
    return (np.concatenate([z+o for o, _ in env.nuclear]),
            np.concatenate([w*p for _, p in env.nuclear]))

def probabilities(method, fields, wait=WAIT, ctl=HONEST, cfg=Cfg(), env=Env()):
    fields = np.atleast_1d(np.asarray(fields, float))
    offs, wts = noise_grid(env)
    P = probs_at(method, (fields[:, None]+offs).ravel(), wait, ctl, cfg)
    return np.einsum('fnq,n->fq', P.reshape(len(fields), len(offs), 2), wts)

def response(method, fields, wait=WAIT, ctl=HONEST, cfg=Cfg(), env=Env()):
    return probabilities(method, fields, wait, ctl, cfg, env) @ np.array([1., -1.])

_CAL = {}
def calibrate(method, wait=WAIT, cfg=Cfg(), env=Env(), extent=.3, points=41):
    # Honest calibration, frozen afterwards. Returns None if the branch is not monotone.
    key = (method, wait, cfg, env, extent, points)
    if key not in _CAL:
        grid = np.linspace(-extent, extent, points)
        sig = response(method, grid, wait, HONEST, cfg, env)
        d = np.diff(sig)
        if np.all(d > 0) or np.all(d < 0):
            o = np.argsort(sig)
            _CAL[key] = PchipInterpolator(sig[o], grid[o], extrapolate=False)
        else:
            _CAL[key] = None
    return _CAL[key]

def max_bias(method, ctl, fields=FIELDS, wait=WAIT, cfg=Cfg(), env=Env()):
    # Max |estimate - true| over the field grid (nT); NaN if any reading is out of range.
    inv = calibrate(method, wait, cfg, env)
    if inv is None:
        return np.nan, 0.0
    est = inv(response(method, fields, wait, ctl, cfg, env))
    err = 1000*(est - fields)
    ok = np.isfinite(err)
    return (float(np.max(np.abs(err))) if ok.all() else np.nan), float(ok.mean())

def sensitivity(method, ctl=HONEST, wait=WAIT, cfg=Cfg(), env=Env(), n0=.05, n1=.035, overhead=3.):
    # eta_true uses the actual (disturbed) slope; eta_reported is the noise of the frozen estimator.
    h = .0005
    pp = probabilities(method, [-h, 0, h], wait, ctl, cfg, env)
    sig = pp @ np.array([1., -1.]); slope = (sig[2]-sig[0])/(2*h)
    nom = response(method, [-h, h], wait, HONEST, cfg, env); nslope = (nom[1]-nom[0])/(2*h)
    p = pp[1]; var = np.sum((n1+(n0-n1)*p)/(n0-n1)**2 + p*(1-p))
    num = 1000*np.sqrt(var*2*(overhead + pulse_time(method, cfg) + wait)*1e-6)
    return num/abs(slope), num/abs(nslope)

print('engine ready')


# %% Analysis block 5
OC_AMPS_MHZ = [1.056622, 0.911127, 4.438505, 3.450269, 6.024526, 0.387528, 9.745779, 1.654862, 0.338571, 6.31541, 3.676903, 7.819442, 0.56561, 9.160424, 5.365173, 0.335426, 9.257769, 4.022773, 3.716678, 8.431501, 9.3524, 6.932224, 0.434238, 8.609887, 0.242267, 1.783032, 6.545758, 0.493736, 4.274106, 4.800949, 9.535023, 5.518012, 0.459897, 9.376237, 0.070763, 9.979768, 1.764912, 0.634871, 8.911042, 8.15162]
OC_PHASES = [3.854498, 0.360426, 3.949865, 4.863947, 5.106377, 4.232477, 4.210433, 1.023652, 0.195063, 0.560381, 5.799376, 3.75438, 5.638623, 1.768592, 3.23021, 0.809406, 1.434819, 1.596861, 3.826696, 4.0406, 3.665388, 3.262403, 3.703445, 6.25891, 2.897957, 5.267793, -0.624144, 2.432043, 3.491547, 3.242534, 1.450896, 2.322265, 5.925285, 4.040194, 3.651582, 2.171915, -0.354628, 0.882951, -1.630475, 5.539145]
OC_T = 2.2

RUN_OC_DESIGN = False
def design_oc(T=2.2, N=40, scales=np.linspace(.5, 1.5, 11), starts=6, seed=1, peak=TAU*10):
    def su2(theta, phi):
        c, s = np.cos(theta/2), np.sin(theta/2)
        U = np.empty(theta.shape+(2, 2), complex)
        U[..., 0, 0] = c; U[..., 1, 1] = c
        U[..., 0, 1] = -1j*s*np.exp(-1j*phi); U[..., 1, 0] = -1j*s*np.exp(1j*phi)
        return U
    target = su2(np.array(np.pi), np.array(0.0)); dt = T/N
    unpack = lambda p: (peak*0.5*(1+np.tanh(p[:N])), p[N:])
    def infid(p):
        u, ph = unpack(p)
        Us = su2(scales[:, None]*u[None, :]*dt, np.broadcast_to(ph, (len(scales), N)))
        U = np.broadcast_to(np.eye(2), (len(scales), 2, 2)).astype(complex)
        for k in range(N): U = Us[:, k] @ U
        return 1 - (np.abs(np.einsum('ij,sij->s', np.conj(target), U))**2/4).mean()
    rng = np.random.default_rng(seed); best = None
    for _ in range(starts):
        p0 = np.concatenate([rng.normal(0, 1, N), rng.uniform(0, TAU, N)])
        r = minimize(infid, p0, method='L-BFGS-B', options=dict(maxiter=3000, maxfun=400000))
        best = r if best is None or r.fun < best.fun else best
    u, ph = unpack(best.x)
    return list(u/TAU), list(ph), best.fun

if RUN_OC_DESIGN:
    OC_AMPS_MHZ, OC_PHASES, f = design_oc(); print('redesigned, mean infidelity', f)

CFG = Cfg(oc_pulse=(tuple(np.array(OC_AMPS_MHZ)*TAU), tuple(OC_PHASES), OC_T))
print('OC pulse: 40 segments,', OC_T, 'us, peak', round(max(OC_AMPS_MHZ), 2), 'MHz (limit 10 MHz)')

# %% Analysis block 7
paper = {'hard': [164.98, 15.84, 15.66, 204.75], 'bb1': [34.22, 0.55, 0.56, 97.88], 'stirap': [20.48, 3.12, 4.40, 7.88]}
F13 = np.linspace(-.2, .2, 13)
rows = []
for m, ref in paper.items():
    new = [max_bias(m, Ctl(s, s), F13, cfg=CFG)[0] for s in (.6, .9, 1.1, 1.5)]
    rows.append({'method': m, **{f'x{s} new': round(v, 2) for s, v in zip((.6, .9, 1.1, 1.5), new)},
                 'max |new - paper| (nT)': round(max(abs(a-b) for a, b in zip(new, ref)), 3)})
check1 = pd.DataFrame(rows); display(check1)
assert check1['max |new - paper| (nT)'].max() < 0.05, 'engine does not reproduce the paper'

RUN_SLOW_REFERENCE_51 = False

# This reference does not read segments() or _drive_unitaries().
def analytic_plan(method, ctl, cfg, phase_B=0., analysis=False):
    eB = np.exp(1j*phase_B)
    if method in ('hard', 'bb1', 'oc'):
        if method == 'oc':
            amps, phs, total = cfg.oc_pulse
            plan = [(float(a), float(p), total/len(amps)) for a,p in zip(amps,phs)]
        else:
            rotations = [(np.pi,0.)] if method == 'hard' else BB1
            plan = [(cfg.peak,p,angle/cfg.peak) for angle,p in rotations]
        return [(T, lambda t,a=a,p=p: (a/np.sqrt(2)*ctl.sA*np.exp(1j*p),
                  a/np.sqrt(2)*ctl.sB*np.exp(1j*p)*eB)) for a,p,T in plan]
    if method != 'stirap': raise ValueError(method)
    Ts,Th = cfg.arp_time,cfg.half_time
    def arp(reverse):
        def envelope(t):
            ph = (-1 if reverse else 1)*cfg.chirp_span*(t*t/Ts-t)
            return 0j, cfg.peak*ctl.sB*np.sin(np.pi*t/Ts)*np.exp(1j*ph)*eB
        return Ts,envelope
    def half(reverse):
        def envelope(t):
            x = (Th-t)/Th if reverse else t/Th
            if x < .2: R=np.sin(np.pi*x/.4)**2
            elif x > .8: R=np.cos(np.pi*(x-.8)/.4)**2
            else: R=1.
            u=np.clip((x-.2)/.6,0.,1.)
            theta=np.pi/8*(1-np.cos(np.pi*u))
            return cfg.peak*ctl.sA*R*np.cos(theta),cfg.peak*ctl.sB*R*np.sin(theta)*eB
        return Th,envelope
    return [half(True),arp(True)] if analysis else [arp(False),half(False)]

def reference_P(method, field, ctl, cfg=CFG, wait=WAIT):
    def propagate(plan, state, absolute_start):
        for duration,envelope in plan:
            origin=absolute_start
            def rhs(t,y):
                A,B=envelope(t)
                if cfg.cross_df:
                    phase=TAU*cfg.cross_df*(origin+t)
                    A,B=A+B*np.exp(1j*phase),B+A*np.exp(-1j*phase)
                H=np.diag(GAMMA*field*ZD).astype(complex)
                H[1,0]=B/2;H[0,1]=np.conj(B)/2
                H[1,2]=A/2;H[2,1]=np.conj(A)/2
                return -1j*H@y
            # Resolve the fast oscillation without sharing the splitting grid.
            maxstep=min(.002,1/(12*abs(cfg.cross_df))) if cfg.cross_df else .002
            sol=solve_ivp(rhs,(0.,duration),state,method='DOP853',
                          rtol=1e-10,atol=1e-12,max_step=maxstep)
            if not sol.success: raise RuntimeError(sol.message)
            state=sol.y[:,-1];absolute_start+=duration
        return state,absolute_start
    state,t=propagate(analytic_plan(method,ctl,cfg),PSI0.copy(),0.)
    state=state*np.exp(-1j*GAMMA*field*ZD*wait)
    values=[]
    for phase in (np.pi/2,-np.pi/2):
        final,_=propagate(analytic_plan(method,ctl,cfg,phase,True),state.copy(),t+wait)
        values.append(float(abs(final[1])**2))
    return np.array(values)

checks=[('selective',m,CFG) for m in METHODS]
checks += [('5 mT cross-driving','stirap',replace(CFG,cross_df=280.)),
           ('51 mT cross-driving','hard',replace(CFG,cross_df=2856.))]
if RUN_SLOW_REFERENCE_51:
    checks.append(('51 mT cross-driving','stirap',replace(CFG,cross_df=2856.)))
reference_rows=[]
for scenario,m,config in checks:
    print('Independent reference:',scenario,m,elapsed(),flush=True)
    ctl=Ctl(1.5,1.5);field=.1
    direct=reference_P(m,field,ctl,config)
    split=probs_at(m,[field],WAIT,ctl,config)[0]
    coarse_difference=float(np.max(abs(direct-split)))
    checked_config=config
    if coarse_difference >= 1e-5:
        checked_config=replace(config,dt=config.dt/2)
        split=probs_at(m,[field],WAIT,ctl,checked_config)[0]
        print('  Refined outer step for reference comparison to',checked_config.dt,'us',flush=True)
    reference_rows.append(dict(scenario=scenario,method=m,field_uT=field,amplitude_scale=1.5,
        coarse_probability_difference=coarse_difference, checked_dt_us=checked_config.dt,
        max_probability_difference=float(np.max(abs(direct-split))),
        reference_Pplus=direct[0],reference_Pminus=direct[1],
        split_Pplus=split[0],split_Pminus=split[1]))
reference_checks=pd.DataFrame(reference_rows)
reference_checks['passes_probability_tolerance']=reference_checks.max_probability_difference < 1e-5
reference_checks.to_csv(f'{OUT}/independent_reference_checks.csv',index=False)
display(reference_checks)
if not reference_checks.passes_probability_tolerance.all():
    print('WARNING: reference tolerance exceeded. Refine numerics before quoting these cases.')

# Check stored OC design fidelity without rerunning the optimizer.
from scipy.linalg import expm
oc_errors=[];amps,phs,total=CFG.oc_pulse
target=np.array([[0,-1j],[-1j,0]],complex)
for scale in np.linspace(.5,1.5,11):
    U=np.eye(2,dtype=complex)
    for amp,phase in zip(amps,phs):
        H=scale*amp/2*np.array([[0,np.exp(-1j*phase)],[np.exp(1j*phase),0]])
        U=expm(-1j*H*total/len(amps))@U
    oc_errors.append(dict(amplitude_scale=scale,infidelity=1-abs(np.trace(target.conj().T@U))**2/4))
oc_design_check=pd.DataFrame(oc_errors)
oc_design_check.to_csv(f'{OUT}/oc_design_fidelity.csv',index=False)
print('Stored OC mean design infidelity:',oc_design_check.infidelity.mean())
for m in METHODS:
    st,_=evolve(np.array([PSI0]),[0.],segments(m,HONEST,CFG),CFG)
    print(LABEL[m],'nominal populations:',np.abs(st[0])**2)
print('Validation complete',elapsed())


# %% Analysis block 9
SCALES = np.round(np.arange(0.5, 1.5001, 0.05), 3)
rows = []
for m in METHODS:
    for s in SCALES:
        b, v = max_bias(m, Ctl(s, s), cfg=CFG)
        rows.append({'method': m, 'amplitude_scale': s, 'amplitude_change_percent': round(100*(s-1)),
                     'max_abs_bias_nT': b, 'valid_fraction': v})
    print(m, 'done', elapsed(), flush=True)
sweep = pd.DataFrame(rows); sweep.to_csv(f'{OUT}/result1_common_amplitude_sweep.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
for m in METHODS:
    d = sweep[sweep.method == m]
    ax.plot(d.amplitude_change_percent, d.max_abs_bias_nT, 'o-', ms=3, color=COLOR[m], label=LABEL[m])
ax.set(xlabel='Common microwave amplitude change (%)', ylabel='Maximum absolute field bias (nT)',
       title='Result 1: bias over −200 to +200 nT, frozen honest calibration')
ax.legend(); fig.tight_layout(); fig.savefig(f'{OUT}/fig1_common_amplitude.png', dpi=200); plt.show()

table1 = sweep[sweep.amplitude_scale.isin([0.6, 0.9, 1.1, 1.5])].pivot(
    index='amplitude_change_percent', columns='method', values='max_abs_bias_nT')[METHODS].round(2)
table1.columns = [LABEL[m] for m in METHODS]
table1.to_csv(f'{OUT}/table1_bias.csv'); display(table1)

# %% Analysis block 11
n0, n1, overhead = .05, .035, 3.
rows = []
for m in METHODS:
    inv = calibrate(m, cfg=CFG); dinv = inv.derivative()
    eta_nominal = sensitivity(m, cfg=CFG)[0]
    for s in (0.9, 0.6, 1.5):
        pp = probabilities(m, FIELDS, ctl=Ctl(s, s), cfg=CFG); S = pp @ np.array([1., -1.])
        err = 1000*(inv(S) - FIELDS)
        if not np.all(np.isfinite(err)):
            rows.append({'method': LABEL[m], 'amplitude_change_percent': round(100*(s-1)),
                         'eta_nominal_nT_rtHz': eta_nominal, 'valid_fraction': float(np.isfinite(err).mean()),
                         'status': 'invalid calibration branch; crossover undefined'})
            continue
        k = int(np.argmax(np.abs(err)))
        p = pp[k]; V = np.sum((n1+(n0-n1)*p)/(n0-n1)**2 + p*(1-p))
        eta_b = 1000*np.sqrt(V*2*(overhead + pulse_time(m, CFG) + WAIT)*1e-6)*abs(float(dinv(S[k])))
        E = abs(err[k])
        rows.append({'method': LABEL[m], 'amplitude_change_percent': round(100*(s-1)),
                     'eta_nominal_nT_rtHz': round(eta_nominal, 2), 'field_of_max_bias_nT': round(1000*FIELDS[k]),
                     'max_bias_nT': round(E, 2), 'eta_rep_at_that_field_nT_rtHz': round(eta_b, 2),
                     'valid_fraction': 1., 'status': 'ok',
                     'crossover_time_s': round((eta_b/E)**2, 3) if E > 1e-3 else np.inf})
sens = pd.DataFrame(rows); sens.to_csv(f'{OUT}/table2_sensitivity_and_crossover.csv', index=False)
display(sens)


# %% Analysis block 13
cases = {'A only -10%': Ctl(.9, 1), 'B only -10%': Ctl(1, .9), 'A only -40%': Ctl(.6, 1), 'B only -40%': Ctl(1, .6)}
rows = [{'case': name, **{LABEL[m]: round(max_bias(m, c, cfg=CFG)[0], 2) for m in METHODS}} for name, c in cases.items()]
one_tone = pd.DataFrame(rows); one_tone.to_csv(f'{OUT}/table3_one_tone.csv', index=False); display(one_tone)
print(elapsed())

# %% Analysis block 15
OFF = -A_HF/GAMMA   # positive offset from majority mI=+1 to mI=0; uT
UNPOL = tuple((A_HF*m/GAMMA, 1/3) for m in (-1, 0, 1))
PART95 = ((0.0, .95), (OFF, .025), (2*OFF, .025))
SCENARIOS = {
    'Ideal (selective drive, polarized)':        (CFG, Env()),
    '5 mT, linear MW, polarized':                (replace(CFG, cross_df=280.), Env()),
    '5 mT, linear MW, unpolarized 14N':          (replace(CFG, cross_df=280.), Env(nuclear=UNPOL)),
    '51 mT, linear MW, 95% polarized 14N':       (replace(CFG, cross_df=2856.), Env(nuclear=PART95)),
}
rows = []
for name, (cfg, env) in SCENARIOS.items():
    for m in METHODS:
        row = {'scenario': name, 'method': LABEL[m]}
        if calibrate(m, cfg=cfg, env=env) is None:
            row.update({'calibration': 'not monotone (unusable)'})
        else:
            row['calibration'] = 'ok'
            for s in (0.6, 0.9, 1.1, 1.5):
                v = max_bias(m, Ctl(s, s), cfg=cfg, env=env)[0]
                row[f'{round(100*(s-1)):+d}%'] = round(v, 2) if np.isfinite(v) else 'out of range'
            row['eta_nominal_nT_rtHz'] = round(sensitivity(m, cfg=cfg, env=env)[0], 2)
        rows.append(row)
    print(name, 'done', elapsed(), flush=True)
scen = pd.DataFrame(rows); scen.to_csv(f'{OUT}/result2_hardware_scenarios.csv', index=False); display(scen)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
for ax, col in zip(axes, ['-10%', '+50%']):
    names = list(SCENARIOS); x = np.arange(len(names)); w = 0.2
    for i, m in enumerate(METHODS):
        vals = [pd.to_numeric(scen[(scen.scenario == n) & (scen.method == LABEL[m])][col], errors='coerce').iloc[0] for n in names]
        ax.bar(x+(i-1.5)*w, vals, w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(x); ax.set_xticklabels([n.replace(', ', '\n') for n in names], fontsize=7)
    ax.set(title=f'Common amplitude {col}', ylabel='Max field bias (nT)'); ax.set_yscale('log')
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{OUT}/fig2_hardware_scenarios.png', dpi=200); plt.show()


# %% Analysis block 17
rows = []
for s in SCALES:
    st, _ = evolve(np.array([PSI0]), [0.0], segments('stirap', Ctl(s, s), CFG), CFG)
    c = st[0]; phase_err = np.angle(-c[2]/c[0])        # target (|+1> - |-1>)/sqrt2
    inv = calibrate('stirap', cfg=CFG)
    reading_at_zero = 1000*inv(response('stirap', [0.0], ctl=Ctl(s, s), cfg=CFG))[0]
    rows.append({'amplitude_scale': s, 'prep_phase_error_rad': phase_err,
                 'prep_equivalent_field_nT': 1000*phase_err/(2*GAMMA*WAIT),
                 'imbalance': abs(c[0])**2-abs(c[2])**2, 'residual_P0': abs(c[1])**2,
                 'reading_at_zero_field_nT': reading_at_zero})
diag = pd.DataFrame(rows); diag.to_csv(f'{OUT}/stirap_diagnostics.csv', index=False)
r = np.corrcoef(diag.prep_equivalent_field_nT, diag.reading_at_zero_field_nT)[0, 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(100*(diag.amplitude_scale-1), diag.reading_at_zero_field_nT, 'o-', label='Reading error at zero field (full sequence)')
ax.plot(100*(diag.amplitude_scale-1), diag.prep_equivalent_field_nT, 's--', label='Preparation phase error, as equivalent field')
ax.set(xlabel='Common amplitude change (%)', ylabel='nT', title=f'STIRAP diagnostic (correlation r = {r:.2f})')
ax.legend(); fig.tight_layout(); fig.savefig(f'{OUT}/fig3_stirap_diagnostic.png', dpi=200); plt.show()
print(f'correlation between preparation phase error and zero-field reading error: r = {r:.2f}')
fit_slope,fit_intercept=np.polyfit(diag.prep_equivalent_field_nT,diag.reading_at_zero_field_nT,1)
pd.DataFrame([dict(correlation=r,fit_slope=fit_slope,fit_intercept_nT=fit_intercept,
                    note='Correlation is diagnostic, not a causal proof.')]).to_csv(f'{OUT}/phase_correlation_summary.csv',index=False)


# %% Analysis block 19
rows = []
def emax(m, s, cfg=CFG, env=Env()): return max_bias(m, Ctl(s, s), cfg=cfg, env=env)[0]
for m in ['stirap', 'oc']:
    for s in (0.6, 1.5):
        base = emax(m, s)
        for dt in (0.001, 0.00025):
            rows.append({'test': f'dt = {dt*1000:g} ns', 'method': m, 'scale': s, 'baseline_nT': base,
                         'change_nT': emax(m, s, replace(CFG, dt=dt)) - base})
        rows.append({'test': '41 noise nodes', 'method': m, 'scale': s, 'baseline_nT': base,
                     'change_nT': emax(m, s, env=Env(nodes=41)) - base})
c5 = replace(CFG, cross_df=280.)
for m in ['bb1', 'stirap']:
    base = emax(m, 1.5, c5)
    rows.append({'test': 'cross-drive sub-step x2 finer (5 mT)', 'method': m, 'scale': 1.5, 'baseline_nT': base,
                 'change_nT': emax(m, 1.5, replace(c5, cross_res=0.025)) - base})
conv = pd.DataFrame(rows).round(4); conv.to_csv(f'{OUT}/convergence.csv', index=False); display(conv)
print('largest change (nT):', conv.change_nT.abs().max(), '|', elapsed())
# Targeted 51 mT checks: include the assumed partial nuclear population.
c51=replace(CFG,cross_df=2856.)
e51=Env(nuclear=PART95)
for m,scale in [('stirap',1.5),('stirap',.9),('bb1',1.5)]:
    print('51 mT convergence:',m,scale,elapsed(),flush=True)
    base=emax(m,scale,c51,e51)
    for name,config,environment in [
        ('51 mT cross sub-step twice finer',replace(c51,cross_res=.025),e51),
        ('51 mT outer step twice finer',replace(c51,dt=.00025),e51),
        ('51 mT 41 noise nodes',c51,replace(e51,nodes=41)),
    ]:
        refined=emax(m,scale,config,environment)
        rows.append(dict(test=name,method=m,scale=scale,baseline_nT=base,
                         change_nT=refined-base,refined_nT=refined))
conv=pd.DataFrame(rows)
conv['passes_bias_tolerance']=np.isfinite(conv.change_nT) & (abs(conv.change_nT)<.1)
conv.to_csv(f'{OUT}/convergence.csv',index=False)
display(conv)
if not conv.passes_bias_tolerance.all():
    print('WARNING: at least one bias-convergence check is invalid or exceeds 0.1 nT. Review before quoting results.')
print('Full convergence checks finished',elapsed())


# %% Analysis block 21
from dataclasses import asdict
import scipy, matplotlib

def get(m,s):
    return sweep[(sweep.method==m)&np.isclose(sweep.amplitude_scale,s)].max_abs_bias_nT.iloc[0]
print('Main ideal-model results:')
for s in (.9,1.5):
    print(f'{100*(s-1):+.0f}%: '+', '.join(f'{LABEL[m]} {get(m,s):.3f} nT' for m in METHODS))
print('Hardware scenarios:');display(scen)
# CSVs, including results printed earlier, are always saved independently of notebook outputs.
check1.to_csv(f'{OUT}/baseline_reproduction.csv',index=False)
for number in plt.get_fignums():
    fig=plt.figure(number)
    fig.savefig(f'{OUT}/figure_{number:02d}.png',dpi=200,bbox_inches='tight')
    fig.savefig(f'{OUT}/figure_{number:02d}.pdf',bbox_inches='tight')
settings=dict(run_id=RUN_ID,elapsed_seconds=time.time()-T_START,python=sys.version,
              platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
              pandas=pd.__version__,matplotlib=matplotlib.__version__,
              configuration=asdict(CFG),environment=asdict(Env()),
              scenarios={name:dict(cfg=asdict(c),env=asdict(e)) for name,(c,e) in SCENARIOS.items()},
              wait_us=WAIT,fields_uT=FIELDS.tolist(),amplitude_scales=SCALES.tolist(),
              slow_51mT_reference_enabled=RUN_SLOW_REFERENCE_51)
with open(f'{OUT}/run_settings.json','w') as f:json.dump(settings,f,indent=2)
status=dict(reference_checks_pass=bool(reference_checks.passes_probability_tolerance.all()),
            convergence_checks_pass=bool(conv.passes_bias_tolerance.all()),
            baseline_reproduction_pass=bool(check1['max |new - paper| (nT)'].max()<.05),
            scope='Selected cases only; not experimental validation or proof of omitted-physics accuracy.')
with open(f'{OUT}/run_status.json','w') as f:json.dump(status,f,indent=2)
with open(f'{OUT}/README.txt','w') as f:
    f.write('Reproducible simulation outputs. Read run_status.json before interpreting results.\n'
            'CSV files contain the numerical tables; figures are PNG and PDF.\n'
            'run_settings.json stores parameters, versions, nuclear weights, and OC coefficients.\n'
            'run_log.txt records printed progress; frontend display tables are separately saved as CSV.\n'
            'No full electron-nuclear or carrier-counter-rotating model is included.\n')
print('Validation status:',status)
RUN_LOG.flush()
manifest=[]
for path in sorted(Path(OUT).glob('*')):
    if path.is_file() and path.name!='manifest.json':
        manifest.append(dict(file=path.name,bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
with open(f'{OUT}/manifest.json','w') as f:json.dump(manifest,f,indent=2)
archive=Path(f'NV_robust_control_results_{RUN_ID}.zip').resolve()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(Path(OUT).glob('*')):
        if path.is_file():z.write(path,arcname=path.name)
# Stop logging before announcing the completed archive, so its log and manifest remain consistent.
sys.stdout=sys.stdout._nv_original;sys.stderr=sys.stderr._nv_original;RUN_LOG.close()
print('Saved results:',archive)
print('Save the executed notebook separately to preserve displayed outputs.')
display(FileLink(archive.name))
# The archive remains on disk; download it through the notebook file browser.

