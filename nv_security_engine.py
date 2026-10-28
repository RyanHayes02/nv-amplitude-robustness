"""Engine extracted without changing equations from run_analysis.py, commit d093f51.
Only engine definitions and stored OC coefficients are retained; no optimizer or
original benchmark runs on import. Source SHA256: 89c7081486cf7f35eced02afcf53ba154275b5f6d322dac150f5db3ec49a035e
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator
WAIT=6.0
FIELDS=np.linspace(-0.2,0.2,25)
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


OC_AMPS_MHZ = [1.056622, 0.911127, 4.438505, 3.450269, 6.024526, 0.387528, 9.745779, 1.654862, 0.338571, 6.31541, 3.676903, 7.819442, 0.56561, 9.160424, 5.365173, 0.335426, 9.257769, 4.022773, 3.716678, 8.431501, 9.3524, 6.932224, 0.434238, 8.609887, 0.242267, 1.783032, 6.545758, 0.493736, 4.274106, 4.800949, 9.535023, 5.518012, 0.459897, 9.376237, 0.070763, 9.979768, 1.764912, 0.634871, 8.911042, 8.15162]
OC_PHASES = [3.854498, 0.360426, 3.949865, 4.863947, 5.106377, 4.232477, 4.210433, 1.023652, 0.195063, 0.560381, 5.799376, 3.75438, 5.638623, 1.768592, 3.23021, 0.809406, 1.434819, 1.596861, 3.826696, 4.0406, 3.665388, 3.262403, 3.703445, 6.25891, 2.897957, 5.267793, -0.624144, 2.432043, 3.491547, 3.242534, 1.450896, 2.322265, 5.925285, 4.040194, 3.651582, 2.171915, -0.354628, 0.882951, -1.630475, 5.539145]
OC_T = 2.2
CFG = Cfg(oc_pulse=(tuple(np.array(OC_AMPS_MHZ)*TAU), tuple(OC_PHASES), OC_T))
