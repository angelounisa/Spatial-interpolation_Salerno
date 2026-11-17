"""Equazione di Richards 1D con il metodo delle linee, adattando l'impostazione di A. Ireson
(github.com/amireson/RichardsEquation_improved): stato = potenziale matriciale psi nei centri cella,
integrazione con odeint (LSODA, Jacobiano a banda), immagazzinamento specifico Ss (capacita' > 0 a saturazione),
infiltrazione limitata dalla capacita' della superficie (eccesso -> ruscellamento), drenaggio libero al fondo.
Adattamenti: tre orizzonti van Genuchten (0-22.5, 22.5-45, 45-150 cm), evaporazione del suolo nudo come pozzo
che decresce esponenzialmente con la profondita' (scala zE) e si riduce quando il suolo si asciuga
(tra psi_w e psi_d, in scala logaritmica), forzanti orarie a gradino. Unita': cm, ore."""
import sys, json, time
import numpy as np
from numba import njit
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
sys.argv = [sys.argv[0], '/home/claude/outputs/vwc'] + sys.argv[1:]
from laio_vwc import P, ET0, O, H, i_start, i_split, i_end, NAMES, metrics

OUT = '/home/claude/outputs/vwc'
DZ = 2.0; L = 150.0
z = np.arange(DZ / 2, L, DZ); NC = len(z)
HOR = np.where(z < 22.5, 0, np.where(z < 45, 1, 2))
SENS = np.array([np.argmin(abs(z - d)) for d in (15, 30, 60)])
TH_S = np.nanmax(O, axis=0) + 0.01
SS = 1e-6
ALPHA0 = np.array([0.79906, 0.03746, 0.01227])


@njit(cache=True)
def props(psi, thr, ths, al, n, ks):
    m = 1 - 1 / n
    se = np.empty_like(psi)
    for i in range(psi.size):
        se[i] = 1.0 if psi[i] >= 0 else (1 + (-psi[i] * al[i]) ** n[i]) ** (-m[i])
    th = thr + (ths - thr) * se
    dse = al * m / (1 - m) * se ** (1 / m) * (1 - se ** (1 / m)) ** m
    C = se * SS + (ths - thr) * dse
    K = ks * se ** 0.5 * (1 - (1 - se ** (1 / m)) ** m) ** 2
    return th, C, K


@njit(cache=True)
def rhs(psi, t, dz, z, rain, pe, thr, ths, al, n, ks, zE, lw, ld):
    nc = psi.size
    k = int(t)
    if k >= rain.size: k = rain.size - 1
    th, C, K = props(psi, thr, ths, al, n, ks)
    q = np.zeros(nc + 1)
    cap = -K[0] * (psi[0] / dz * 2.0 - 1.0)              # capacita' d'infiltrazione (psi superficie = 0)
    q[0] = min(rain[k], cap)
    for i in range(1, nc):
        q[i] = -0.5 * (K[i] + K[i - 1]) * ((psi[i] - psi[i - 1]) / dz - 1.0)
    q[nc] = K[nc - 1]
    rd = np.exp(-z / zE); rd /= np.sum(rd * dz)
    out = np.empty(nc)
    for i in range(nc):
        lp = np.log10(max(-psi[i], 1e-3))
        a = 1.0 if lp <= lw else (0.0 if lp >= ld else (ld - lp) / (ld - lw))
        out[i] = (-(q[i + 1] - q[i]) / dz - pe[k] * a * rd[i]) / C[i]
    return out


TH_MIN = np.nanmin(O, axis=0)


def unpack(v):
    al = ALPHA0 * 10 ** v[3:6]; nn = v[6:9]; ks = 10 ** v[0:3]
    thr = (np.clip(v[13:16], 0, 0.95) * TH_MIN) if len(v) > 13 else np.zeros(3)
    return dict(thr=thr[HOR], ths=TH_S[HOR], al=al[HOR], n=nn[HOR], ks=ks[HOR], c=v[9], zE=v[10], lw=v[11], ld=v[12])


def psi_from_theta(th, ths, al, n, thr=0.0):
    se = np.clip((th - thr) / (ths - thr), 1e-6, 0.999)
    return -((se ** (-1 / (1 - 1 / n)) - 1) ** (1 / n)) / al


def simulate(v, i0, i1, th3):
    p = unpack(v)
    ps3 = np.array([psi_from_theta(th3[j], TH_S[j], p['al'][SENS[j]], p['n'][SENS[j]], p['thr'][SENS[j]]) for j in range(3)])
    psi0 = np.interp(z, [15, 30, 60], ps3)
    rain = P[i0:i1] / 10.0; pe = p['c'] * ET0[i0:i1] / 10.0
    t = np.arange(0, i1 - i0 + 1, dtype=float)
    with np.errstate(all='ignore'):
        psi = odeint(rhs, psi0, t, args=(DZ, z, rain, pe, p['thr'], p['ths'], p['al'], p['n'], p['ks'], p['zE'], p['lw'], p['ld']),
                     ml=1, mu=1, mxstep=50000, hmax=0.5, rtol=1e-5, atol=1e-4)
    th = np.empty((i1 - i0, 3))
    for j in range(3):
        c = SENS[j]
        th[:, j] = props(psi[1:, c].copy(), p['thr'][c:c + 1].repeat(len(t) - 1), p['ths'][c:c + 1].repeat(len(t) - 1),
                         p['al'][c:c + 1].repeat(len(t) - 1), p['n'][c:c + 1].repeat(len(t) - 1), p['ks'][c:c + 1].repeat(len(t) - 1))[0]
    return th


# v = [logKs1..3 cm/h, log10 fattore alpha 1..3, n1..3, c, zE cm, log10|psi_w| cm, log10|psi_d| cm]
BOUNDS = [(-1.5, 2.0)] * 3 + [(-1.0, 1.0)] * 3 + [(1.08, 2.0)] * 3 + [(0.2, 1.5), (3.0, 60.0), (1.0, 3.5), (3.6, 5.5)] + [(0.0, 0.95)] * 3


def score(sim, ob):
    if not np.all(np.isfinite(sim)): return 10.0
    m = ~np.isnan(ob)
    return float(np.mean([np.mean((sim[m[:, j], j] - ob[m[:, j], j]) ** 2) / np.nanvar(ob[:, j]) for j in range(3)]))


class Obj:
    def __init__(self, i0, i1):
        self.i0, self.i1 = i0, i1
        self.ob = O[i0:i1].copy(); self.ob[:24 * 30] = np.nan
    def __call__(self, v):
        try:
            return score(simulate(v, self.i0, self.i1, O[self.i0]), self.ob)
        except Exception:
            return 10.0


def calibrate(i0, i1, maxiter, seed=3):
    f = Obj(i0, i1)
    return differential_evolution(f, BOUNDS, seed=seed, popsize=6, maxiter=maxiter, tol=1e-6, polish=False, workers=-1, updating='deferred')


def local(i0, i1, v_start, maxfev):
    from scipy.optimize import minimize
    f = Obj(i0, i1); lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    return minimize(f, np.clip(v_start, lo, hi), method='Powell', bounds=BOUNDS, options=dict(maxfev=maxfev, xtol=1e-3, ftol=1e-4))


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[2] == 'local':
    which = sys.argv[3]; MF = int(sys.argv[4])
    V0 = np.array([1.5, 1.5, 0.5, -0.5, 0.0, -0.5, 1.2, 1.2, 1.2, 1.0, 30.0, 2.3, 3.8, 0.5, 0.5, 0.5])
    rng_ = {'A': (i_start, i_split), 'B': (i_split, i_end), 'F': (i_start, i_end)}[which]
    if which == 'F' and len(sys.argv) > 5:                # parte dalla migliore delle due tarature annuali
        V0 = np.load(sys.argv[5])
    t0 = time.time(); r = local(*rng_, V0, MF)
    np.save(f'{OUT}/rich2_{which}.npy', r.x); print(which, r.fun, r.nfev, round(time.time() - t0), np.round(r.x, 3).tolist(), flush=True)
    sys.exit(0)

if __name__ == '__main__':
    v0 = np.array([0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 1.2, 1.25, 1.2, 0.7, 15.0, 2.5, 4.5])
    t0 = time.time(); s = simulate(v0, i_start, i_start + 100, O[i_start]); print('compilato', round(time.time() - t0, 1), flush=True)
    t0 = time.time(); s = simulate(v0, i_start, i_end, O[i_start]); dtr = time.time() - t0
    print('simulazione intera', round(dtr, 1), 's | score', round(score(s, O[i_start:i_end]), 3), np.nanmin(s, 0).round(3), np.nanmax(s, 0).round(3), flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == 'calib':
        MI = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        rep = {}
        ra = calibrate(i_start, i_split, MI); np.save(f'{OUT}/rich_A.npy', ra.x); print('A', ra.fun, round(time.time() - t0), flush=True)
        rb = calibrate(i_split, i_end, MI); np.save(f'{OUT}/rich_B.npy', rb.x); print('B', rb.fun, round(time.time() - t0), flush=True)
        rf = calibrate(i_start, i_end, MI); np.save(f'{OUT}/rich_F.npy', rf.x); print('F', rf.fun, round(time.time() - t0), flush=True)
        sA = simulate(ra.x, i_start, i_end, O[i_start]); sB = simulate(rb.x, i_start, i_end, O[i_start]); sF = simulate(rf.x, i_start, i_end, O[i_start])
        k = i_split - i_start
        rep['tarato 2025 → verificato 2026'] = metrics(sA[k:], O[i_split:i_end])
        rep['tarato 2026 → verificato 2025'] = metrics(sB[:k], O[i_start:i_split])
        rep['tarato su tutto'] = metrics(sF, O[i_start:i_end])
        rep['v'] = {kk: r_.x.tolist() for kk, r_ in (('A', ra), ('B', rb), ('F', rf))}
        json.dump(rep, open(f'{OUT}/taratura_richards.json', 'w'), indent=1)
        for kk in list(rep)[:3]: print(kk, {n: {m: round(x, 3) for m, x in d.items()} for n, d in rep[kk].items()})
