"""Bilancio idrico del suolo con la funzione di perdita di Laio et al. (2001), a tre strati, forzato
dalla pioggia oraria interpolata e dall'evaporazione potenziale (Hargreaves, ripartita sulle ore di sole).

Strato i (sensore a 15, 30, 60 cm), contenuto d'acqua theta, spessore Z_i (mm):
  Z_i dtheta/dt = ingresso_i - E_i(theta) - L_i(theta)
  E_i   = c_i * ET0(t) * min(1, max(0, (theta - th_h)/(th_* - th_h)))     suolo nudo: E_w = 0, s_w = s_h
  L_i   = Ks_i * (exp(b_i (theta - th_fc)) - 1) / (exp(b_i (th_s - th_fc)) - 1)   per theta > th_fc
  ingresso_1 = pioggia (eccesso oltre la saturazione -> ruscellamento); ingresso_i = L_{i-1}
th_h e th_s fissati dai minimi/massimi osservati; th_*, th_fc, Ks, b, c tarati.
"""
import sys, json
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import differential_evolution

OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/outputs/vwc'
import os; os.makedirs(OUT, exist_ok=True)
LAT = 40.693522
Z = np.array([225.0, 225.0, 300.0])            # 0-22.5, 22.5-45, 45-75 cm
NAMES = ['15 cm', '30 cm', '60 cm']

# ------------------------------------------------------------------ dati
raw = pd.read_pickle('/home/claude/data/vwc/raw.pkl').set_index('t')
g = raw[['v15', 'v30', 'v60', 's15', 's30', 's60']].resample('1h', label='right', closed='right')
obs = g.mean().where(g.count() >= 9)
s = pd.read_csv('/home/claude/spatial-interpolation_salerno/dati/salerno_punto_orario_2016_2026_tarato.csv',
                parse_dates=['data_ora_fine_intervallo_solare']).set_index('data_ora_fine_intervallo_solare')
H = s.index
P = s.pioggia_stima_tarata_mm.fillna(0).to_numpy()
T = s.temperatura_degC.interpolate(limit_direction='both')

# ET0 Hargreaves giornaliera, ripartita sulle ore in proporzione al seno dell'altezza solare
day = (H - pd.Timedelta(minutes=1)).floor('D')
Td = T.groupby(day).agg(['mean', 'min', 'max'])
J = Td.index.dayofyear.to_numpy()
phi = np.radians(LAT); dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365); dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
ws = np.arccos(-np.tan(phi) * np.tan(dec))
Ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws)) * 0.408
ET0d = pd.Series(0.0023 * Ra * (Td['mean'] + 17.8) * np.sqrt((Td['max'] - Td['min']).clip(lower=0)), index=Td.index)
mid = H - pd.Timedelta(minutes=30)                        # meta' dell'ora (ora solare = UTC+1)
jh = mid.dayofyear.to_numpy(); hr = mid.hour + mid.minute / 60
B = 2 * np.pi * (jh - 81) / 364; eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
tsol = hr + (4 * (14.760576 - 15) + eot) / 60
decl = 0.409 * np.sin(2 * np.pi * jh / 365 - 1.39)
sinel = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(np.radians(15 * (tsol - 12))), 0, None)
w = pd.Series(sinel, index=H).groupby(day).transform(lambda x: x / x.sum() if x.sum() > 0 else x)
ET0 = (w.to_numpy() * ET0d.reindex(day).to_numpy())
ET0 = np.nan_to_num(ET0)

O = obs.reindex(H)[['v15', 'v30', 'v60']].to_numpy()
TH_H = np.maximum(np.nanmin(O, axis=0) - 0.003, 0.005)
TH_S = np.nanmax(O, axis=0) + 0.01


@njit(cache=True)
def simulate(P, E0, th0, prm, Z, th_h, th_s, nsub):
    n = P.shape[0]; out = np.empty((n, 3)); th = th0.copy(); dt = 1.0 / nsub
    for t in range(n):
        for k in range(nsub):
            inflow = P[t] * dt
            for i in range(3):
                tst, tfc, Ks, b, c = prm[i, 0], prm[i, 1], prm[i, 2], prm[i, 3], prm[i, 4]
                W = th[i] * Z[i] + inflow
                cap = th_s[i] * Z[i]
                excess = 0.0
                if W > cap:
                    excess = W - cap; W = cap
                if i > 0 and excess > 0:               # non entra: resta nello strato sopra se c'e' spazio
                    room = th_s[i - 1] * Z[i - 1] - th[i - 1] * Z[i - 1]
                    back = min(room, excess); th[i - 1] += back / Z[i - 1]
                x = W / Z[i]
                f = (x - th_h[i]) / (tst - th_h[i])
                if f < 0: f = 0.0
                if f > 1: f = 1.0
                E = min(c * E0[t] * dt * f, max(W - th_h[i] * Z[i], 0.0))
                W -= E; x = W / Z[i]
                L = 0.0
                if x > tfc:
                    L = Ks * dt * (np.exp(b * (x - tfc)) - 1.0) / (np.exp(b * (th_s[i] - tfc)) - 1.0)
                    L = min(L, (x - tfc) * Z[i])
                W -= L
                th[i] = W / Z[i]
                inflow = L
        out[t, :] = th
    return out


def unpack(v):
    prm = np.empty((3, 5))
    for i in range(3):
        a, f, lk, b, c = v[5 * i:5 * i + 5]
        tst = TH_H[i] + a * (TH_S[i] - TH_H[i])
        tfc = tst + f * (TH_S[i] - tst)
        prm[i] = (tst, tfc, 10 ** lk, b, c)
    return prm


BOUNDS = [(0.02, 0.9), (0.0, 0.95), (-1.5, 2.5), (5, 200), (0.0, 1.5)] * 3


def run(v, i0, i1, th0):
    return simulate(P[i0:i1], ET0[i0:i1], th0, unpack(v), Z, TH_H, TH_S, 4)


def score(sim, ob):
    m = ~np.isnan(ob)
    e = [np.mean((sim[m[:, j], j] - ob[m[:, j], j]) ** 2) / np.var(ob[m[:, j], j]) for j in range(3)]
    return float(np.mean(e))


def metrics(sim, ob):
    r = {}
    for j, nm in enumerate(NAMES):
        m = ~np.isnan(ob[:, j]); a, o = sim[m, j], ob[m, j]
        cc = np.corrcoef(a, o)[0, 1]
        r[nm] = dict(NSE=1 - np.mean((a - o) ** 2) / np.var(o), KGE=1 - np.sqrt((cc - 1) ** 2 + (a.std() / o.std() - 1) ** 2 + (a.mean() / o.mean() - 1) ** 2),
                     corr=cc, RMSE=float(np.sqrt(np.mean((a - o) ** 2))), bias=float(np.mean(a - o)))
    return r


i_start = H.get_loc(obs.v15.first_valid_index())
i_split = H.get_loc(pd.Timestamp('2026-02-01 00:00'))
i_end = H.get_loc(min(obs.v15.last_valid_index(), H[-1])) + 1
WARM = 24 * 30


def calibrate(i0, i1, seed=1):
    th0 = O[i0]; ob = O[i0:i1].copy(); ob[:WARM] = np.nan
    f = lambda v: score(run(v, i0, i1, th0), ob)
    res = differential_evolution(f, BOUNDS, seed=seed, popsize=20, maxiter=600, tol=1e-7, polish=True, workers=1, updating='deferred')
    return res.x, res.fun


if __name__ == '__main__':
    import time
    t0 = time.time(); run(np.array([b[0] for b in BOUNDS]), 0, 100, O[i_start]); print('compilato', time.time() - t0)
    rep = {}
    # verifica incrociata: tarato 2025 -> verifica 2026, e viceversa
    va, fa = calibrate(i_start, i_split)
    simA = run(va, i_start, i_end, O[i_start])
    rep['tarato feb2025-gen2026'] = dict(taratura=metrics(simA[:i_split - i_start], O[i_start:i_split]),
                                        verifica=metrics(simA[i_split - i_start:], O[i_split:i_end]))
    print('A', fa, time.time() - t0)
    vb, fb = calibrate(i_split, i_end)
    simB = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(vb), Z, TH_H, TH_S, 4)
    rep['tarato feb2026-nov2026'] = dict(taratura=metrics(simB[i_split - i_start:], O[i_split:i_end]),
                                        verifica=metrics(simB[:i_split - i_start], O[i_start:i_split]))
    print('B', fb, time.time() - t0)
    vf, ff = calibrate(i_start, i_end)
    simF = run(vf, i_start, i_end, O[i_start])
    rep['tarato tutto'] = dict(taratura=metrics(simF, O[i_start:i_end]))
    print('F', ff, time.time() - t0)
    for nm, v in (('A', va), ('B', vb), ('F', vf)):
        print(nm, np.round(unpack(v), 4).tolist())
    rep['parametri'] = {k: unpack(v).tolist() for k, v in (('A', va), ('B', vb), ('F', vf))}
    rep['th_h'] = TH_H.tolist(); rep['th_s'] = TH_S.tolist(); rep['Z'] = Z.tolist()
    json.dump(rep, open(f'{OUT}/taratura_laio.json', 'w'), indent=1)
    np.save(f'{OUT}/vf.npy', vf); np.save(f'{OUT}/va.npy', va); np.save(f'{OUT}/vb.npy', vb)
    for k in ('tarato feb2025-gen2026', 'tarato feb2026-nov2026'):
        for ph in ('taratura', 'verifica'):
            print(k, ph, {n: {m: round(x, 3) for m, x in d.items()} for n, d in rep[k][ph].items()})
    print('tutto', {n: {m: round(x, 3) for m, x in d.items()} for n, d in rep['tarato tutto']['taratura'].items()})
