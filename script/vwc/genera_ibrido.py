"""Serie finale 2016-2026: modello fisico/ecoidrologico + ERA5-Land, combinati per profondita' con regressione
lineare tarata su tutto il periodo misurato e varianza riportata a quella misurata (per non smorzare i picchi).
Banda 5-95% dai residui della verifica incrociata 2025 <-> 2026 della stessa combinazione.
Uso: python genera_ibrido.py <base: laio|richards|entrambi>"""
import sys, json
import numpy as np, pandas as pd
BASE = sys.argv[1] if len(sys.argv) > 1 else 'laio'
sys.argv = [sys.argv[0], '/home/claude/outputs/vwc']
from laio_vwc import *
import ibrido_era5 as IB
OUT = '/home/claude/outputs/vwc'
E5 = IB.E5
th0 = np.nanmean(O[(H.month <= 2)], axis=0)
SIML = simulate(P, ET0, th0, unpack(np.load(f'{OUT}/vf.npy')), Z, TH_H, TH_S, 4)
bases = {'laio': [SIML]}
if BASE in ('richards', 'entrambi'):
    import richards_mol as R
    import os
    vR = np.load(f'{OUT}/rich2_F.npy' if os.path.exists(f'{OUT}/rich2_F.npy') else f'{OUT}/rich_F.npy')
    SIMR = R.simulate(vR, 0, len(H), th0)
    bases['richards'] = [SIMR]
    if BASE == 'richards': bases.pop('laio')
use = list(bases)
mt = np.zeros(len(H), bool); mt[i_start:i_end] = True
HY = np.empty((len(H), 3)); coef = {}
for j in range(3):
    X = np.column_stack([np.ones(len(H))] + [bases[b][0][:, j] for b in use] + [E5[:, 0], E5[:, 1], E5[:, 2]])
    m = mt & ~np.isnan(O[:, j]) & np.all(np.isfinite(X), 1)
    beta = np.linalg.lstsq(X[m], O[m, j], rcond=None)[0]
    p_ = X @ beta
    mu, sp = p_[m].mean(), p_[m].std(); p_ = O[m, j].mean() + (p_ - mu) * O[m, j].std() / sp
    # ore senza ERA5 (rare): solo modello di base riscalato
    bad = ~np.isfinite(p_)
    if bad.any():
        Xb = X[:, :1 + len(use)]; bb = np.linalg.lstsq(Xb[m], O[m, j], rcond=None)[0]; p_[bad] = (Xb @ bb)[bad]
    HY[:, j] = np.clip(p_, TH_H[j], TH_S[j])
    coef[NAMES[j]] = dict(zip(['costante'] + use + ['era5_0_7', 'era5_7_28', 'era5_28_100'], np.round(beta, 4).tolist()))
print('coefficienti', coef)
# residui di verifica incrociata della stessa combinazione
IB.VAR = True
models = {}
cvL = IB.cv_sims('laio')
if BASE == 'laio': models['m + ERA5-Land'] = cvL
else:
    cvR = IB.cv_sims('richards'); models['m + ERA5-Land'] = cvR
res = IB.evaluate(models)
k = i_split - i_start
pa = IB.PRED[('m + ERA5-Land', '2025→2026')][k:]; pb = IB.PRED[('m + ERA5-Land', '2026→2025')][24 * 30:k]
resid = np.vstack([pa - O[i_split:i_end], pb - O[i_start + 24 * 30:i_split]])
mon = np.r_[H[i_split:i_end].month, H[i_start + 24 * 30:i_split].month]
wet = lambda mm: np.isin(mm, [10, 11, 12, 1, 2, 3, 4])
lo = np.empty_like(HY); hi = np.empty_like(HY); wm = wet(H.month)
for j in range(3):
    for msk_r, msk in ((wet(mon), wm), (~wet(mon), ~wm)):
        r = resid[msk_r, j]; r = r[~np.isnan(r)]; a, b = np.percentile(r, [5, 95])
        lo[msk, j] = np.clip(HY[msk, j] - b, TH_H[j], TH_S[j]); hi[msk, j] = np.clip(HY[msk, j] - a, TH_H[j], TH_S[j])
cover = {NAMES[j]: float(np.nanmean((O[i_start:i_end, j] >= lo[i_start:i_end, j]) & (O[i_start:i_end, j] <= hi[i_start:i_end, j]))) for j in range(3)}
full = metrics(HY[i_start:i_end], O[i_start:i_end])
print('copertura', cover); print('periodo misurato (taratura)', {n: {m: round(x, 3) for m, x in d.items()} for n, d in full.items()})
ser = pd.DataFrame({'data_ora_fine_intervallo_solare': H.strftime('%Y-%m-%d %H:%M'), 'pioggia_mm': np.round(P, 2), 'ET0_mm': np.round(ET0, 3)})
for j, nm in enumerate(['15cm', '30cm', '60cm']):
    ser[f'vwc_{nm}'] = HY[:, j].round(4); ser[f'vwc_{nm}_p05'] = lo[:, j].round(4); ser[f'vwc_{nm}_p95'] = hi[:, j].round(4)
for b in use:
    for j, nm in enumerate(['15cm', '30cm', '60cm']):
        ser[f'vwc_{nm}_{b}'] = bases[b][0][:, j].round(4)
for i, nm in enumerate(['0_7cm', '7_28cm', '28_100cm']):
    ser[f'era5land_{nm}'] = np.round(E5[:, i], 3)
for j, nm in enumerate(['15cm', '30cm', '60cm']):
    ser[f'vwc_{nm}_misurato'] = np.round(O[:, j], 4)
ser.to_csv(f'{OUT}/vwc_sintetico_orario_2016_2026.csv', index=False)
json.dump(dict(base=use, coefficienti=coef, copertura=cover, taratura=full, verifica=res['m + ERA5-Land']), open(f'{OUT}/ibrido_finale.json', 'w'), indent=1)
print('scritto', ser.shape)
