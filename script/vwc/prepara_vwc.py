"""Dati della sezione VWC della dashboard -> assets/vwc.json (serie finale = modello + ERA5-Land)."""
import sys, json, os
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], '/home/claude/outputs/vwc']
sys.path.insert(0, '/home/claude/work/vwc')
from laio_vwc import *
import ibrido_era5 as IB
OUT = '/home/claude/outputs/vwc'; REPO = '/home/claude/spatial-interpolation_salerno'
S = json.load(open(f'{OUT}/sintesi_vwc.json')); F = json.load(open(f'{OUT}/ibrido_finale.json'))
d = pd.read_csv(f'{OUT}/vwc_sintetico_orario_2016_2026.csv', parse_dates=[0]).set_index('data_ora_fine_intervallo_solare')
k = i_split - i_start
HAS_R = os.path.exists(f'{OUT}/rich2_A.npy') and os.path.exists(f'{OUT}/rich2_B.npy')
L = IB.cv_sims('laio')
models = {'Laio': L, 'ERA5-Land riscalato': None, 'Laio + ERA5-Land': L}
if HAS_R:
    Rr = IB.cv_sims('richards'); models.update({'Richards': Rr, 'Richards + ERA5-Land': Rr})
IB.VAR = False; res = IB.evaluate(models)
IB.VAR = True
best = F['base'][0].capitalize() + ' + ERA5-Land'
res_v = IB.evaluate({k_: v for k_, v in models.items() if '+ ERA5-Land' in k_})
cvp = lambda name: np.vstack([IB.PRED[(name, '2026→2025')][:k], IB.PRED[(name, '2025→2026')][k:]])
raw = lambda AB: np.vstack([AB[1][:k], AB[0][k:]])
CVS = {'hyb': cvp(best), 'laio': raw(L)}
IB.VAR = False; IB.evaluate({'ERA5-Land riscalato': None}); CVS['era5'] = cvp('ERA5-Land riscalato')
if HAS_R: CVS['rich'] = raw(Rr)
r3 = lambda a: [None if (x is None or not np.isfinite(x)) else round(float(x), 3) for x in a]
per = d.iloc[i_start:i_end]
ore = dict(t0=per.index[0].strftime('%Y-%m-%d %H:%M'), n=len(per), pioggia=[round(float(x), 1) for x in per.pioggia_mm], versioni=list(CVS))
for j, c in enumerate(['15cm', '30cm', '60cm']):
    ore[c] = dict(mis=r3(per[f'vwc_{c}_misurato']), sim=r3(per[f'vwc_{c}']), lo=r3(per[f'vwc_{c}_p05']), hi=r3(per[f'vwc_{c}_p95']),
                  **{kk: r3(v[:, j]) for kk, v in CVS.items()})
day = (d.index - pd.Timedelta(minutes=1)).floor('D'); g = d.groupby(day)
giorni = dict(t=[x.strftime('%Y-%m-%d') for x in g.size().index])
for c in ['15cm', '30cm', '60cm']:
    giorni[c] = g[f'vwc_{c}'].mean().round(3).tolist(); giorni[c + '_mis'] = r3(g[f'vwc_{c}_misurato'].mean())
# tabella di confronto (media delle due verifiche)
rows = []
allres = dict(res); allres.update({k_ + ' (varianza conservata)': v for k_, v in res_v.items()})
for nm, dd in allres.items():
    for dep in NAMES:
        vals = [dd[l][dep] for l in dd]
        rows.append(dict(modello=nm, profondita=dep, **{m: round(float(np.mean([v[m] for v in vals])), 3) for m in ('NSE', 'KGE', 'corr', 'RMSE', 'bias_picchi')}))
conf = pd.DataFrame(rows); conf.to_csv(f'{REPO}/dati/confronto_modelli_vwc.csv', index=False)
ver_best = {lab: {n: {m: round(float(v), 3) for m, v in x.items()} for n, x in mm.items()} for lab, mm in res_v[best].items()}
rep = S['verifica']; prm = np.array(rep['parametri']['F'])
param = [dict(strato=n, Z=Z[j], th_h=TH_H[j], th_star=prm[j, 0], th_fc=prm[j, 1], th_s=TH_S[j], Ks=prm[j, 2], beta=prm[j, 3], c=prm[j, 4]) for j, n in enumerate(NAMES)]
rich = None
if HAS_R:
    import richards_mol as R
    vF = np.load(f'{OUT}/rich2_F.npy') if os.path.exists(f'{OUT}/rich2_F.npy') else np.load(f'{OUT}/rich2_A.npy')
    p = R.unpack(vF)
    rich = [dict(orizzonte=h, Ks=float(p['ks'][R.HOR == i][0]), alpha=float(p['al'][R.HOR == i][0]), n=float(p['n'][R.HOR == i][0]),
                 thr=float(p['thr'][R.HOR == i][0]), ths=float(p['ths'][R.HOR == i][0])) for i, h in enumerate(['0–22,5 cm', '22,5–45 cm', '45–150 cm'])]
    rich = dict(orizzonti=rich, c=float(p['c']), zE=float(p['zE']), psi_w=float(10 ** p['lw']), psi_d=float(10 ** p['ld']))
J = dict(ore=ore, giorni=giorni, confronto=rows, verifica_best=ver_best, best=best, coef=F['coefficienti'], copertura=F['copertura'],
         parametri=param, richards=rich, poisson=S['poisson'], periodi_secchi=S['periodi_secchi'], pdf=S['pdf'], check=S['check_analitica'],
         quantili=S['quantili'], NY=S['NY'])
json.dump(J, open(f'{REPO}/assets/vwc.json', 'w'), separators=(',', ':'))
print('vwc.json', round(os.path.getsize(f'{REPO}/assets/vwc.json') / 1e6, 2), 'MB | Richards', HAS_R)
print(conf.to_string())
