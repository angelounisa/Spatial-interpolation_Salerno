"""Dati della sezione VWC della dashboard -> assets/vwc.json"""
import sys, json
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], '/home/claude/outputs/vwc']
sys.path.insert(0, '/home/claude/work/vwc')
from laio_vwc import *
OUT = '/home/claude/outputs/vwc'; REPO = '/home/claude/spatial-interpolation_salerno'
S = json.load(open(f'{OUT}/sintesi_vwc.json'))
d = pd.read_csv(f'{OUT}/vwc_sintetico_orario_2016_2026.csv', parse_dates=[0]).set_index('data_ora_fine_intervallo_solare')
va, vb = np.load(f'{OUT}/va.npy'), np.load(f'{OUT}/vb.npy')
simA = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(va), Z, TH_H, TH_S, 4)
simB = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(vb), Z, TH_H, TH_S, 4)
k = i_split - i_start
CV = np.vstack([simB[:k], simA[k:]])                         # sempre fuori taratura
r3 = lambda a: [None if (x is None or np.isnan(x)) else round(float(x), 3) for x in a]
per = d.iloc[i_start:i_end]
ore = dict(t0=per.index[0].strftime('%Y-%m-%d %H:%M'), n=len(per), pioggia=[round(float(x), 1) for x in per.pioggia_mm])
for j, c in enumerate(['15cm', '30cm', '60cm']):
    ore[c] = dict(mis=r3(per[f'vwc_{c}_misurato']), sim=r3(per[f'vwc_{c}']), lo=r3(per[f'vwc_{c}_p05']), hi=r3(per[f'vwc_{c}_p95']), cv=r3(CV[:, j]))
day = (d.index - pd.Timedelta(minutes=1)).floor('D')
g = d.groupby(day)
giorni = dict(t=[x.strftime('%Y-%m-%d') for x in g.size().index], pioggia=g.pioggia_mm.sum().round(1).tolist())
for c in ['15cm', '30cm', '60cm']:
    giorni[c] = g[f'vwc_{c}'].mean().round(3).tolist()
    giorni[c + '_mis'] = r3(g[f'vwc_{c}_misurato'].mean())
rep = S['verifica']
def mt(x): return {n: {m: round(float(v), 3) for m, v in dd.items()} for n, dd in x.items()}
ver = {'tarato 2025 → verificato 2026': mt(rep['tarato feb2025-gen2026']['verifica']),
       'tarato 2026 → verificato 2025': mt(rep['tarato feb2026-nov2026']['verifica']),
       'tarato su tutto (usato per la serie)': mt(rep['tarato tutto']['taratura'])}
prm = np.array(rep['parametri']['F'])
param = [dict(strato=n, Z=Z[j], th_h=TH_H[j], th_star=prm[j, 0], th_fc=prm[j, 1], th_s=TH_S[j], Ks=prm[j, 2], beta=prm[j, 3], c=prm[j, 4]) for j, n in enumerate(NAMES)]
# medie annue sintetiche
ann = d.groupby(d.index.year).agg(pioggia=('pioggia_mm', 'sum'), v15=('vwc_15cm', 'mean'), v30=('vwc_30cm', 'mean'), v60=('vwc_60cm', 'mean'),
                                   v15min=('vwc_15cm', lambda x: x.quantile(.05)))
J = dict(ore=ore, giorni=giorni, verifica=ver, catena=mt(S['catena']), copertura=S['copertura_banda'], parametri=param,
         poisson=S['poisson'], periodi_secchi=S['periodi_secchi'], pdf=S['pdf'], check=S['check_analitica'], quantili=S['quantili'],
         vg={c: {k2: round(float(v), 4) for k2, v in dd.items()} for c, dd in S['vg'].items()}, NY=S['NY'],
         annuale=[dict(anno=int(y), **{k2: round(float(v), 3) for k2, v in r.items()}) for y, r in ann.iterrows()])
json.dump(J, open(f'{REPO}/assets/vwc.json', 'w'), separators=(',', ':'))
import os; print('vwc.json', os.path.getsize(f'{REPO}/assets/vwc.json') / 1e6, 'MB')
print(ver); print(param)
