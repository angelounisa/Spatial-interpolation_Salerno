"""Controllo orologio del logger VWC: ritardo tra inizio pioggia (ora solare) e prima risposta a 15 cm."""
import pandas as pd, numpy as np
d = pd.read_pickle('/home/claude/data/vwc/raw.pkl').set_index('t')
s = pd.read_csv('/home/claude/spatial-interpolation_salerno/dati/salerno_punto_orario_2016_2026_tarato.csv',
                parse_dates=['data_ora_fine_intervallo_solare']).set_index('data_ora_fine_intervallo_solare')
P = s.pioggia_stima_tarata_mm
rows = []
wet = P[P >= 0.4]
# inizio evento: ora piovosa preceduta da >=12 ore asciutte (<0.2)
dry = P.rolling(12).max().shift(1) < 0.2
starts = P.index[(P >= 0.4) & dry]
for t0 in starts:
    tot = P[t0:t0 + pd.Timedelta(hours=12)].sum()
    if tot < 4 or t0 < d.index[0] + pd.Timedelta(hours=3) or t0 > d.index[-1]:
        continue
    onset = t0 - pd.Timedelta(hours=1)                     # inizio dell'ora piovosa
    base = d.v15[onset - pd.Timedelta(hours=2):onset].median()
    w = d.v15[onset - pd.Timedelta(hours=3):onset + pd.Timedelta(hours=12)]
    up = w[w > base + 0.004]
    if len(up) == 0:
        continue
    lag = (up.index[0] - onset).total_seconds() / 60
    rows.append(dict(inizio=onset, tot=tot, lag_min=lag, base=base))
R = pd.DataFrame(rows)
legal = lambda t: ((t >= '2025-03-30 02:00') & (t < '2025-10-26 02:00')) | ((t >= '2026-03-29 02:00') & (t < '2026-10-25 02:00'))
R['periodo'] = np.where(legal(R.inizio), 'ora legale (calendario)', 'ora solare (calendario)')
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
print(R.groupby('periodo').lag_min.describe().round(0))
R.to_csv('/home/claude/outputs/vwc_ritardi_eventi.csv', index=False)
