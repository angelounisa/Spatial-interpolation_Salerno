"""Taratura della pioggia stimata sul punto con la pioggia misurata (fattore unico).

Uso: python taratura_pioggia.py <serie_oraria_stimata.csv> <Pioggia_unificata.xlsx> <output.csv>
- fattore = totale misurato / totale stimato nelle ore in comune (risultato: 1.076)
- serie finale: misura dove disponibile, altrove stima x fattore
"""
import sys
import numpy as np
import pandas as pd

EST, MIS, OUT = sys.argv[1:4]
est = pd.read_csv(EST, parse_dates=["data_ora_fine_intervallo"]).set_index("data_ora_fine_intervallo")
m = pd.read_excel(MIS); m.columns = ["t", "P"]; m = m.set_index(pd.to_datetime(m.t)).P
c = pd.concat([m.rename("mis"), est.pioggia_mm.rename("est")], axis=1).dropna()
k = c.mis.sum() / c.est.sum()
print(f"ore in comune {len(c)} | fattore {k:.3f}")
cal = (est.pioggia_mm * round(k, 3)).round(2)
mis = m.reindex(est.index)
out = pd.DataFrame({
    "data_ora_fine_intervallo": est.index.strftime("%Y-%m-%d %H:%M"),
    "temperatura_degC": est.temperatura_degC.values, "temperatura_sd_kriging": est.temperatura_sd_kriging.values,
    "n_termometri": est.n_termometri.values,
    "pioggia_mm": mis.where(mis.notna(), cal).round(2).values,
    "fonte_pioggia": np.where(mis.notna(), "misura", "stima tarata"),
    "pioggia_stima_tarata_mm": cal.values, "pioggia_misurata_mm": mis.values, "n_pluviometri": est.n_pluviometri.values})
out.to_csv(OUT, index=False)
