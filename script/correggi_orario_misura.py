"""Riporta la pioggia misurata sul punto all'ora solare (UTC+1) dei dati del Centro Funzionale.

Il confronto evento per evento con i pluviometri a 10 minuti mostra che il registratore locale
e' avanti di un'ora (ora legale) dal 31/01/2025 al 30/10/2025 e dal 29/03/2026 in poi; tra il
30/10/2025 e il 29/03/2026 e' allineato. Nei periodi sfasati si toglie un'ora all'etichetta.

Uso: python correggi_orario_misura.py <Pioggia_unificata.xlsx> <Pioggia_unificata_ora_solare.xlsx>
"""
import sys
import numpy as np
import pandas as pd

m = pd.read_excel(sys.argv[1]); m.columns = ["t", "P"]; m.t = pd.to_datetime(m.t)
legale = (m.t < "2025-10-30 00:00") | (m.t >= "2026-03-29 03:00")
m["t_solare"] = m.t - pd.to_timedelta(np.where(legale, 60, 0), unit="min")
m["spostata_di_1h"] = legale
m = m.drop_duplicates("t_solare", keep="first").set_index("t_solare").sort_index()
m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="1h"))
pd.DataFrame({"Data_fine_ora_solare": m.index, "P_unificata_mm": m.P.values, "ora_originale": m.t.values,
              "spostata_di_1h": m.spostata_di_1h.fillna(False).astype(bool).values}).to_excel(sys.argv[2], index=False)
print("ore", len(m), "| spostate", int(m.spostata_di_1h.fillna(False).sum()))
