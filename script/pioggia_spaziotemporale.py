"""Interpolazione spazio-temporale della pioggia oraria sul punto, tarata sulla misura locale.

Idea: la pioggia oraria misurata sul punto (ora h-1..h, orologio del registratore locale) viene
stimata dai 6 pluviometri a 10 minuti usando, per ciascuno, la finestra di 60 minuti che meglio
corrisponde nel tempo (anticipo/ritardo dovuto allo spostamento delle celle e all'orologio).

Metodi confrontati, sempre tarati su una parte dei mesi e verificati sugli altri:
  A  stima attuale (kriging con orografia x fattore)
  B  kriging con orografia sulle finestre spostate di uno sfasamento comune (stimato)
  C  spazio-temporale: per ogni pluviometro lo sfasamento migliore; pesi non negativi
     (minimi quadrati), riscalati per conservare il totale
  D  come C, ma con due finestre per pluviometro (sfasamento migliore e +-30 min)

Uso: python pioggia_spaziotemporale.py <cartella file 10 min> <Pioggia_unificata.xlsx> <serie oraria attuale csv> <output>
"""
import sys
import glob
import numpy as np
import pandas as pd
from scipy.optimize import nnls

DIR10, MIS, EST, OUT = sys.argv[1:5]
DELTAS = list(range(-90, 40, 10))

m = pd.read_excel(MIS).iloc[:, :2]; m.columns = ["t", "P"]; m = m.set_index(pd.to_datetime(m.t)).P   # misura in ora solare (vedi correzione)
est = pd.read_csv(EST, parse_dates=["data_ora_fine_intervallo"]).set_index("data_ora_fine_intervallo")
G10 = {}
for f in sorted(glob.glob(f"{DIR10}/stazione_*_pioggia_mm_10min.csv")):
    n = f.split("stazione_")[1].split("_pioggia")[0]
    G10[n] = pd.read_csv(f, parse_dates=["data_ora"]).set_index("data_ora").valore
G10 = pd.DataFrame(G10)
GAUGES = list(G10.columns)
ROLL = G10.rolling(6, min_periods=6).sum()          # somma dei 60 minuti che terminano all'etichetta


def window(hours, delta):
    """Pioggia dei pluviometri nella finestra di 60 min che termina a h + delta minuti."""
    R = ROLL.reindex(hours + pd.Timedelta(minutes=delta)); R.index = hours
    return R


ALLH = est.index                                      # ore della serie completa 2016-2026
W = {d: window(ALLH, d) for d in DELTAS}

# pesi equivalenti del kriging attuale (ore con tutti i pluviometri): stima = somma w_i * P_i
full = W[0].notna().all(axis=1)
w_krig = np.linalg.lstsq(W[0][full].to_numpy(), est.pioggia_mm[full].to_numpy(), rcond=None)[0]

H = m.dropna().index.intersection(ALLH)
obs = m.reindex(H)


def predict(method, train, test_idx):
    """Ritorna la stima sulle ore test_idx tarata sulle ore train."""
    o = obs[train]
    if method == "A":
        k = o.sum() / est.pioggia_mm[train].sum()
        return est.pioggia_mm[test_idx] * k, {}
    if method == "B":
        best = max(DELTAS, key=lambda d: np.corrcoef(o, (W[d].loc[train].fillna(0).to_numpy() @ w_krig))[0, 1])
        s_tr = W[best].loc[train].fillna(0).to_numpy() @ w_krig
        k = o.sum() / s_tr.sum()
        return pd.Series(W[best].loc[test_idx].fillna(0).to_numpy() @ w_krig * k, index=test_idx), {"delta": best}
    # C / D: sfasamento migliore per pluviometro (sulla correlazione in taratura)
    cols_tr, cols_te, info = [], [], {}
    for g in GAUGES:
        best = max(DELTAS, key=lambda d: np.corrcoef(o, W[d].loc[train, g].fillna(0))[0, 1])
        info[g] = best
        ds = [best] if method == "C" else sorted({best, max(best - 30, DELTAS[0]), min(best + 30, DELTAS[-1])})
        for d in ds:
            cols_tr.append(W[d].loc[train, g].fillna(0).to_numpy()); cols_te.append(W[d].loc[test_idx, g].fillna(0).to_numpy())
    Xtr, Xte = np.column_stack(cols_tr), np.column_stack(cols_te)
    w, _ = nnls(Xtr, o.to_numpy())
    k = o.sum() / (Xtr @ w).sum()
    return pd.Series(Xte @ w * k, index=test_idx), info


def metrics(e, o):
    e = e.clip(lower=0)
    d = pd.DataFrame({"e": e, "o": o})
    dd = d.resample("1D").sum()
    s6 = d.resample("6h").sum()
    wet_o, wet_e = d.o >= 0.2, d.e >= 0.2
    # eventi separati da >= 6 ore asciutte (in misura e stima)
    wet = (d.o > 0.1) | (d.e > 0.1)
    dry = (~wet).astype(int); run = dry.groupby((dry == 0).cumsum()).cumsum()
    start = wet & ((run.shift(1) >= 6) | run.shift(1).isna())
    eid = start.astype(int).cumsum().where(wet | (run < 6))
    E = d.groupby(eid).sum(); E = E[E.o >= 10]
    return dict(corr_1h=np.corrcoef(d.e, d.o)[0, 1], rmse_1h=np.sqrt(((d.e - d.o) ** 2).mean()),
                ore_rilevate=100 * (wet_o & wet_e).sum() / wet_o.sum(), falsi_allarmi=100 * (wet_e & ~wet_o).sum() / wet_e.sum(),
                corr_6h=np.corrcoef(s6.e, s6.o)[0, 1], corr_24h=np.corrcoef(dd.e, dd.o)[0, 1],
                err_eventi_pct=100 * np.mean(np.abs(E.e - E.o)) / E.o.mean(), bias_tot_pct=100 * (d.e.sum() / d.o.sum() - 1))


months = obs.index.to_period("M")
um = sorted(months.unique())
splits = {"mesi alterni A": [p for i, p in enumerate(um) if i % 2 == 0], "mesi alterni B": [p for i, p in enumerate(um) if i % 2 == 1],
          "tarato 2025, verifica 2026": [p for p in um if p.year == 2025], "tarato 2026, verifica 2025": [p for p in um if p.year == 2026]}
rows, infos = [], []
for name, trm in splits.items():
    tr = obs.index[months.isin(trm)]; te = obs.index[~months.isin(trm)]
    for meth in "ABCD":
        e, info = predict(meth, tr, te)
        rows.append(dict(verifica=name, metodo=meth, **metrics(e, obs[te])))
        if meth in "BC":
            infos.append(dict(verifica=name, metodo=meth, **info))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
print("\nMEDIA delle 4 verifiche:")
print(R.groupby("metodo").mean(numeric_only=True).round(3).to_string())
print("\nsfasamenti scelti (minuti, negativo = finestra dei pluviometri anticipata):")
print(pd.DataFrame(infos).to_string(index=False))
R.to_csv(f"{OUT}/validazione_pioggia_spaziotemporale.csv", index=False)

# ------------------------------------------------------------------ serie finale con il metodo scelto
METHOD = sys.argv[5] if len(sys.argv) > 5 else None
if METHOD:
    e_all, info = predict(METHOD, obs.index, ALLH)
    print("\nmetodo", METHOD, "tarato su tutto il periodo:", info)
    e_all = e_all.clip(lower=0).round(2)
    mis = m.reindex(ALLH)
    out = pd.DataFrame({"data_ora_fine_intervallo": ALLH.strftime("%Y-%m-%d %H:%M"),
                        "pioggia_mm": mis.where(mis.notna(), e_all).round(2).values,
                        "fonte_pioggia": np.where(mis.notna(), "misura", "stima spazio-temporale"),
                        "pioggia_stima_mm": e_all.values, "pioggia_misurata_mm": mis.values,
                        "pioggia_stima_precedente_mm": (est.pioggia_mm * 1.075).round(2).values})
    out.to_csv(f"{OUT}/pioggia_oraria_spaziotemporale_2016_2026.csv", index=False)
    print("scritto:", len(out), "ore | mancanti stima:", int(np.isnan(e_all).sum()))
