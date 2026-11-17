"""Temperatura e pioggia orarie sul punto, con interpolazione che tiene conto dell'orografia.

Metodo (interpolazione assistita dalla climatologia):
  1. Medie di lungo periodo (2001-2020) dalla rete regionale entro 30 km, spiegate con
     variabili orografiche ricavate da TINITALY 10 m (orografia.py):
        Tmax mensile : quota + posizione topografica a 2 km + esposizione a SW (1 km)
        Tmin mensile : quota (le altre variabili non migliorano la stima)
        pioggia annua: quota + posizione topografica a 5 km + distanza dal mare
     piu' interpolazione (IDW) degli scarti del modello. Ne risulta la media del punto e,
     per confronto onesto, quella di ogni stazione oraria stimata senza la stazione stessa.
  2. Ogni ora, ogni stazione oraria viene trasferita al punto:
        temperatura: T_s(t) + [N_punto - N_s](mese, ora)
        pioggia    : P_s(t) * MAP_punto / MAP_s
     dove N e' la media climatica (per la temperatura una miscela di Tmax e Tmin che segue
     l'ora del giorno) e N_s e' la media osservata della stazione.
  3. Kriging ordinario dei valori trasferiti.

Uso: python orario_orografia.py <cartella CSV 10 min> <cartella repo giornaliero> <cartella orografia> <cartella output>
"""
import sys
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SRC, DAILY, ORO, OUT = sys.argv[1:5]

# sensore orario -> (tipo, nome nell'anagrafica regionale)
HOURLY = {
    "SalernoGenioCivile": ("P", "Salerno genio civile"),
    "Cologna": ("P", "Cologna"),
    "Pellezzano": ("P", "Pellezzano"),
    "CavaDeiTirreni": ("P", "Cava dei tirreni"),
    "Baronissi": ("P", "Baronissi"),
    "CastiglioneDelGenovesi": ("P", "Castiglione del genovesi"),
    "SalernoMeteo": ("T", "Salerno meteo"),
    "MercatoSSeverino": ("T", "Mercato s.severino"),
    "Tramonti": ("T", "Tramonti"),
    "Pizzolano": ("T", "Pizzolano"),
}
# variabili scelte con orografia2.py (validazione lasciando fuori una stazione alla volta)
COV_TX = ["z", "rel_2km", "tpi_500m"]      # 32 stazioni nel mosaico TINITALY
COV_TN = ["z", "rel_1km", "hv_5km"]        # 32 stazioni
COV_P = ["z", "tpi_5km", "dmare_km"]       # stazioni entro 30 km (sull'area vasta il modello peggiora)
Q_DYN = sys.argv[5] if len(sys.argv) > 5 else "rel_1km"   # variabile del termine orografico orario
T_LIMITS, P10_MAX, T_JUMP = (-20.0, 48.0), 40.0, 8.0

reg = pd.read_csv(f"{ORO}/stazioni_mosaico_orografia2.csv")
pt = pd.read_csv(f"{ORO}/punto_orografia2.csv").iloc[0]
reg["xk"], reg["yk"] = reg.E / 1000.0, reg.N / 1000.0
reg["pioggia_annua_mm"] = reg.pioggia
PX, PY = pt.E / 1000.0, pt.N / 1000.0

# ------------------------------------------------------------------ medie mensili regionali
info = pd.read_csv(f"{DAILY}/anagraficaSensori/reg.campania.sensori.info.csv", sep=";")


def daily(name):
    d = pd.read_csv(f"{DAILY}/csv/{name}_campania_2000_2020.csv", sep=";")
    d = d[d.yy >= 2001]
    return d


dTx, dTn = daily("Tmax"), daily("Tmin")
sid_by_name = {}
for r in reg.itertuples():
    ids = info[(info.SiteName.str.replace(r" meteo [12]$", " meteo", regex=True) == r.nome)].SiteID.astype(str).tolist()
    sid_by_name[r.nome] = [i for i in ids]


def monthly_normals(d):
    out = {}
    for nome, ids in sid_by_name.items():
        ids = [i for i in ids if i in d]
        if not ids:
            continue
        v = d[ids].mean(axis=1)
        if v.notna().sum() < 8 * 365:
            continue
        out[nome] = v.groupby(d.mm).mean()
    return pd.DataFrame(out).T            # righe stazioni, colonne mesi 1..12


NTX, NTN = monthly_normals(dTx), monthly_normals(dTn)


def fit_predict(df, y, cols, target, exclude=None, power=2.0):
    """Regressione sulle variabili orografiche + IDW degli scarti; ritorna la stima sul target."""
    d = df.dropna(subset=[y])
    if exclude is not None:
        d = d[d.nome != exclude]
    X = np.c_[np.ones(len(d)), d[cols].to_numpy()]
    b = np.linalg.lstsq(X, d[y].to_numpy(), rcond=None)[0]
    res = d[y].to_numpy() - X @ b
    w = 1 / np.maximum(np.hypot(d.xk - target["xk"], d.yk - target["yk"]), 0.1) ** power
    return float(np.r_[1, [target[c] for c in cols]] @ b + w @ res / w.sum())


def tgt(nome=None):
    if nome is None:
        return dict(xk=PX, yk=PY, **{c: pt[c] for c in set(COV_TX + COV_TN + COV_P + [Q_DYN])})
    r = reg[reg.nome == nome].iloc[0]
    return dict(xk=r.xk, yk=r.yk, **{c: r[c] for c in set(COV_TX + COV_TN + COV_P + [Q_DYN])})


def normals_T(target, exclude=None):
    tx, tn = np.zeros(12), np.zeros(12)
    for m in range(1, 13):
        dfx = reg.merge(NTX[[m]].rename(columns={m: "v"}), left_on="nome", right_index=True)
        dfn = reg.merge(NTN[[m]].rename(columns={m: "v"}), left_on="nome", right_index=True)
        tx[m - 1] = fit_predict(dfx, "v", COV_TX, target, exclude)
        tn[m - 1] = fit_predict(dfn, "v", COV_TN, target, exclude)
    return tx, tn


def normal_P(target, exclude=None):
    return fit_predict(reg[reg.dist_punto_km <= 30], "pioggia_annua_mm", COV_P, target, exclude)


# ------------------------------------------------------------------ dati orari
def load(name, kind):
    files = sorted(glob.glob(f"{SRC}/{kind}_{name}_*.csv"))
    if not files:
        return None, None
    d = pd.concat([pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str) for f in files], ignore_index=True)
    d.columns = ["data", "ora", "valore", "unita"]
    t = pd.to_datetime(d.data + " " + d.ora, format="%d-%m-%Y %H:%M:%S")
    s = pd.Series(pd.to_numeric(d.valore.str.replace(",", ".", regex=False), errors="coerce").to_numpy(), index=t).sort_index()
    n_raw, n_dup = len(s), int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="first")]
    s = s[(s.index.minute % 10 == 0) & (s.index.second == 0)]
    if kind == "T":
        bad = (s < T_LIMITS[0]) | (s > T_LIMITS[1]) | ((s.diff().abs() > T_JUMP) & (s.diff(-1).abs() > T_JUMP))
    else:
        bad = (s < 0) | (s > P10_MAX)
    n_bad, n_nan = int(bad.sum()), int(s.isna().sum())
    s = s.mask(bad)
    full = pd.date_range(s.index.min().floor("D"), s.index.max(), freq="10min")
    s = s.reindex(full)
    pd.DataFrame({"data_ora": full.strftime("%Y-%m-%d %H:%M"), "valore": s.to_numpy()}).to_csv(
        f"{OUT}/stazione_{name}_{'temperatura_degC' if kind == 'T' else 'pioggia_mm'}_10min.csv", index=False)
    rep = dict(sensore=name, tipo="temperatura" if kind == "T" else "pioggia", file_anno=len(files), righe_lette=n_raw,
               duplicati_rimossi=n_dup, valori_vuoti=n_nan, valori_scartati_QC=n_bad,
               inizio=str(s.first_valid_index()), fine=str(s.last_valid_index()),
               completezza_pct=round(100 * s.notna().mean(), 2))
    g = s.resample("1h", closed="right", label="right"); n = g.count()
    h = g.mean().where(n >= 4) if kind == "T" else g.sum(min_count=1).where(n == 6)
    return h, rep


hours = pd.date_range("2016-01-01 01:00", "2026-11-16 00:00", freq="1h")
H, reps = {}, []
for name, (kind, _) in HOURLY.items():
    h, rep = load(name, kind)
    if h is None:
        print("MANCANTE:", name)
        continue
    H[name] = h.reindex(hours)
    reps.append(rep)
pd.DataFrame(reps).to_csv(f"{OUT}/verifica_stazioni.csv", index=False)
P_names = [n for n in H if HOURLY[n][0] == "P"]
T_names = [n for n in H if HOURLY[n][0] == "T"]


def xy(names):
    return np.array([[reg[reg.nome == HOURLY[n][1]].xk.iloc[0] - PX, reg[reg.nome == HOURLY[n][1]].yk.iloc[0] - PY] for n in names])


XP, XT = xy(P_names), xy(T_names)
Pm = np.column_stack([H[n].to_numpy() for n in P_names])
Tm = np.column_stack([H[n].to_numpy() for n in T_names])


# ------------------------------------------------------------------ kriging ordinario
def expo(h, nug, psill, rng):
    return nug + psill * (1.0 - np.exp(-h / rng))


def krige(V, xy_, x0, par, exclude=None):
    ok = ~np.isnan(V)
    if exclude is not None:
        ok[:, exclude] = False
    est = np.full(len(V), np.nan); sd = np.full(len(V), np.nan); n = ok.sum(axis=1)
    codes = ok @ (1 << np.arange(V.shape[1]))
    for c in np.unique(codes):
        if c == 0:
            continue
        cols = np.where((c >> np.arange(V.shape[1])) & 1)[0]; rows = codes == c
        k = len(cols)
        d = np.hypot(*(xy_[cols][:, None, :] - xy_[cols][None, :, :]).transpose(2, 0, 1))
        A = np.ones((k + 1, k + 1)); A[:k, :k] = np.where(d > 0, expo(d, *par), 0.0); A[k, k] = 0
        rhs = np.ones(k + 1); rhs[:k] = expo(np.hypot(*(xy_[cols] - np.asarray(x0)).T), *par)
        w = np.linalg.solve(A, rhs)
        est[rows] = V[np.ix_(rows, cols)] @ w[:k]; sd[rows] = np.sqrt(max(w @ rhs, 0))
    return est, sd, n


def pair_vario(V, xy_):
    r = []
    for i in range(V.shape[1]):
        for j in range(i + 1, V.shape[1]):
            d = V[:, i] - V[:, j]; ok = ~np.isnan(d)
            r.append((np.hypot(*(xy_[i] - xy_[j])), 0.5 * np.mean(d[ok] ** 2)))
    return np.array(r)


hh = hours.hour.to_numpy(); mi = hours.month.to_numpy() - 1
wday = 0.5 * (1 + np.cos(2 * np.pi * (hh - 14) / 24.0))


def hourly_normal(tx, tn):
    return wday * tx[mi] + (1 - wday) * tn[mi]


# ------------------------------------------------------------------ medie climatiche
obsTX = {n: NTX.loc[HOURLY[n][1]].to_numpy() for n in T_names}
obsTN = {n: NTN.loc[HOURLY[n][1]].to_numpy() for n in T_names}
obsP = {n: reg[reg.nome == HOURLY[n][1]].pioggia_annua_mm.iloc[0] for n in P_names}
txP, tnP = normals_T(tgt())
mapP = normal_P(tgt())

# ------------------------------------------------------------------ temperatura
NT_st = np.column_stack([hourly_normal(obsTX[n], obsTN[n]) for n in T_names])
NT_pt = hourly_normal(txP, tnP)
A = Tm - NT_st                                                  # anomalie orarie delle stazioni
par_t = (2.4, 7.7, 250.0)
a_est, t_sd, t_n = krige(A, XT, (0, 0), par_t)
QT = np.array([reg[reg.nome == HOURLY[n][1]][Q_DYN].iloc[0] for n in T_names])


def dynamic(Aa, xy_, q, q0, x0, lam, exclude=None):
    """Ogni ora: anomalie = a + g*(q - media) (g con penalita' lam), kriging dei residui,
    poi + g*(q0 - media). Ore con meno di 3 stazioni: kriging semplice."""
    cols = [k for k in range(Aa.shape[1]) if k != exclude]
    S = Aa[:, cols]; qq = q[cols]; full = ~np.isnan(S).any(axis=1)
    qc = qq - qq.mean()
    g = np.zeros(len(Aa))
    Sc = S[full] - S[full].mean(axis=1, keepdims=True)
    g[full] = (Sc @ qc) / (qc @ qc + lam)
    R = Aa - g[:, None] * (q[None, :] - qq.mean())
    est, _, _ = krige(R, xy_, x0, par_t, exclude=exclude)
    return est + g * (q0 - qq.mean()), g


print("termine orario:", Q_DYN, "| stazioni:", dict(zip(T_names, QT.round(3))), "| punto:", round(float(pt[Q_DYN]), 3))
t_est = a_est + NT_pt
# confronto: solo correzione con la quota (gradiente climatologico, come nella versione precedente)
LAMS = [0.01, 0.05, 0.2]
cvT = []
for j, n in enumerate(T_names):
    obs = Tm[:, j]
    txj, tnj = normals_T(tgt(HOURLY[n][1]), exclude=HOURLY[n][1])     # media stimata SENZA la stazione
    est_oro, sdj, _ = krige(A, XT, XT[j], par_t, exclude=j)
    est_oro = est_oro + hourly_normal(txj, tnj)
    est_plain, _, _ = krige(Tm, XT, XT[j], par_t, exclude=j)
    Aj = Tm - np.column_stack([hourly_normal(*normals_T(tgt(HOURLY[m][1]), exclude=HOURLY[n][1])) if m != n
                               else hourly_normal(txj, tnj) for m in T_names])
    dyn = {}
    for lam in LAMS:
        e_d, _ = dynamic(Aj, XT, QT, QT[j], XT[j], lam, exclude=j)
        dyn[lam] = e_d + hourly_normal(txj, tnj)
    # solo quota: regressione sulla sola quota (Tmax e Tmin)
    sv_tx, sv_tn = COV_TX[:], COV_TN[:]
    globals()["COV_TX"], globals()["COV_TN"] = ["z"], ["z"]
    txz, tnz = normals_T(tgt(HOURLY[n][1]), exclude=HOURLY[n][1])
    zst = [normals_T(tgt(HOURLY[m][1]), exclude=HOURLY[n][1]) for m in T_names]
    globals()["COV_TX"], globals()["COV_TN"] = sv_tx, sv_tn
    Az = Tm - np.column_stack([hourly_normal(*zs) for zs in zst])
    est_z, _, _ = krige(Az, XT, XT[j], par_t, exclude=j)
    est_z = est_z + hourly_normal(txz, tnz)
    ok = ~np.isnan(obs) & ~np.isnan(est_oro) & ~np.isnan(est_plain) & ~np.isnan(est_z)
    for lam in LAMS:
        ok &= ~np.isnan(dyn[lam])
    night = ok & (wday < 0.25); day = ok & (wday > 0.75)
    f = lambda e, m: float(np.sqrt(np.mean((e[m] - obs[m]) ** 2)))
    cvT.append(dict(stazione=n, n_ore=int(ok.sum()),
                    RMSE_solo_distanza=f(est_plain, ok), RMSE_quota=f(est_z, ok), RMSE_orografia=f(est_oro, ok),
                    ME_orografia=float(np.mean(est_oro[ok] - obs[ok])),
                    RMSE_notte_quota=f(est_z, night), RMSE_notte_orografia=f(est_oro, night),
                    RMSE_giorno_quota=f(est_z, day), RMSE_giorno_orografia=f(est_oro, day),
                    **{f"RMSE_dinamico_l{lam:g}": f(dyn[lam], ok) for lam in LAMS},
                    **{f"RMSE_notte_dinamico_l{lam:g}": f(dyn[lam], night) for lam in LAMS}))

# ------------------------------------------------------------------ pioggia
wet = np.nanmax(Pm, axis=1) > 0
vp = pair_vario(Pm[wet], XP)
var_wet = float(np.nanmean(np.nanvar(Pm[wet], axis=0)))
try:
    (rng_p,), _ = curve_fit(lambda h, r: expo(h, 0.0, var_wet, r), vp[:, 0], vp[:, 1], p0=[5.0], bounds=([0.5], [100.0]))
except Exception:
    rng_p = 10.0
par_p = (0.0, var_wet, float(rng_p))
fac = np.array([mapP / obsP[n] for n in P_names])
p_est, p_sd, p_n = krige(Pm * fac[None, :], XP, (0, 0), par_p)
p_est = np.clip(p_est, 0, None)
p_plain, _, _ = krige(Pm, XP, (0, 0), par_p)
p_plain = np.clip(p_plain, 0, None)
cvP = []
for j, n in enumerate(P_names):
    obs = Pm[:, j]
    mj = normal_P(tgt(HOURLY[n][1]), exclude=HOURLY[n][1])
    e_oro, _, _ = krige(Pm * np.array([mj / obsP[m] for m in P_names])[None, :], XP, XP[j], par_p, exclude=j)
    e_pl, _, _ = krige(Pm, XP, XP[j], par_p, exclude=j)
    ok = ~np.isnan(obs) & ~np.isnan(e_oro); yrs = ok.sum() / 8766.0
    cvP.append(dict(stazione=n, n_ore=int(ok.sum()), media_regionale_2001_2020_mm=obsP[n],
                    media_stimata_senza_stazione_mm=round(mj, 0),
                    RMSE_orario_solo_distanza=float(np.sqrt(np.mean((e_pl[ok] - obs[ok]) ** 2))),
                    RMSE_orario_orografia=float(np.sqrt(np.mean((np.clip(e_oro[ok], 0, None) - obs[ok]) ** 2))),
                    totale_annuo_oss_mm=obs[ok].sum() / yrs,
                    totale_annuo_solo_distanza_mm=np.clip(e_pl[ok], 0, None).sum() / yrs,
                    totale_annuo_orografia_mm=np.clip(e_oro[ok], 0, None).sum() / yrs))

# ------------------------------------------------------------------ output
out = pd.DataFrame({
    "data_ora_fine_intervallo": hours.strftime("%Y-%m-%d %H:%M"),
    "temperatura_degC": np.round(t_est, 2), "temperatura_sd_kriging": np.round(t_sd, 2), "n_termometri": t_n,
    "temperatura_media_climatica_punto": np.round(NT_pt, 2),
    "pioggia_mm": np.round(p_est, 2), "pioggia_sd_kriging": np.round(p_sd, 2), "n_pluviometri": p_n,
    "pioggia_solo_distanza_mm": np.round(p_plain, 2),
})
out.to_csv(f"{OUT}/salerno_punto_orario_2016_2026.csv", index=False)
cvT = pd.DataFrame(cvT).round(3); cvP = pd.DataFrame(cvP).round(3)
cvT.to_csv(f"{OUT}/validazione_temperatura.csv", index=False)
cvP.to_csv(f"{OUT}/validazione_pioggia.csv", index=False)
clim = pd.DataFrame({"mese": range(1, 13), "tmax_punto": txP.round(2), "tmin_punto": tnP.round(2)})
clim.to_csv(f"{OUT}/medie_climatiche_punto_2001_2020.csv", index=False)

pd.set_option("display.width", 250)
print("stazioni orarie T:", T_names, "\nP:", P_names)
print("punto: Tmax/Tmin mensili", txP.round(1), tnP.round(1), "| pioggia annua %.0f mm" % mapP)
print("fattori pioggia:", dict(zip(P_names, fac.round(3))), "| portata variogramma %.1f km" % rng_p)
print(cvT.to_string(index=False)); print(cvP.to_string(index=False))
print("mancanti:", int(np.isnan(t_est).sum()), int(np.isnan(p_est).sum()), "su", len(out))
yr = out.assign(a=hours.year).groupby("a").agg(T=("temperatura_degC", "mean"), P=("pioggia_mm", "sum"), P_dist=("pioggia_solo_distanza_mm", "sum"))
print(yr.round(1).to_string())
