"""Interpolazione giornaliera di temperatura e pioggia su un punto sopra Salerno.

Dati: Centro Funzionale Multirischi Regione Campania, valori giornalieri 2000-2020
(raccolti nel repository GitHub valori-climatologici-1991-2020/Campania).

Temperatura: kriging ordinario dei valori detrendizzati con la quota
             (gradiente verticale stimato giorno per giorno).
Pioggia:     IDW (potenza 2) sui pluviometri piu' vicini.

Uso: python interp_salerno.py <cartella sensori del repository> <cartella output>
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SRC, OUT = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- parametri
LAT0, LON0 = 40.693522, 14.760576
Z0 = 241.0               # quota del punto [m], EU-DEM 25 m (OpenTopoData)
Y_START, Y_END = 2001, 2020
T_RADIUS_KM = 60.0       # stazioni termometriche usate
P_RADIUS_KM = 15.0       # pluviometri usati per l'IDW
P_NMAX = 6               # numero massimo di pluviometri per giorno
IDW_POWER = 2.0
MIN_N_SLOPE = 10         # minimo stazioni per stimare il gradiente del giorno
SLOPE_LIMITS = (-10.0, 3.0)   # limiti fisici del gradiente [degC/km]
COLOC_KM = 0.3           # sensori piu' vicini di cosi' vengono mediati
RNG = np.random.default_rng(42)


def to_xy(lat, lon):
    """Proiezione equirettangolare locale centrata sul punto [km]."""
    x = np.radians(np.asarray(lon) - LON0) * 6371.0 * np.cos(np.radians(LAT0))
    y = np.radians(np.asarray(lat) - LAT0) * 6371.0
    return x, y


# ---------------------------------------------------------------- lettura
info = pd.read_csv(f"{SRC}/anagraficaSensori/reg.campania.sensori.info.csv", sep=";")
info["x"], info["y"] = to_xy(info.Latitude, info.Longitude)
info["dist"] = np.hypot(info.x, info.y)
info["sid"] = info.SiteID.astype(str)


def read(name):
    df = pd.read_csv(f"{SRC}/csv/{name}_campania_2000_2020.csv", sep=";")
    df.index = pd.to_datetime(dict(year=df.yy, month=df.mm, day=df.dd))
    df = df.drop(columns=["yy", "mm", "dd"])
    return df.loc[f"{Y_START}-01-01":f"{Y_END}-12-31"]


tmax, tmin, prec = read("Tmax"), read("Tmin"), read("Prec")
days = pd.date_range(f"{Y_START}-01-01", f"{Y_END}-12-31", freq="D")
assert tmax.index.equals(days) and prec.index.equals(days), "calendario incompleto"


def merge_colocated(meta, data):
    """Media i sensori co-locati (es. 'meteo 1' / 'meteo 2')."""
    meta = meta.sort_values("dist").reset_index(drop=True)
    used, rows, cols = set(), [], {}
    for i, r in meta.iterrows():
        if i in used:
            continue
        grp = meta[(np.hypot(meta.x - r.x, meta.y - r.y) < COLOC_KM)].index
        grp = [g for g in grp if g not in used]
        used.update(grp)
        ids = [s for s in meta.loc[grp, "sid"] if s in data.columns]
        if not ids:
            continue
        key = "+".join(ids)
        cols[key] = data[ids].mean(axis=1, skipna=True)
        rows.append(dict(key=key, name=r.SiteName, x=r.x, y=r.y, z=r.Elevation_dem,
                         dist=r.dist, lat=r.Latitude, lon=r.Longitude))
    return pd.DataFrame(rows).set_index("key"), pd.DataFrame(cols)


# ---------------------------------------------------------------- QC temperatura
bad = (tmax < -25) | (tmax > 50) | (tmin < -30) | (tmin > 40) | (tmax < tmin)
tmax, tmin = tmax.mask(bad), tmin.mask(bad)
t_meta = info[info.idSensoreTemp.notna() & (info.dist <= T_RADIUS_KM)]
meta_t, tx = merge_colocated(t_meta, tmax)
_, tn = merge_colocated(t_meta, tmin)
tn = tn[tx.columns]
X = meta_t[["x", "y"]].to_numpy()
Zkm = meta_t["z"].to_numpy() / 1000.0
D = np.hypot(X[:, None, 0] - X[None, :, 0], X[:, None, 1] - X[None, :, 1])


def daily_slopes(V):
    """Gradiente verticale giornaliero [degC/km]; NaN se non stimabile."""
    b = np.full(V.shape[0], np.nan)
    for i in range(V.shape[0]):
        ok = ~np.isnan(V[i])
        if ok.sum() >= MIN_N_SLOPE and np.ptp(Zkm[ok]) > 0.4:
            b[i] = np.polyfit(Zkm[ok], V[i, ok], 1)[0]
    return np.clip(b, *SLOPE_LIMITS)


def fill_slopes(b, index):
    """Dove il gradiente non e' stimabile usa la mediana mensile."""
    s = pd.Series(b, index=index)
    clim = s.groupby(s.index.month).median()
    return s.fillna(pd.Series(s.index.month, index=index).map(clim)).to_numpy(), clim


def detrend_qc(V, b):
    """Detrend con la quota e rimozione degli outlier (> 8 degC dalla mediana del giorno)."""
    R = V - b[:, None] * Zkm[None, :]
    med = np.nanmedian(R, axis=1, keepdims=True)
    out = np.abs(R - med) > 8.0
    return np.where(out, np.nan, R), int(out.sum())


def pooled_variogram(R, edges):
    """Variogramma sperimentale mediato su tutti i giorni (residui dalla media del giorno)."""
    Rc = R - np.nanmean(R, axis=1, keepdims=True)
    iu = np.triu_indices(R.shape[1], 1)
    d = D[iu]
    num, den = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for i in RNG.choice(R.shape[0], size=min(2000, R.shape[0]), replace=False):
        g = 0.5 * (Rc[i, iu[0]] - Rc[i, iu[1]]) ** 2
        ok = ~np.isnan(g)
        k = np.digitize(d[ok], edges) - 1
        m = (k >= 0) & (k < len(num))
        np.add.at(num, k[m], g[ok][m])
        np.add.at(den, k[m], 1)
    return 0.5 * (edges[:-1] + edges[1:]), num / den, den


def expo(h, nug, psill, rng):
    return nug + psill * (1.0 - np.exp(-h / rng))


def fit_variogram(R):
    edges = np.arange(0, 65, 5.0)
    h, g, n = pooled_variogram(R, edges)
    p, _ = curve_fit(expo, h, g, p0=[0.3, 1.0, 20.0], sigma=1 / np.sqrt(n),
                     bounds=([0, 0, 1], [10, 20, 300]))
    return p, pd.DataFrame(dict(lag_km=h, gamma=g, n_pairs=n))


def ok_predict(xy, vals, x0, par):
    """Kriging ordinario su un punto. Ritorna stima e varianza di kriging."""
    n = len(vals)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = np.where(d > 0, expo(d, *par), 0.0)
    A[n, n] = 0.0
    rhs = np.ones(n + 1)
    rhs[:n] = expo(np.hypot(xy[:, 0] - x0[0], xy[:, 1] - x0[1]), *par)
    w = np.linalg.solve(A, rhs)
    return float(w[:n] @ vals), float(max(w @ rhs, 0.0))


def krige_series(R, b, par, x0, z0_km, exclude=None):
    est, sd, nst = (np.full(len(R), np.nan) for _ in range(3))
    for i in range(len(R)):
        ok = ~np.isnan(R[i])
        if exclude is not None:
            ok[exclude] = False
        if ok.sum() < 3:
            continue
        v, var = ok_predict(X[ok], R[i, ok], x0, par)
        est[i], sd[i], nst[i] = v + b[i] * z0_km, np.sqrt(var), ok.sum()
    return est, sd, nst


def idw_series(R, b, x0, z0_km, exclude, use_trend=True):
    """Baseline: IDW (con o senza correzione di quota)."""
    est = np.full(len(R), np.nan)
    w = 1.0 / np.maximum(np.hypot(X[:, 0] - x0[0], X[:, 1] - x0[1]), 0.05) ** 2
    for i in range(len(R)):
        ok = ~np.isnan(R[i])
        ok[exclude] = False
        if ok.sum() < 3:
            continue
        v = R[i, ok] if use_trend else R[i, ok] + b[i] * Zkm[ok]
        est[i] = (w[ok] @ v) / w[ok].sum() + (b[i] * z0_km if use_trend else 0.0)
    return est


results, report = {}, []
cv_rows, vario_tabs = [], {}
for name, df in (("Tmax", tx), ("Tmin", tn)):
    V = df.to_numpy()
    b_raw = daily_slopes(V)
    b, clim = fill_slopes(b_raw, days)
    R, n_out = detrend_qc(V, b)
    par, vtab = fit_variogram(R)
    vario_tabs[name] = vtab
    est, sd, nst = krige_series(R, b, par, (0.0, 0.0), Z0 / 1000.0)
    results[name] = dict(est=est, sd=sd, n=nst, slope=b)
    report.append(dict(variabile=name, n_stazioni=len(meta_t), outlier_rimossi=n_out,
                       giorni_gradiente_stimato=int(np.isfinite(b_raw).sum()),
                       gradiente_mediano_degC_km=float(np.nanmedian(b_raw)),
                       nugget=par[0], sill=par[0] + par[1], range_pratico_km=3 * par[2]))
    # ---- cross-validation leave-one-out sulle stazioni entro 25 km
    for j in np.where(meta_t.dist.to_numpy() <= 25)[0]:
        obs = V[:, j]
        m_ = ~np.isnan(obs)
        if m_.sum() < 365:
            continue
        x0 = X[j]
        e_k, s_k, _ = krige_series(R, b, par, x0, Zkm[j], exclude=j)
        e_i = idw_series(R, b, x0, Zkm[j], j, use_trend=True)
        e_p = idw_series(R, b, x0, Zkm[j], j, use_trend=False)
        ok = m_ & ~np.isnan(e_k)
        err = e_k[ok] - obs[ok]
        cv_rows.append(dict(variabile=name, stazione=meta_t.name.iloc[j],
                            dist_dal_punto_km=meta_t.dist.iloc[j], quota_m=meta_t.z.iloc[j],
                            n_giorni=int(ok.sum()), ME=err.mean(),
                            RMSE_kriging=np.sqrt((err ** 2).mean()),
                            RMSE_std=np.sqrt(((err / np.maximum(s_k[ok], 1e-6)) ** 2).mean()),
                            RMSE_IDW_con_quota=np.sqrt(np.nanmean((e_i[ok] - obs[ok]) ** 2)),
                            RMSE_IDW_senza_quota=np.sqrt(np.nanmean((e_p[ok] - obs[ok]) ** 2))))

# ---------------------------------------------------------------- pioggia
p_meta = info[info.idSensorePrec.notna() & (info.dist <= 25)]
meta_p, pr = merge_colocated(p_meta, prec)
pr = pr.mask(pr < 0)
# QC: valori > 300 mm/giorno non confermati dai vicini (mediana dei vicini < 30 mm)
P = pr.to_numpy()
PX = meta_p[["x", "y"]].to_numpy()
med_all = np.nanmedian(P, axis=1, keepdims=True)
susp = (P > 300) & (med_all < 30)
n_susp = int(susp.sum())
P = np.where(susp, np.nan, P)


def idw_rain(x0, exclude=None, nearest_only=False):
    d = np.hypot(PX[:, 0] - x0[0], PX[:, 1] - x0[1])
    cand = np.where(d <= P_RADIUS_KM)[0]
    if exclude is not None:
        cand = cand[cand != exclude]
    cand = cand[np.argsort(d[cand])]
    est, nused, dmin = (np.full(len(P), np.nan) for _ in range(3))
    for i in range(len(P)):
        ok = cand[~np.isnan(P[i, cand])][: (1 if nearest_only else P_NMAX)]
        if len(ok) == 0:
            continue
        w = 1.0 / np.maximum(d[ok], 0.05) ** IDW_POWER
        est[i], nused[i], dmin[i] = (w @ P[i, ok]) / w.sum(), len(ok), d[ok].min()
    return est, nused, dmin


p_est, p_n, p_dmin = idw_rain((0.0, 0.0))
cvp = []
for j in np.where(meta_p.dist.to_numpy() <= 8)[0]:
    obs = P[:, j]
    if (~np.isnan(obs)).sum() < 3 * 365:
        continue
    e, _, _ = idw_rain(PX[j], exclude=j)
    e1, _, _ = idw_rain(PX[j], exclude=j, nearest_only=True)
    ok = ~np.isnan(obs) & ~np.isnan(e) & ~np.isnan(e1)
    yrs = ok.sum() / 365.25
    cvp.append(dict(stazione=meta_p.name.iloc[j], dist_dal_punto_km=meta_p.dist.iloc[j],
                    n_giorni=int(ok.sum()), ME_mm_giorno=(e[ok] - obs[ok]).mean(),
                    RMSE_IDW_mm=np.sqrt(((e[ok] - obs[ok]) ** 2).mean()),
                    RMSE_piu_vicino_mm=np.sqrt(((e1[ok] - obs[ok]) ** 2).mean()),
                    corr_giornaliera=np.corrcoef(e[ok], obs[ok])[0, 1],
                    totale_annuo_oss_mm=obs[ok].sum() / yrs,
                    totale_annuo_stim_mm=e[ok].sum() / yrs))

# ---------------------------------------------------------------- output
out = pd.DataFrame({
    "data": days.strftime("%Y-%m-%d"),
    "tmax_degC": results["Tmax"]["est"], "tmax_sd_kriging": results["Tmax"]["sd"],
    "tmin_degC": results["Tmin"]["est"], "tmin_sd_kriging": results["Tmin"]["sd"],
    "tmedia_degC": (results["Tmax"]["est"] + results["Tmin"]["est"]) / 2,
    "n_stazioni_T": results["Tmax"]["n"],
    "gradiente_tmax_degC_km": results["Tmax"]["slope"],
    "gradiente_tmin_degC_km": results["Tmin"]["slope"],
    "pioggia_mm": p_est, "n_pluviometri": p_n, "dist_pluviometro_piu_vicino_km": p_dmin,
})
out.round(2).to_csv(f"{OUT}/salerno_punto_giornaliero_2001_2020.csv", index=False)
pd.DataFrame(report).round(3).to_csv(f"{OUT}/temperatura_modello.csv", index=False)
pd.DataFrame(cv_rows).round(3).to_csv(f"{OUT}/temperatura_cross_validation.csv", index=False)
pd.DataFrame(cvp).round(3).to_csv(f"{OUT}/pioggia_cross_validation.csv", index=False)
meta_t.round(4).to_csv(f"{OUT}/stazioni_temperatura_usate.csv")
meta_p[meta_p.dist <= P_RADIUS_KM].round(4).to_csv(f"{OUT}/pluviometri_usati.csv")
for k, v in vario_tabs.items():
    v.round(4).to_csv(f"{OUT}/variogramma_{k}.csv", index=False)

pd.set_option("display.width", 250)
print(pd.DataFrame(report).round(3).to_string())
for k, v in vario_tabs.items():
    print(k, "variogramma\n", v.round(3).to_string())
print(pd.DataFrame(cv_rows).round(2).to_string())
print("pioggia: valori sospetti rimossi", n_susp)
print(pd.DataFrame(cvp).round(2).to_string())
print(out.describe().round(2).T.to_string())
print("giorni mancanti:", out.isna().sum().to_dict())
ann = out.assign(anno=days.year).groupby("anno").agg(tmedia=("tmedia_degC", "mean"), pioggia=("pioggia_mm", "sum"),
                                                     nT=("n_stazioni_T", "mean"), nP=("n_pluviometri", "mean"))
print(ann.round(1).to_string())
