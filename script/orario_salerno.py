"""Temperatura e pioggia orarie su un punto sopra Salerno (2016-2026).

Dati: Centro Funzionale Multirischi Regione Campania, estrazione sub-giornaliera
(passo 10 minuti, dati NON validati), un CSV per sensore e per anno.

Passi:
 1. unisce gli anni di ogni sensore in un unico file a 10 minuti, con controlli;
 2. aggrega a passo orario (ora indicata = fine dell'intervallo);
 3. kriging ordinario sul punto:
      - pioggia: sui pluviometri vicini, riportati alla quota del punto con la relazione
        pioggia media annua - quota della rete regionale;
      - temperatura: sui valori riportati alla quota del punto con il gradiente verticale.

Uso: python orario_salerno.py <cartella CSV> <cartella sensori repo giornaliero> <cartella output>
"""
import sys
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SRC, DAILY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

LAT0, LON0, Z0 = 40.693522, 14.760576, 245.4   # quota dal DTM HR-DTM-5m (Zenodo 10.5281/zenodo.18872933)

# sensore: (tipo, lat, lon, quota [m]) -- coordinate dall'anagrafica del Centro Funzionale,
# quote lette dal DTM HR-DTM-5m (5 m, EPSG:6875) nella cella che contiene la stazione
STATIONS = {
    "SalernoGenioCivile": ("P", 40.6780833, 14.7514444, 3.9),
    "Cologna": ("P", 40.7190, 14.7760, 107.8),
    "Pellezzano": ("P", 40.7290, 14.7560, 357.2),
    "SalernoMeteo": ("T", 40.64365, 14.83642, 11.6),
    "MercatoSSeverino": ("T", 40.77833, 14.75278, 133.0),
    "Tramonti": ("T", 40.70278, 14.64611, 406.3),
    "Pizzolano": ("T", 40.78500, 14.78639, 325.9),
}
T_LIMITS = (-20.0, 48.0)       # degC
P10_MAX = 40.0                 # mm in 10 minuti: oltre e' considerato errore strumentale
T_JUMP = 8.0                   # degC tra due letture consecutive a 10 min
SLOPE_LIMITS = (-10.0, 3.0)    # degC/km
MIN_ELEV_SPAN = 250.0          # m, minimo per stimare il gradiente dell'ora dalle stazioni


def to_xy(lat, lon):
    x = np.radians(np.asarray(lon) - LON0) * 6371.0 * np.cos(np.radians(LAT0))
    y = np.radians(np.asarray(lat) - LAT0) * 6371.0
    return x, y


# ------------------------------------------------------------------ 1. unione e controlli
def load_station(name, kind):
    files = sorted(glob.glob(f"{SRC}/{kind}_{name}_*.csv"))
    if not files:
        return None, None
    parts = []
    for f in files:
        d = pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str)
        d.columns = ["data", "ora", "valore", "unita"]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    n_raw = len(d)
    t = pd.to_datetime(d.data + " " + d.ora, format="%d-%m-%Y %H:%M:%S")
    v = pd.to_numeric(d.valore.str.replace(",", ".", regex=False), errors="coerce")
    s = pd.Series(v.to_numpy(), index=t).sort_index()
    n_dup = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="first")]
    off_grid = int((s.index.minute % 10 != 0).sum() + (s.index.second != 0).sum())
    s = s[(s.index.minute % 10 == 0) & (s.index.second == 0)]
    n_nan = int(s.isna().sum())
    if kind == "T":
        bad = (s < T_LIMITS[0]) | (s > T_LIMITS[1])
        jump = (s.diff().abs() > T_JUMP) & (s.diff(-1).abs() > T_JUMP)   # picco isolato
        bad |= jump
    else:
        bad = (s < 0) | (s > P10_MAX)
    n_bad = int(bad.sum())
    s = s.mask(bad)
    full = pd.date_range(s.index.min().floor("D"), s.index.max(), freq="10min")
    s = s.reindex(full)
    rep = dict(sensore=name, tipo="temperatura" if kind == "T" else "pioggia", file_anno=len(files),
               righe_lette=n_raw, duplicati_rimossi=n_dup, fuori_passo_10min=off_grid,
               valori_vuoti=n_nan, valori_scartati_QC=n_bad, inizio=str(s.first_valid_index()),
               fine=str(s.last_valid_index()), passi_attesi=len(full),
               completezza_pct=round(100 * s.notna().mean(), 2))
    if kind == "P":
        rep["totale_medio_annuo_mm"] = round(s.sum() / (s.notna().sum() / 52596.0), 0)
    else:
        rep["media_degC"] = round(s.mean(), 2)
    out = pd.DataFrame({"data_ora": full.strftime("%Y-%m-%d %H:%M"), "valore": s.to_numpy()})
    out.to_csv(f"{OUT}/stazione_{name}_{'temperatura_degC' if kind == 'T' else 'pioggia_mm'}_10min.csv",
               index=False)
    return s, rep


series, reports = {}, []
for name, (kind, lat, lon, z) in STATIONS.items():
    s, rep = load_station(name, kind)
    if s is None:
        print("MANCANTE:", name)
        continue
    series[name] = s
    reports.append(rep)
rep_df = pd.DataFrame(reports)
rep_df.to_csv(f"{OUT}/verifica_stazioni.csv", index=False)

# ------------------------------------------------------------------ 2. aggregazione oraria
def hourly(s, kind):
    # ora indicata = fine dell'intervallo: (h-1:00, h:00]
    g = s.resample("1h", closed="right", label="right")
    n = g.count()
    if kind == "T":
        return g.mean().where(n >= 4)
    return g.sum(min_count=1).where(n == 6)


P_names = [n for n in series if STATIONS[n][0] == "P"]
T_names = [n for n in series if STATIONS[n][0] == "T"]
hours = pd.date_range("2016-01-01 01:00", "2026-11-16 00:00", freq="1h")
PH = pd.DataFrame({n: hourly(series[n], "P") for n in P_names}).reindex(hours)
TH = pd.DataFrame({n: hourly(series[n], "T") for n in T_names}).reindex(hours)


def meta(names):
    lat = np.array([STATIONS[n][1] for n in names]); lon = np.array([STATIONS[n][2] for n in names])
    x, y = to_xy(lat, lon)
    return np.c_[x, y], np.array([STATIONS[n][3] for n in names]) / 1000.0


XP, _ = meta(P_names)
XT, ZT = meta(T_names)


# ------------------------------------------------------------------ kriging ordinario
def expo(h, nug, psill, rng):
    return nug + psill * (1.0 - np.exp(-h / rng))


def ok_weights(xy, x0, par):
    n = len(xy)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    A = np.ones((n + 1, n + 1)); A[:n, :n] = np.where(d > 0, expo(d, *par), 0.0); A[n, n] = 0.0
    rhs = np.ones(n + 1); rhs[:n] = expo(np.hypot(xy[:, 0] - x0[0], xy[:, 1] - x0[1]), *par)
    w = np.linalg.solve(A, rhs)
    return w[:n], max(float(w @ rhs), 0.0)


def krige_matrix(V, xy, x0, par, exclude=None):
    """Kriging ordinario ora per ora; i pesi sono calcolati una volta per ogni combinazione
    di stazioni disponibili. Ritorna stima, deviazione standard di kriging, n. stazioni."""
    ok = ~np.isnan(V)
    if exclude is not None:
        ok[:, exclude] = False
    est = np.full(len(V), np.nan); sd = np.full(len(V), np.nan); n = ok.sum(axis=1)
    codes = ok @ (1 << np.arange(V.shape[1]))
    for c in np.unique(codes):
        if c == 0:
            continue
        cols = np.where((c >> np.arange(V.shape[1])) & 1)[0]
        rows = codes == c
        if len(cols) == 1:
            est[rows] = V[rows, cols[0]]
            sd[rows] = np.sqrt(2 * expo(np.hypot(*(xy[cols[0]] - np.asarray(x0))), *par))
            continue
        w, var = ok_weights(xy[cols], x0, par)
        est[rows] = V[np.ix_(rows, cols)] @ w
        sd[rows] = np.sqrt(var)
    return est, sd, n


def idw_matrix(V, xy, x0, exclude=None, power=2.0):
    w = 1.0 / np.maximum(np.hypot(xy[:, 0] - x0[0], xy[:, 1] - x0[1]), 0.05) ** power
    W = np.where(np.isnan(V), 0.0, w[None, :])
    if exclude is not None:
        W[:, exclude] = 0.0
    den = W.sum(axis=1)
    return np.where(den > 0, (np.nan_to_num(V) * W).sum(axis=1) / np.where(den > 0, den, 1), np.nan)


def pair_variogram(V, xy):
    """Semivarianza media nel tempo per ogni coppia di stazioni."""
    rows = []
    for i in range(V.shape[1]):
        for j in range(i + 1, V.shape[1]):
            d = V[:, i] - V[:, j]
            ok = ~np.isnan(d)
            rows.append((np.hypot(*(xy[i] - xy[j])), 0.5 * np.mean(d[ok] ** 2), int(ok.sum())))
    return pd.DataFrame(rows, columns=["dist_km", "gamma", "n_ore"]).sort_values("dist_km")


# ------------------------------------------------------------------ 3a. pioggia
Pm = PH.to_numpy()
wet = np.nanmax(Pm, axis=1) > 0                       # ore con pioggia in almeno un pluviometro
vp = pair_variogram(Pm[wet], XP)
var_wet = np.nanmean(np.nanvar(Pm[wet], axis=0))
# variogramma esponenziale senza nugget, portata stimata dalle coppie di pluviometri (ore piovose)
try:
    (rng_p,), _ = curve_fit(lambda h, r: expo(h, 0.0, var_wet, r), vp.dist_km, vp.gamma, p0=[5.0],
                            bounds=([0.5], [100.0]))
except Exception:
    rng_p = 10.0
par_p = (0.0, var_wet, float(rng_p))
# effetto della quota sulla pioggia: relazione pioggia media annua - quota dei pluviometri
# della rete regionale entro 25 km (giornalieri 2001-2020). Ogni pluviometro viene riportato
# alla quota del punto con il rapporto MAP(z_punto)/MAP(z_stazione), poi kriging ordinario.
_info = pd.read_csv(f"{DAILY}/anagraficaSensori/reg.campania.sensori.info.csv", sep=";")
_x, _y = to_xy(_info.Latitude, _info.Longitude)
_d = pd.read_csv(f"{DAILY}/csv/Prec_campania_2000_2020.csv", sep=";")
_d = _d[_d.yy >= 2001]
_rows = []
for _sid, _z, _dist in zip(_info.SiteID.astype(str), _info.Elevation_dem, np.hypot(_x, _y)):
    if _dist <= 25 and _sid in _d and _d[_sid].notna().sum() >= 8 * 365:
        _rows.append((_z, _d[_sid].mean() * 365.25))
_rows = np.array(_rows)
MAP_B, MAP_A = np.polyfit(_rows[:, 0], _rows[:, 1], 1)
MAP_R = np.corrcoef(_rows[:, 0], _rows[:, 1])[0, 1]
ZP = np.array([STATIONS[n][3] for n in P_names])


def map_z(z):
    return MAP_A + MAP_B * np.asarray(z, dtype=float)


def rain_to(z_target):
    return Pm * (map_z(z_target) / map_z(ZP))[None, :]


p_plain, _, _ = krige_matrix(Pm, XP, (0.0, 0.0), par_p)
p_plain = np.clip(p_plain, 0, None)
p_est, p_sd, p_n = krige_matrix(rain_to(Z0), XP, (0.0, 0.0), par_p)
p_est = np.clip(p_est, 0, None)
cv_p = []
for j, nme in enumerate(P_names):
    obs = Pm[:, j]
    ek, _, _ = krige_matrix(rain_to(ZP[j]), XP, XP[j], par_p, exclude=j)
    ei, _, _ = krige_matrix(Pm, XP, XP[j], par_p, exclude=j)     # kriging senza quota
    ok = ~np.isnan(obs) & ~np.isnan(ek)
    yrs = ok.sum() / 8766.0
    w_ = ok & ((obs > 0) | (ek > 0))
    cv_p.append(dict(stazione=nme, n_ore=int(ok.sum()), ME_mm_h=(ek[ok] - obs[ok]).mean(),
                     RMSE_kriging_mm_h=np.sqrt(((ek[ok] - obs[ok]) ** 2).mean()),
                     RMSE_kriging_senza_quota_mm_h=np.sqrt(((ei[ok] - obs[ok]) ** 2).mean()),
                     RMSE_kriging_ore_piovose=np.sqrt(((ek[w_] - obs[w_]) ** 2).mean()),
                     corr_oraria=np.corrcoef(ek[ok], obs[ok])[0, 1],
                     totale_annuo_oss_mm=obs[ok].sum() / yrs, totale_annuo_stim_mm=np.clip(ek[ok], 0, None).sum() / yrs,
                     totale_annuo_stim_senza_quota_mm=np.clip(ei[ok], 0, None).sum() / yrs, quota_m=ZP[j]))

# ------------------------------------------------------------------ 3b. temperatura
Tm = TH.to_numpy()
# gradiente climatologico (mese x ora) dai dati giornalieri 2001-2020 della rete regionale:
# di notte vale il gradiente delle minime, nel primo pomeriggio quello delle massime.
info = pd.read_csv(f"{DAILY}/anagraficaSensori/reg.campania.sensori.info.csv", sep=";")
ix, iy = to_xy(info.Latitude, info.Longitude)
near = info[(np.hypot(ix, iy) <= 60) & info.idSensoreTemp.notna()]
zkm = dict(zip(near.SiteID.astype(str), near.Elevation_dem / 1000.0))


def daily_slope(name):
    d = pd.read_csv(f"{DAILY}/csv/{name}_campania_2000_2020.csv", sep=";")
    idx = pd.to_datetime(dict(year=d.yy, month=d.mm, day=d.dd))
    cols = [c for c in d.columns if c in zkm]
    V = d[cols].to_numpy(); z = np.array([zkm[c] for c in cols]); b = np.full(len(V), np.nan)
    for i in range(len(V)):
        ok = ~np.isnan(V[i])
        if ok.sum() >= 10 and np.ptp(z[ok]) > 0.4:
            b[i] = np.polyfit(z[ok], V[i, ok], 1)[0]
    s = pd.Series(np.clip(b, *SLOPE_LIMITS), index=idx)
    return s.groupby(s.index.month).median()


g_max, g_min = daily_slope("Tmax"), daily_slope("Tmin")
hh = hours.hour.to_numpy(); mm = hours.month.to_numpy()
wday = 0.5 * (1 + np.cos(2 * np.pi * (hh - 14) / 24.0))     # 1 alle 14, 0 alle 02
slope_clim = wday * g_max.reindex(mm).to_numpy() + (1 - wday) * g_min.reindex(mm).to_numpy()

# gradiente dell'ora dalle stazioni, quando coprono un dislivello sufficiente
slope = slope_clim.copy(); src = np.zeros(len(hours), dtype=int)
if len(T_names) >= 3:
    for i in range(len(Tm)):
        ok = ~np.isnan(Tm[i])
        if ok.sum() >= 3 and np.ptp(ZT[ok]) * 1000 >= MIN_ELEV_SPAN:
            slope[i] = np.clip(np.polyfit(ZT[ok], Tm[i, ok], 1)[0], *SLOPE_LIMITS); src[i] = 1

R = Tm + slope[:, None] * (Z0 / 1000.0 - ZT[None, :])       # temperature riportate alla quota del punto
vt = pair_variogram(R - np.nanmean(R, axis=1, keepdims=True), XT) if len(T_names) >= 2 else None
# variogramma dei residui dalla rete regionale giornaliera (media di Tmax e Tmin): l'effetto pepita domina
par_t = (2.4, 7.7, 250.0)
t_est, t_sd, t_n = krige_matrix(R, XT, (0.0, 0.0), par_t)
t_alt, _, _ = krige_matrix(Tm + slope_clim[:, None] * (Z0 / 1000.0 - ZT[None, :]), XT, (0.0, 0.0), par_t)
cv_t = []
for j, nme in enumerate(T_names):
    if len(T_names) < 2:
        break
    obs = Tm[:, j]
    sl = slope_clim.copy()
    if len(T_names) >= 4:                                  # gradiente ristimato senza la stazione j
        keep = [k for k in range(len(T_names)) if k != j]
        for i in np.where(src == 1)[0]:
            ok = ~np.isnan(Tm[i, keep])
            if ok.sum() >= 3 and np.ptp(ZT[keep][ok]) * 1000 >= MIN_ELEV_SPAN:
                sl[i] = np.clip(np.polyfit(ZT[keep][ok], Tm[i, keep][ok], 1)[0], *SLOPE_LIMITS)
    Rj = Tm + sl[:, None] * (ZT[j] - ZT[None, :])
    ek, sk, _ = krige_matrix(Rj, XT, XT[j], par_t, exclude=j)
    e_nolapse, _, _ = krige_matrix(Tm, XT, XT[j], par_t, exclude=j)
    ok = ~np.isnan(obs) & ~np.isnan(ek)
    err = ek[ok] - obs[ok]
    cv_t.append(dict(stazione=nme, quota_m=ZT[j] * 1000, dist_dal_punto_km=float(np.hypot(*XT[j])),
                     n_ore=int(ok.sum()), ME_degC=err.mean(), RMSE_kriging_degC=np.sqrt((err ** 2).mean()),
                     RMSE_senza_correzione_quota=np.sqrt(np.nanmean((e_nolapse[ok] - obs[ok]) ** 2)),
                     RMSE_std=np.sqrt(((err / np.maximum(sk[ok], 1e-6)) ** 2).mean())))

# ------------------------------------------------------------------ output
out = pd.DataFrame({
    "data_ora_fine_intervallo": hours.strftime("%Y-%m-%d %H:%M"),
    "temperatura_degC": np.round(t_est, 2), "temperatura_sd_kriging": np.round(t_sd, 2),
    "n_termometri": t_n, "gradiente_degC_km": np.round(slope, 2),
    "gradiente_da": np.where(src == 1, "stazioni", "climatologia"),
    "pioggia_con_quota_mm": np.round(p_est, 2), "pioggia_sd_kriging": np.round(p_sd, 2), "n_pluviometri": p_n,
    "pioggia_senza_quota_mm": np.round(p_plain, 2),
})
out.to_csv(f"{OUT}/salerno_punto_orario_2016_2026.csv", index=False)
pd.DataFrame(cv_p).round(3).to_csv(f"{OUT}/orario_pioggia_cross_validation.csv", index=False)
pd.DataFrame(cv_t).round(3).to_csv(f"{OUT}/orario_temperatura_cross_validation.csv", index=False)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(rep_df.to_string())
print("\nPIOGGIA variogramma coppie (ore piovose):\n", vp.round(3).to_string(), "\nvar:", round(var_wet, 3), "range_km:", round(par_p[2], 2))
print("pioggia-quota: MAP = %.0f + %.3f*z (n=%d, r=%.2f); fattori verso il punto:" % (MAP_A, MAP_B, len(_rows), MAP_R), dict(zip(P_names, np.round(map_z(Z0) / map_z(ZP), 3))))
print(pd.DataFrame(cv_p).round(3).to_string())
print("\nTEMPERATURA stazioni:", T_names, "\ngradienti mensili Tmax:", g_max.round(2).tolist(), "\nTmin:", g_min.round(2).tolist())
print("ore con gradiente da stazioni:", int(src.sum()), "su", len(src), "| gradiente mediano:", round(float(np.median(slope)), 2))
if vt is not None:
    print("variogramma coppie residui:\n", vt.round(3).to_string())
print(pd.DataFrame(cv_t).round(3).to_string())
dd = t_est - t_alt
print("confronto gradiente da stazioni vs climatologico: diff media", round(float(np.nanmean(dd)), 2), "| RMS", round(float(np.sqrt(np.nanmean(dd ** 2))), 2))
print("\nmancanti:", out.isna().sum()[["temperatura_degC", "pioggia_con_quota_mm"]].to_dict(), "su", len(out))
ann = out.assign(anno=hours.year).groupby("anno").agg(T=("temperatura_degC", "mean"), P=("pioggia_con_quota_mm", "sum"), P_senza_quota=("pioggia_senza_quota_mm", "sum"),
      oreT=("temperatura_degC", "count"), oreP=("pioggia_con_quota_mm", "count"), Pmax_h=("pioggia_con_quota_mm", "max"),
      Tmin=("temperatura_degC", "min"), Tmax=("temperatura_degC", "max"))
print(ann.round(1).to_string())
