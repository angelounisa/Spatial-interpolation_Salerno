"""Variabili orografiche estese (TINITALY 10 m) e selezione per le medie climatiche.

Rispetto a orografia.py:
  - usa tutte le stazioni della rete regionale che cadono nel mosaico dei due fogli
    (E 950-1050 km, N 4500-4550 km, UTM 32N), non solo quelle entro 30 km;
  - aggiunge variabili legate ai meccanismi fisici:
      hv_R     altezza sul fondovalle: quota - quota minima nel raggio R (1, 2, 5 km)
               -> di notte l'aria fredda scende e ristagna in basso (fascia termica)
      rel_R    posizione relativa nel rilievo: (z - min) / (max - min) nel raggio R
      bar_mare dislivello della cresta piu' alta tra il sito e il mare piu' vicino
               -> influenza marina schermata o no
      bar_SW   ostacolo sopravvento: quota massima verso sud-ovest entro 15 km - quota del sito
               -> ombra pluviometrica / effetto stau
  - indica dove cade il punto rispetto alle stazioni (estrapolazione o no).

Uso: python orografia2.py <cartella fogli tinitaly> <repo giornaliero> <cartella output>
"""
import sys
import math
import itertools
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, signal

DTM_DIR, DAILY, OUT = sys.argv[1:4]
LAT0, LON0 = 40.693522, 14.760576


def utm32(lat, lon, a=6378137.0, f=1 / 298.257223563, k0=0.9996, lon0=9.0, FE=500000.0):
    n = f / (2 - f); A = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64)
    al = [n / 2 - 2 * n * n / 3 + 5 * n ** 3 / 16, 13 * n * n / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240]
    e = math.sqrt(f * (2 - f)); ph = math.radians(lat); la = math.radians(lon - lon0)
    t = math.sinh(math.atanh(math.sin(ph)) - e * math.atanh(e * math.sin(ph)))
    xi = math.atan2(t, math.cos(la)); eta = math.atanh(math.sin(la) / math.sqrt(1 + t * t))
    x, y = eta, xi
    for j in range(1, 4):
        x += al[j - 1] * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
        y += al[j - 1] * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
    return FE + k0 * A * x, k0 * A * y


def read_tile(name):
    with tifffile.TiffFile(f"{DTM_DIR}/{name}/{name}.tif") as t:
        p = t.pages[0]
        return p.asarray().astype(np.float32), *p.tags["ModelTiepointTag"].value[3:5]


zw, X0, Y1 = read_tile("w45095_s10")
ze, xe, _ = read_tile("w45010_s10")
ov = int(round((X0 + zw.shape[1] * 10 - xe) / 10))
Z = np.hstack([zw, ze[:, ov:]]); del zw, ze
nod = Z <= -9998
sea = nod | (Z <= 0.5)
lab, _ = ndimage.label(sea)
edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
sea = np.isin(lab, edge[edge > 0]); del lab
Z = np.where(nod, 0.0, Z).astype(np.float32); del nod
F, RES = 5, 10.0
R50 = RES * F
ny, nx = Z.shape[0] // F, Z.shape[1] // F
Z50 = Z[:ny * F, :nx * F].reshape(ny, F, nx, F).mean(axis=(1, 3))
S50 = sea[:ny * F, :nx * F].reshape(ny, F, nx, F).mean(axis=(1, 3)) > 0.5
XMAX, YMIN = X0 + nx * R50, Y1 - ny * R50


def disk_mean(a, radius_m):
    r = int(round(radius_m / R50)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx ** 2 + yy ** 2 <= r * r).astype(float); k /= k.sum()
    return signal.fftconvolve(np.pad(a, r, mode="edge"), k, mode="valid")


G = {}
for r in (1000, 2000, 5000):
    s = 2 * int(r / R50) + 1
    mn = ndimage.minimum_filter(Z50, size=s, mode="nearest"); mx = ndimage.maximum_filter(Z50, size=s, mode="nearest")
    G[f"hv_{r // 1000}km"] = Z50 - mn
    G[f"rel_{r // 1000}km"] = (Z50 - mn) / np.maximum(mx - mn, 1.0)
for r in (500, 2000, 5000):
    G[f"tpi_{r}m" if r < 1000 else f"tpi_{r // 1000}km"] = Z50 - disk_mean(Z50, r)
dist, (iy, ix) = ndimage.distance_transform_edt(~S50, return_indices=True)
G["dmare_km"] = dist * R50 / 1000.0
zs = ndimage.gaussian_filter(Z50, 1000 / R50); gy, gx = np.gradient(zs, R50)
G["espSW_1km"] = 1000 * (gx - gy) / np.sqrt(2)


def cell(e, n):
    return int((Y1 - n) // R50), int((e - X0) // R50)


def profile_max(r0, c0, r1, c1):
    m = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    rr = np.clip(np.linspace(r0, r1, m).round().astype(int), 0, ny - 1)
    cc = np.clip(np.linspace(c0, c1, m).round().astype(int), 0, nx - 1)
    return float(Z50[rr, cc].max())


def covariates(e, n):
    r, c = cell(e, n)
    rr, cc = int((Y1 - n) // RES), int((e - X0) // RES)
    out = {"z": float(Z[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2].mean())}
    for k, g in G.items():
        out[k] = float(g[r, c])
    out["bar_mare"] = max(profile_max(r, c, iy[r, c], ix[r, c]) - out["z"], 0.0)
    L = int(15000 / R50)                                    # verso SW: +riga (sud), -colonna (ovest)
    out["bar_SW"] = max(profile_max(r, c, r + L / np.sqrt(2), c - L / np.sqrt(2)) - out["z"], 0.0)
    return out


# ------------------------------------------------------------------ stazioni nel mosaico
info = pd.read_csv(f"{DAILY}/anagraficaSensori/reg.campania.sensori.info.csv", sep=";")
info["nome"] = info.SiteName.str.replace(r" meteo [12]$", " meteo", regex=True)
E, N = zip(*[utm32(a, b) for a, b in zip(info.Latitude, info.Longitude)])
info["E"], info["N"] = E, N
m = 1500
info = info[(info.E > X0 + m) & (info.E < XMAX - m) & (info.N > YMIN + m) & (info.N < Y1 - m)]
rd = lambda nm: (lambda d: d[d.yy >= 2001])(pd.read_csv(f"{DAILY}/csv/{nm}_campania_2000_2020.csv", sep=";"))
Tx, Tn, P = rd("Tmax"), rd("Tmin"), rd("Prec")
rows = []
for nome, g in info.groupby("nome"):
    ids = g.SiteID.astype(str).tolist()
    rec = dict(nome=nome, E=g.E.mean(), N=g.N.mean())
    it = [i for i in ids if i in Tx]
    if it:
        tx, tn = Tx[it].mean(axis=1), Tn[it].mean(axis=1)
        if tx.notna().sum() >= 8 * 365 and tn.notna().sum() >= 8 * 365:
            rec.update(tmax=tx.mean(), tmin=tn.mean())
    ip = [i for i in ids if i in P]
    if ip:
        p = P[ip].mean(axis=1)
        if p.notna().sum() >= 8 * 365:
            rec["pioggia"] = p.mean() * 365.25
    if len(rec) > 3:
        rec.update(covariates(rec["E"], rec["N"]))
        rows.append(rec)
st = pd.DataFrame(rows)
pe, pn = utm32(LAT0, LON0)
pt = covariates(pe, pn)
st["dist_punto_km"] = np.hypot(st.E - pe, st.N - pn) / 1000
st.to_csv(f"{OUT}/stazioni_mosaico_orografia2.csv", index=False)
pd.DataFrame([dict(nome="PUNTO", E=pe, N=pn, **pt)]).to_csv(f"{OUT}/punto_orografia2.csv", index=False)

COVS = [k for k in pt if k != "z"]
pd.set_option("display.width", 250)
print("stazioni nel mosaico: T", st.tmax.notna().sum(), "| P", st.pioggia.notna().sum())
print("\nDOVE CADE IL PUNTO rispetto alle stazioni (percentile tra le stazioni con temperatura / pioggia):")
for k in ["z"] + COVS:
    vt = st.dropna(subset=["tmax"])[k]; vp = st.dropna(subset=["pioggia"])[k]
    print(f"  {k:10s} punto {pt[k]:8.2f} | stazioni T {vt.min():8.2f}..{vt.max():8.2f} (perc {100 * (vt < pt[k]).mean():5.1f}) "
          f"| stazioni P perc {100 * (vp < pt[k]).mean():5.1f}")


def loo(df, y, cols):
    d = df.dropna(subset=[y]); X = np.c_[np.ones(len(d)), d[cols].to_numpy()] if cols else np.ones((len(d), 1))
    Y = d[y].to_numpy(); XY = d[["E", "N"]].to_numpy() / 1000; err = []
    for i in range(len(d)):
        k = np.arange(len(d)) != i
        b = np.linalg.lstsq(X[k], Y[k], rcond=None)[0]
        w = 1 / np.maximum(np.hypot(*(XY[k] - XY[i]).T), 0.1) ** 2
        err.append(X[i] @ b + w @ (Y[k] - X[k] @ b) / w.sum() - Y[i])
    return float(np.sqrt(np.mean(np.square(err)))), len(d)


def family(c):
    return c.split("_")[0]


for y in ("tmax", "tmin", "pioggia"):
    b0, n = loo(st, y, []); bz, _ = loo(st, y, ["z"])
    res = []
    for k in (1, 2):
        for combo in itertools.combinations(COVS, k):
            if len({family(c) for c in combo}) < k:
                continue
            r, _ = loo(st, y, ["z", *combo]); res.append((r, combo))
    res.sort()
    print(f"\n{y} (n={n}): solo distanza {b0:.3f} | quota {bz:.3f}")
    for r, c in res[:6]:
        print(f"   quota + {' + '.join(c):28s} {r:.3f}")


# ------------------------------------------------------------------ strati GeoTIFF per QGIS/SAGA
if len(sys.argv) > 4:
    LDIR = sys.argv[4]
    import os
    os.makedirs(LDIR, exist_ok=True)
    geokeys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32632)
    layers = {"quota_m": Z50, **G}
    for k, a in layers.items():
        tifffile.imwrite(f"{LDIR}/{k}.tif", np.asarray(a, dtype=np.float32), compression="zlib",
                         extratags=[(33550, 12, 3, (R50, R50, 0.0)),
                                    (33922, 12, 6, (0.0, 0.0, 0.0, float(X0), float(Y1), 0.0)),
                                    (34735, 3, len(geokeys), geokeys)])
    print("strati scritti:", sorted(layers))
