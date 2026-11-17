"""Variabili orografiche da TINITALY 10 m e modello delle medie climatiche.

1. unisce i fogli w45095 e w45010 (EPSG:32632, 10 m);
2. per il punto e le stazioni regionali calcola:
     z        quota (media 3x3 celle, 30 m)
     tpi_X    posizione topografica: quota - quota media nel raggio X (500 m, 1, 2, 5 km)
              (negativo = fondovalle/conca, positivo = versante alto/cresta)
     dmare    distanza dal mare [km]
     espSW_X  pendenza del rilievo lisciato (sigma X km) verso NE: >0 se il versante
              e' esposto ai flussi umidi da sud-ovest (sopravvento)
3. sceglie, con validazione lasciando fuori una stazione alla volta, le variabili che
   spiegano meglio le medie 2001-2020 di Tmax, Tmin e pioggia annua.

Uso: python orografia.py <cartella fogli tinitaly> <csv stazioni> <csv punto> <cartella output>
"""
import sys
import itertools
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, signal

DTM_DIR, ST_CSV, PT_CSV, OUT = sys.argv[1:5]


def read_tile(name):
    with tifffile.TiffFile(f"{DTM_DIR}/{name}/{name}.tif") as t:
        p = t.pages[0]
        z = p.asarray().astype(np.float32)
        x0, y1 = p.tags["ModelTiepointTag"].value[3:5]
    return z, x0, y1


zw, xw, yw = read_tile("w45095_s10")
ze, xe, ye = read_tile("w45010_s10")
assert yw == ye and zw.shape[0] == ze.shape[0]
ov = int(round((xw + zw.shape[1] * 10 - xe) / 10))          # colonne sovrapposte
Z = np.hstack([zw, ze[:, ov:]])
X0, Y1, RES = xw, yw, 10.0
nodata = Z <= -9998
sea = nodata | (Z <= 0.5)
# il mare e' l'insieme connesso di celle a quota ~0 che tocca il bordo del mosaico
lab, _ = ndimage.label(sea)
edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
sea = np.isin(lab, edge[edge > 0])
Zl = np.where(nodata, 0.0, Z).astype(np.float32)
del lab, Z, zw, ze
print("mosaico", Zl.shape, "E %.0f-%.0f N %.0f-%.0f" % (X0, X0 + Zl.shape[1] * RES, Y1 - Zl.shape[0] * RES, Y1),
      "| celle mare %.1f%%" % (100 * sea.mean()))

# griglia a 50 m per le variabili a scala di km
F = 5
Z50 = Zl[: Zl.shape[0] // F * F, : Zl.shape[1] // F * F].reshape(Zl.shape[0] // F, F, -1, F).mean(axis=(1, 3))
S50 = sea[: Zl.shape[0] // F * F, : Zl.shape[1] // F * F].reshape(Zl.shape[0] // F, F, -1, F).mean(axis=(1, 3)) > 0.5
del sea
R50 = RES * F


def disk_mean(a, radius_m):
    r = int(round(radius_m / R50))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx ** 2 + yy ** 2 <= r * r).astype(float)
    k /= k.sum()
    ap = np.pad(a, r, mode="edge")
    return signal.fftconvolve(ap, k, mode="valid")


cov_grids = {}
for r in (500, 1000, 2000, 5000):
    cov_grids[f"tpi_{r // 1000 if r >= 1000 else r}{'km' if r >= 1000 else 'm'}"] = Z50 - disk_mean(Z50, r)
cov_grids["dmare_km"] = ndimage.distance_transform_edt(~S50) * R50 / 1000.0
for s in (1, 3):
    zs = ndimage.gaussian_filter(Z50, s * 1000 / R50)
    gy, gx = np.gradient(zs, R50)            # gy: verso il basso (sud) nella griglia
    dzdE, dzdN = gx, -gy
    cov_grids[f"espSW_{s}km"] = 1000 * (dzdE + dzdN) / np.sqrt(2)   # m di salita per km verso NE


def sample(e, n):
    c = int((e - X0) // RES); r = int((Y1 - n) // RES)
    z = float(np.mean(Zl[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]))
    out = {"z": z}
    c5, r5 = int((e - X0) // R50), int((Y1 - n) // R50)
    for k, g in cov_grids.items():
        out[k] = float(g[r5, c5])
    return out


st = pd.read_csv(ST_CSV)
pt = pd.read_csv(PT_CSV)
rows = [dict(nome=r.nome, **sample(r.est_utm32n, r.nord_utm32n)) for r in st.itertuples()]
cov = pd.DataFrame(rows)
st = pd.concat([st.reset_index(drop=True), cov.drop(columns="nome")], axis=1)
ptc = sample(pt.est_utm32n[0], pt.nord_utm32n[0])
st.to_csv(f"{OUT}/stazioni_regionali_con_orografia.csv", index=False)
pd.DataFrame([dict(nome="PUNTO DI INTERESSE", **ptc)]).to_csv(f"{OUT}/punto_orografia.csv", index=False)

pd.set_option("display.width", 250)
print("\nPUNTO:", {k: round(v, 2) for k, v in ptc.items()})
show = ["nome", "z", "tpi_500m", "tpi_1km", "tpi_2km", "dmare_km", "espSW_1km", "espSW_3km", "tmin_media", "tmax_media", "pioggia_annua_mm"]
print(st[st.stazione_oraria.notna()][show].round(2).to_string(index=False))

# ------------------------------------------------------------- selezione delle variabili
CANDS = [c for c in cov.columns if c not in ("nome", "z")]


def loo_rmse(df, y, cols, idw_res=True):
    df = df.dropna(subset=[y])
    X = np.c_[np.ones(len(df)), df[["z"] + cols].to_numpy()] if cols is not None else np.ones((len(df), 1))
    Y = df[y].to_numpy(); E = df[["est_utm32n", "nord_utm32n"]].to_numpy() / 1000
    err = []
    for i in range(len(df)):
        m = np.arange(len(df)) != i
        b = np.linalg.lstsq(X[m], Y[m], rcond=None)[0]
        p = X[i] @ b
        if idw_res:
            w = 1 / np.maximum(np.hypot(*(E[m] - E[i]).T), 0.1) ** 2
            p += w @ (Y[m] - X[m] @ b) / w.sum()
        err.append(p - Y[i])
    return float(np.sqrt(np.mean(np.square(err)))), len(df)


res = {}
for y in ("tmax_media", "tmin_media", "pioggia_annua_mm"):
    base, n = loo_rmse(st, y, None)
    rz, _ = loo_rmse(st, y, [])
    maxk = 2 if y != "pioggia_annua_mm" else 3
    best = []
    for k in range(1, maxk + 1):
        for combo in itertools.combinations(CANDS, k):
            if sum(c.startswith("tpi") for c in combo) > 1 or sum(c.startswith("espSW") for c in combo) > 1:
                continue
            r, _ = loo_rmse(st, y, list(combo))
            best.append((r, combo))
    best.sort()
    res[y] = dict(n=n, solo_distanza=base, quota=rz, migliori=best[:5])
    print(f"\n{y} (n={n}): solo distanza {base:.3f} | quota {rz:.3f}")
    for r, c in best[:5]:
        print(f"   quota + {' + '.join(c):30s} {r:.3f}")
