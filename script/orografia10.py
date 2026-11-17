"""Variabili orografiche a 10 m (risoluzione piena TINITALY) per stazioni e punto,
selezione con validazione, e strati GeoTIFF a 10 m sull'area del punto.

Per ogni sito, su una finestra di +-6 km del DTM a 10 m:
  z            quota (media 3x3 celle)
  pend_deg     pendenza (su 3x3 celle)
  espS, espSW  esposizione al sole: sin(pendenza)*cos(azimut - 180/225); >0 versante
               rivolto a sud / sud-ovest (piu' caldo di giorno)
  tpi_R        quota - quota media nel raggio R (100 m, 250 m, 500 m, 2 km)
  hv_R, rel_R  altezza sul fondovalle e posizione relativa nel raggio R (1, 2, 5 km)
Da orografia2.py restano: distanza dal mare, esposizione a SW del rilievo lisciato,
ostacoli verso il mare e verso SW.

Uso: python orografia10.py <cartella fogli> <csv stazioni orografia2> <csv punto orografia2> <cartella output>
"""
import sys
import itertools
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

DTM_DIR, ST_CSV, PT_CSV, OUT = sys.argv[1:5]
RES = 10.0


def read_tile(name):
    with tifffile.TiffFile(f"{DTM_DIR}/{name}/{name}.tif") as t:
        p = t.pages[0]
        return p.asarray().astype(np.float32), *p.tags["ModelTiepointTag"].value[3:5]


zw, X0, Y1 = read_tile("w45095_s10")
ze, xe, _ = read_tile("w45010_s10")
ov = int(round((X0 + zw.shape[1] * RES - xe) / RES))
Z = np.hstack([zw, ze[:, ov:]]); del zw, ze
Z[Z <= -9998] = 0.0
NY, NX = Z.shape


def window(e, n, half_m):
    r, c = int((Y1 - n) // RES), int((e - X0) // RES); h = int(half_m / RES)
    r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, NY), max(c - h, 0), min(c + h + 1, NX)
    return Z[r0:r1, c0:c1], r - r0, c - c0, (r0, r1, c0, c1)


def fine_layers(W):
    """Strati a 10 m su una finestra (dict nome -> array)."""
    gy, gx = np.gradient(W, RES)
    dzdE, dzdN = gx, -gy
    slope = np.arctan(np.hypot(dzdE, dzdN))
    az = np.degrees(np.arctan2(-dzdE, -dzdN)) % 360          # direzione verso cui guarda il versante
    L = {"pend_deg": np.degrees(slope),
         "espS": np.sin(slope) * np.cos(np.radians(az - 180)),
         "espSW": np.sin(slope) * np.cos(np.radians(az - 225))}
    for r in (100, 250, 500, 2000):
        L[f"tpi_{r}m" if r < 1000 else f"tpi_{r // 1000}km"] = W - ndimage.uniform_filter(W, size=2 * int(r / RES) + 1, mode="nearest")
    for r in (1000, 2000, 5000):
        s = 2 * int(r / RES) + 1
        mn = ndimage.minimum_filter(W, size=s, mode="nearest"); mx = ndimage.maximum_filter(W, size=s, mode="nearest")
        L[f"hv_{r // 1000}km"] = W - mn
        L[f"rel_{r // 1000}km"] = (W - mn) / np.maximum(mx - mn, 1.0)
    return L


def site(e, n):
    W, r, c, _ = window(e, n, 6000)
    L = fine_layers(W)
    out = {"z": float(W[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].mean())}
    for k, a in L.items():
        out[k] = float(a[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].mean())
    return out


st = pd.read_csv(ST_CSV)
pt = pd.read_csv(PT_CSV).iloc[0]
keep = ["dmare_km", "espSW_1km", "bar_mare", "bar_SW", "tpi_5km"]
rows = []
for r in st.itertuples():
    d = dict(nome=r.nome, E=r.E, N=r.N, tmax=r.tmax, tmin=r.tmin, pioggia=r.pioggia, dist_punto_km=r.dist_punto_km)
    d.update(site(r.E, r.N)); d.update({k: getattr(r, k) for k in keep})
    rows.append(d)
S = pd.DataFrame(rows)
P = dict(nome="PUNTO", E=pt.E, N=pt.N, **site(pt.E, pt.N), **{k: pt[k] for k in keep})
S.to_csv(f"{OUT}/stazioni_orografia_10m.csv", index=False)
pd.DataFrame([P]).to_csv(f"{OUT}/punto_orografia_10m.csv", index=False)

COVS = [k for k in P if k not in ("nome", "E", "N", "z")]
pd.set_option("display.width", 250)
print("PUNTO a 10 m:", {k: round(v, 2) for k, v in P.items() if k not in ("nome", "E", "N")})
print("\nposizione del punto tra le stazioni (percentile, stazioni con temperatura):")
for k in ["z"] + COVS:
    v = S.dropna(subset=["tmax"])[k]
    print(f"  {k:10s} punto {P[k]:8.3f} | stazioni {v.min():8.3f}..{v.max():8.3f} | perc {100 * (v < P[k]).mean():5.1f}")
hourly = ["Salerno meteo", "Mercato s.severino", "Tramonti", "Pizzolano"]
print(S[S.nome.isin(hourly)][["nome", "z", "pend_deg", "espS", "espSW", "tpi_100m", "tpi_250m", "rel_1km"]].round(3).to_string(index=False))


def loo(df, y, cols):
    d = df.dropna(subset=[y]); X = np.c_[np.ones(len(d)), d[cols].to_numpy()] if cols else np.ones((len(d), 1))
    Y = d[y].to_numpy(); XY = d[["E", "N"]].to_numpy() / 1000; err = []
    for i in range(len(d)):
        k = np.arange(len(d)) != i
        b = np.linalg.lstsq(X[k], Y[k], rcond=None)[0]
        w = 1 / np.maximum(np.hypot(*(XY[k] - XY[i]).T), 0.1) ** 2
        err.append(X[i] @ b + w @ (Y[k] - X[k] @ b) / w.sum() - Y[i])
    return float(np.sqrt(np.mean(np.square(err)))), len(d)


fam = lambda c: {"espS": "sole", "espSW": "sole", "pend": "pend"}.get(c, c.split("_")[0])
best = {}
for y, sub in (("tmax", S), ("tmin", S), ("pioggia", S[S.dist_punto_km <= 30])):
    b0, n = loo(sub, y, []); bz, _ = loo(sub, y, ["z"])
    res = []
    for k in (1, 2):
        for combo in itertools.combinations(COVS, k):
            if len({fam(c) for c in combo}) < k:
                continue
            res.append((loo(sub, y, ["z", *combo])[0], combo))
    res.sort(); best[y] = res[0]
    print(f"\n{y} (n={n}): solo distanza {b0:.3f} | quota {bz:.3f}")
    for r, c in res[:6]:
        print(f"   quota + {' + '.join(c):26s} {r:.3f}")

# ------------------------------------------------------------------ strati a 10 m sull'area del punto
if len(sys.argv) > 5:
    import os
    LDIR = sys.argv[5]; os.makedirs(LDIR, exist_ok=True)
    HALF = 11000
    W, r, c, (r0, r1, c0, c1) = window(pt.E, pt.N, HALF + 6000)
    L = {"quota_m": W, **fine_layers(W)}
    m = int(6000 / RES)
    rr0, cc0 = max(r - int(HALF / RES), 0), max(c - int(HALF / RES), 0)
    rr1, cc1 = r + int(HALF / RES) + 1, c + int(HALF / RES) + 1
    x_ul, y_ul = X0 + (c0 + cc0) * RES, Y1 - (r0 + rr0) * RES
    geokeys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32632)
    for k, a in L.items():
        tifffile.imwrite(f"{LDIR}/{k}.tif", np.asarray(a[rr0:rr1, cc0:cc1], dtype=np.float32),
                         compression="zlib",
                         extratags=[(33550, 12, 3, (RES, RES, 0.0)), (33922, 12, 6, (0.0, 0.0, 0.0, float(x_ul), float(y_ul), 0.0)),
                                    (34735, 3, len(geokeys), geokeys)])
    print("\nstrati 10 m:", sorted(L), "| griglia", (rr1 - rr0, cc1 - cc0), "| angolo NO", (x_ul, y_ul))
