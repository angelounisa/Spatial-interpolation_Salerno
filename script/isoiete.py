"""Isoiete nel tempo: campi di pioggia su griglia dai 6 pluviometri, con lo stesso metodo della serie
sul punto (kriging ordinario dei pluviometri riportati alla media annua del punto di griglia con il
modello orografico, x fattore di taratura 1,072).

Prodotti:
  - assets/isoiete.json                 fotogrammi per la dashboard (mensili ed eventi orari)
  - dati/isoiete_mensili_2016_2026.geojson
  - dati/pioggia_mensile_griglia_200m.tif   un livello per mese, EPSG:32632
  - dati/isoiete_mensili_2016_2026.gif / .mp4  animazione con rilievo ombreggiato

Uso: python isoiete.py
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LightSource
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

R = "/home/claude/spatial-interpolation_salerno"
ORO = "/home/claude/outputs/orografia2"
L50 = f"{ORO}/strati_50m"
DIR10 = "/home/claude/outputs/orografia"
K = 1.072
RANGE_KM = 13.2
RES = 200.0
E0, E1, N0, N1 = 977000.0, 998000.0, 4513000.0, 4532000.0
MAXD_KM = 7.0                                   # oltre questa distanza dal pluviometro piu' vicino: non mostrato
to_ll = Transformer.from_crs(32632, 4326, always_xy=True)
to_utm = Transformer.from_crs(4326, 32632, always_xy=True)

gauges = {"SalernoGenioCivile": "Salerno genio civile", "Cologna": "Cologna", "Pellezzano": "Pellezzano",
          "CavaDeiTirreni": "Cava dei tirreni", "Baronissi": "Baronissi", "CastiglioneDelGenovesi": "Castiglione del genovesi"}
NOMI = {"SalernoGenioCivile": "Salerno Genio Civile", "Cologna": "Cologna", "Pellezzano": "Pellezzano",
        "CavaDeiTirreni": "Cava dei Tirreni", "Baronissi": "Baronissi", "CastiglioneDelGenovesi": "Castiglione del Genovesi"}
reg = pd.read_csv(f"{ORO}/stazioni_mosaico_orografia2.csv")
pt = pd.read_csv(f"{ORO}/punto_orografia2.csv").iloc[0]


def layer(name):
    with tifffile.TiffFile(f"{L50}/{name}.tif") as t:
        p = t.pages[0]
        return p.asarray(), p.tags["ModelTiepointTag"].value[3], p.tags["ModelTiepointTag"].value[4]


# ------------------------------------------------------------------ griglia
xs = np.arange(E0 + RES / 2, E1, RES); ys = np.arange(N1 - RES / 2, N0, -RES)
XX, YY = np.meshgrid(xs, ys)
cov = {}
for nm in ("quota_m", "tpi_5km", "dmare_km"):
    a, x0, y1 = layer(nm)
    c = ((XX - x0) // 50).astype(int); r = ((y1 - YY) // 50).astype(int)
    cov[nm] = a[r, c]
mare = cov["dmare_km"] <= 0.0

# media annua orografica su griglia (stesso modello della serie: quota + TPI 5 km + distanza mare, stazioni entro 30 km)
d = reg[(reg.dist_punto_km <= 30) & reg.pioggia.notna()]
X = np.c_[np.ones(len(d)), d.z, d.tpi_5km, d.dmare_km]
b = np.linalg.lstsq(X, d.pioggia.to_numpy(), rcond=None)[0]
res = d.pioggia.to_numpy() - X @ b
Xg = np.stack([np.ones(XX.size), cov["quota_m"].ravel(), cov["tpi_5km"].ravel(), cov["dmare_km"].ravel()], axis=1)
dist = np.hypot((XX.ravel()[:, None] - d.E.to_numpy()[None, :]) / 1000, (YY.ravel()[:, None] - d.N.to_numpy()[None, :]) / 1000)
w = 1 / np.maximum(dist, 0.1) ** 2
MAPg = (Xg @ b + (w @ res) / w.sum(axis=1)).reshape(XX.shape)

gE = np.array([reg[reg.nome == gauges[g]].E.iloc[0] for g in gauges]); gN = np.array([reg[reg.nome == gauges[g]].N.iloc[0] for g in gauges])
MAPs = np.array([reg[reg.nome == gauges[g]].pioggia.iloc[0] for g in gauges])
dmin = np.min(np.hypot((XX[..., None] - gE) / 1000, (YY[..., None] - gN) / 1000), axis=-1)
show = (~mare) & (dmin <= MAXD_KM)


def expo(h, rng=RANGE_KM):
    return 1.0 - np.exp(-h / rng)


def ok_weights(idx):
    """Pesi di kriging ordinario (senza pepita: dipendono solo dalla portata) per i pluviometri idx."""
    n = len(idx); P = np.c_[gE[idx], gN[idx]] / 1000
    D = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1))
    A = np.ones((n + 1, n + 1)); A[:n, :n] = expo(D); A[n, n] = 0
    rhs = np.ones((n + 1, XX.size)); rhs[:n] = expo(np.hypot(XX.ravel()[None, :] / 1000 - P[:, :1], YY.ravel()[None, :] / 1000 - P[:, 1:]))
    return np.linalg.solve(A, rhs)[:n].T          # (celle, n)


W_ALL = ok_weights(np.arange(len(gauges)))


def field(vals):
    """Campo di pioggia (mm) da valori dei pluviometri (NaN = mancante)."""
    ok = ~np.isnan(vals)
    if ok.sum() == 0:
        return np.full(XX.shape, np.nan)
    Wk = W_ALL if ok.all() else ok_weights(np.where(ok)[0])
    v = (vals[ok] / MAPs[ok])
    return np.clip(K * MAPg * (Wk @ v).reshape(XX.shape), 0, None)


# ------------------------------------------------------------------ dati orari dei pluviometri (ora solare)
H = {}
for g in gauges:
    s = pd.read_csv(f"{DIR10}/stazione_{g}_pioggia_mm_10min.csv", parse_dates=["data_ora"]).set_index("data_ora").valore
    grp = s.resample("1h", closed="right", label="right"); H[g] = grp.sum(min_count=1).where(grp.count() == 6)
H = pd.DataFrame(H).loc["2016-01-01 01:00":"2026-11-16 00:00"]
Hm = H.copy(); Hm.index = Hm.index - pd.Timedelta(minutes=1)
ore_mese = pd.Series(1, index=Hm.index).resample("MS").sum()
M = Hm.resample("MS").sum(min_count=1).where(Hm.resample("MS").count().ge(0.85 * ore_mese, axis=0))   # mese valido con >=85% delle ore
serie = pd.read_csv(f"{R}/dati/salerno_punto_orario_2016_2026_tarato.csv", parse_dates=["data_ora_fine_intervallo_solare"]).set_index("data_ora_fine_intervallo_solare")

# controllo: il campo nella cella del punto riproduce la serie (stima tarata)
pe, pn = pt.E, pt.N
ci, cj = int((N1 - pn) // RES), int((pe - E0) // RES)

# ------------------------------------------------------------------ contorni -> poligoni lon/lat
LEV_M = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400, 450, 500, 600, 700, 900]
LEV_H = [0.2, 1, 2, 5, 10, 15, 20, 30, 50]
_B = ["#eef5fd", "#cfe2f8", "#a9cbf1", "#7fb1e8", "#5596dd", "#3a7fd0", "#2a6abb", "#1f569f", "#164383", "#0e3168", "#08234e"]
_cm = LinearSegmentedColormap.from_list("b", _B, N=256)
BLUES = [matplotlib.colors.to_hex(_cm(i / (len(LEV_M) - 2))) for i in range(len(LEV_M) - 1)]   # un colore per fascia


def to_lonlat_ring(ring):
    lon, lat = to_ll.transform(ring[:, 0], ring[:, 1])
    return np.c_[np.round(lon, 5), np.round(lat, 5)]


def contours(F, levels):
    Fm = np.where(show, F, np.nan)
    fig = plt.figure(); ax = fig.add_subplot()
    cs = ax.contourf(XX, YY, Fm, levels=levels + [1e9], extend="neither")
    out = []
    for k, path in enumerate(cs.get_paths()):
        polys = []
        for poly in path.to_polygons():
            if len(poly) >= 4:
                polys.append(poly)
        if not polys:
            continue
        # anelli: esterni e buchi (orientamento); costruzione robusta con differenza simmetrica
        geom = None
        for poly in polys:
            p = Polygon(poly).buffer(0)
            geom = p if geom is None else geom.symmetric_difference(p)
        if geom is None or geom.is_empty:
            continue
        geom = geom.simplify(RES * 0.2, preserve_topology=True)
        parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for pg in parts:
            if pg.area < (RES * RES * 2):
                continue
            rings = [to_lonlat_ring(np.asarray(pg.exterior.coords)).tolist()] + [to_lonlat_ring(np.asarray(h.coords)).tolist() for h in pg.interiors]
            out.append([k, rings])
    plt.close(fig)
    return out


def gvals(vals):
    return {NOMI[g]: (None if np.isnan(v) else round(float(v), 1)) for g, v in zip(gauges, vals)}


frames_m, gj_feats, check = [], [], []
GRID = []
for t, row in M.iterrows():
    vals = row.to_numpy(dtype=float)
    F = field(vals); GRID.append(np.where(show, F, np.nan).astype(np.float32))
    p_series = serie.pioggia_stima_tarata_mm[(serie.index > t) & (serie.index <= t + pd.offsets.MonthBegin(1))].sum()
    check.append((F[ci, cj], p_series))
    feats = contours(F, LEV_M)
    frames_m.append(dict(t=t.strftime("%Y-%m"), f=feats, g=gvals(vals), p=round(float(F[ci, cj]), 0),
                         max=round(float(np.nanmax(np.where(show, F, np.nan))), 0)))
    for k, rings in feats:
        gj_feats.append(dict(type="Feature", properties=dict(mese=t.strftime("%Y-%m"), mm_da=LEV_M[k], mm_a=(LEV_M[k + 1] if k + 1 < len(LEV_M) else None)),
                             geometry=dict(type="Polygon", coordinates=rings)))
check = np.array(check); check = check[~np.isnan(check).any(axis=1)]
print("controllo cella del punto vs serie (mensile): corr %.4f, rapporto medio %.3f, n %d" % (np.corrcoef(check[:, 0], check[:, 1])[0, 1], check[:, 0].sum() / check[:, 1].sum(), len(check)))

# ------------------------------------------------------------------ eventi orari principali (sul punto)
P = serie.pioggia_stima_tarata_mm
wet = P > 0.1; dry = (~wet).astype(int); run = dry.groupby((dry == 0).cumsum()).cumsum()
start = wet & ((run.shift(1) >= 6) | run.shift(1).isna()); eid = start.astype(int).cumsum().where(wet | (run < 6))
ev = P.groupby(eid).agg(["sum", lambda s: s.index.min(), lambda s: s.index.max()]); ev.columns = ["mm", "da", "a"]
ev = ev.sort_values("mm", ascending=False).head(10).sort_values("da")
eventi = []
for _, e in ev.iterrows():
    hrs = pd.date_range(e.da - pd.Timedelta(hours=2), min(e.a + pd.Timedelta(hours=2), e.da + pd.Timedelta(hours=60)), freq="1h")
    fr = []
    for h in hrs:
        vals = H.reindex([h]).to_numpy(dtype=float)[0]
        F = field(vals)
        fr.append(dict(t=h.strftime("%Y-%m-%d %H:%M"), f=contours(F, LEV_H), g=gvals(vals), p=round(float(F[ci, cj]), 1)))
    eventi.append(dict(nome=f"{e.da:%d/%m/%Y} · {e.mm:.0f} mm sul punto", frames=fr))
    print("evento", e.da, round(e.mm, 1), len(fr), "ore")

lat_g = to_ll.transform(gE, gN)
col_m = [matplotlib.colors.to_hex(_cm(i / (len(LEV_M) - 1))) for i in range(len(LEV_M))]
col_h = [matplotlib.colors.to_hex(_cm(0.08 + 0.92 * i / (len(LEV_H) - 1))) for i in range(len(LEV_H))]
out = dict(lev_m=LEV_M, lev_h=LEV_H, colori_m=col_m, colori_h=col_h, mensili=frames_m, eventi=eventi,
           pluviometri=[dict(nome=NOMI[g], lon=round(lo, 5), lat=round(la, 5)) for g, lo, la in zip(gauges, *lat_g)],
           punto=dict(lon=14.760576, lat=40.693522), dominio_km=MAXD_KM)
json.dump(out, open(f"{R}/assets/isoiete.json", "w"), separators=(",", ":"))
json.dump(dict(type="FeatureCollection", name="isoiete_mensili_2016_2026", crs=dict(type="name", properties=dict(name="EPSG:4326")), features=gj_feats),
          open(f"{R}/dati/isoiete_mensili_2016_2026.geojson", "w"), separators=(",", ":"))
GRID = np.stack(GRID)
gk = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32632)
tifffile.imwrite(f"{R}/dati/pioggia_mensile_griglia_200m.tif", GRID, compression="zlib", planarconfig="separate",
                 description=";".join(f["t"] for f in frames_m),
                 extratags=[(33550, 12, 3, (RES, RES, 0.0)), (33922, 12, 6, (0.0, 0.0, 0.0, E0, N1, 0.0)), (34735, 3, len(gk), gk), (42113, 2, 4, "nan")])
print("file:", {f: round(os.path.getsize(f"{R}/{f}") / 1e6, 2) for f in ["assets/isoiete.json", "dati/isoiete_mensili_2016_2026.geojson", "dati/pioggia_mensile_griglia_200m.tif"]})

# ------------------------------------------------------------------ animazione (GIF e MP4) con rilievo ombreggiato
Z10 = []
for nm in ("w45095_s10",):
    with tifffile.TiffFile(f"/home/claude/data/dtm_tinitaly/{nm}/{nm}.tif") as t:
        p = t.pages[0]; z = p.asarray().astype(np.float32); zx0, zy1 = p.tags["ModelTiepointTag"].value[3:5]
c0, c1 = int((E0 - zx0) // 10), int((E1 - zx0) // 10); r0, r1 = int((zy1 - N1) // 10), int((zy1 - N0) // 10)
zc = z[r0:r1, c0:c1]; zc[zc <= -9998] = 0
hs = LightSource(azdeg=315, altdeg=40).hillshade(zc, vert_exag=2, dx=10, dy=10)
cmap = LinearSegmentedColormap.from_list("b", BLUES, N=len(BLUES)); norm = BoundaryNorm(LEV_M, cmap.N)
os.makedirs("/tmp/claude-0/-home-claude/a124db30-5f26-5c32-b83d-d31447d5353d/scratchpad/frames", exist_ok=True)
fdir = "/tmp/claude-0/-home-claude/a124db30-5f26-5c32-b83d-d31447d5353d/scratchpad/frames"
for f in glob.glob(fdir + "/*.png"):
    os.remove(f)
ext = [E0 / 1000, E1 / 1000, N0 / 1000, N1 / 1000]
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
for i, (fr, G) in enumerate(zip(frames_m, GRID)):
    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=110)
    ax.imshow(hs, cmap="gray", extent=ext, vmin=0, vmax=1, alpha=0.9)
    ax.imshow(np.where(zc <= 0.5, 1, np.nan), cmap=LinearSegmentedColormap.from_list("m", ["#cfe0ee", "#cfe0ee"]), extent=ext)
    cf = ax.contourf(XX / 1000, YY / 1000, G, levels=LEV_M, cmap=cmap, norm=norm, alpha=0.55)
    cl = ax.contour(XX / 1000, YY / 1000, G, levels=LEV_M[1:], colors="#08234e", linewidths=0.5, alpha=0.75)
    ax.clabel(cl, fmt="%d", fontsize=7)
    ax.scatter(gE / 1000, gN / 1000, s=28, c="#1baf7a", edgecolors="white", linewidths=0.8, zorder=5)
    for g, x, y in zip(gauges, gE, gN):
        v = fr["g"][NOMI[g]]
        ax.annotate(f"{NOMI[g]}\n{'–' if v is None else f'{v:.0f} mm'}", (x / 1000, y / 1000), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color="#111")
    ax.scatter([pe / 1000], [pn / 1000], s=60, c="#eb6834", edgecolors="white", linewidths=1.2, zorder=6)
    ax.annotate(f"punto\n{fr['p']:.0f} mm", (pe / 1000, pn / 1000), xytext=(5, -16), textcoords="offset points", fontsize=7.5, color="#111", weight="bold")
    yy, mm = fr["t"].split("-")
    ax.set_title(f"Pioggia mensile · {MESI[int(mm) - 1]} {yy}", fontsize=11, loc="left")
    ax.set_xlabel("Est UTM 32N (km)", fontsize=8); ax.set_ylabel("Nord UTM 32N (km)", fontsize=8); ax.tick_params(labelsize=7)
    cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02); cb.set_label("mm/mese", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.text(0.01, 0.01, "Kriging di 6 pluviometri con modello orografico (TINITALY 10 m). Mostrato entro 7 km dai pluviometri.", transform=ax.transAxes, fontsize=6, color="#333")
    fig.tight_layout(); fig.savefig(f"{fdir}/f{i:04d}.png"); plt.close(fig)
os.system(f"ffmpeg -y -loglevel error -framerate 3 -i {fdir}/f%04d.png -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {R}/dati/isoiete_mensili_2016_2026.mp4")
os.system(f"ffmpeg -y -loglevel error -framerate 3 -i {fdir}/f%04d.png -vf 'fps=3,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer' {R}/dati/isoiete_mensili_2016_2026.gif")
print("animazioni:", {f: round(os.path.getsize(f"{R}/dati/{f}") / 1e6, 2) for f in ["isoiete_mensili_2016_2026.mp4", "isoiete_mensili_2016_2026.gif"]})
