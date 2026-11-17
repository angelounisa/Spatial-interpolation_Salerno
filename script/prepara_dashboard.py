"""Prepara i dati riassuntivi (JSON) per la dashboard."""
import json, numpy as np, pandas as pd
R = "/home/claude/spatial-interpolation_salerno"
f = pd.read_csv(f"{R}/dati/salerno_punto_orario_2016_2026_tarato.csv", parse_dates=["data_ora_fine_intervallo_solare"]).set_index("data_ora_fine_intervallo_solare")
est = pd.read_csv("/home/claude/outputs/orografia/salerno_punto_orario_2016_2026.csv", parse_dates=["data_ora_fine_intervallo"]).set_index("data_ora_fine_intervallo")
r = lambda x, n=2: None if pd.isna(x) else round(float(x), n)
# giornaliero (giorno = ore 01..24 di quel giorno -> uso etichetta fine intervallo - 1 minuto)
g = f.copy(); g.index = g.index - pd.Timedelta(minutes=1)
day = pd.DataFrame({"tmed": g.temperatura_degC.resample("1D").mean(), "tmin": g.temperatura_degC.resample("1D").min(),
                    "tmax": g.temperatura_degC.resample("1D").max(), "p": g.pioggia_mm.resample("1D").sum(min_count=1),
                    "mis": (g.fonte_pioggia == "misura").resample("1D").mean()})
day = day[day.index < "2026-11-16"]
daily = [[d.strftime("%Y-%m-%d"), r(x.tmed, 1), r(x.tmin, 1), r(x.tmax, 1), r(x.p, 1), int(x.mis > 0.5)] for d, x in day.iterrows()]
yr = day.groupby(day.index.year).agg(tmed=("tmed", "mean"), tmin=("tmin", "mean"), tmax=("tmax", "mean"), p=("p", "sum"), n=("p", "count"))
annual = [dict(anno=int(a), tmed=r(x.tmed, 2), tmin_media=r(x.tmin, 2), tmax_media=r(x.tmax, 2), pioggia=r(x.p, 0), giorni=int(x.n)) for a, x in yr.iterrows()]
c = day[(day.index.year >= 2016) & (day.index.year <= 2025)]
mon = c.groupby(c.index.month).agg(tmed=("tmed", "mean"), tmin=("tmin", "mean"), tmax=("tmax", "mean"))
pm = c.p.resample("MS").sum(); pm = pm.groupby(pm.index.month).mean()
clim = [dict(mese=int(m), tmed=r(x.tmed, 1), tmin=r(x.tmin, 1), tmax=r(x.tmax, 1), pioggia=r(pm[m], 0)) for m, x in mon.iterrows()]
# validazione pioggia sul punto
k = 1.072
v = pd.concat([f.pioggia_misurata_mm.rename("mis"), est.pioggia_mm.rename("stima")], axis=1).dropna()
v["tarata"] = v.stima * k
vh = v.copy(); vh.index = vh.index - pd.Timedelta(minutes=1)
vd = vh.resample("1D").sum(min_count=20).dropna()
vm = vh.resample("MS").sum()
def met(a, b):
    return dict(corr=r(np.corrcoef(a, b)[0, 1], 3), rmse=r(np.sqrt(np.mean((a - b) ** 2)), 2))
valp = dict(ore=len(v), inizio=str(v.index.min()), fine=str(v.index.max()), fattore=k,
            totale=dict(misura=r(v.mis.sum(), 0), stima=r(v.stima.sum(), 0), tarata=r(v.tarata.sum(), 0)),
            orario=dict(stima=met(v.stima, v.mis), tarata=met(v.tarata, v.mis)),
            giornaliero=dict(stima=met(vd.stima, vd.mis), tarata=met(vd.tarata, vd.mis)),
            giorni=[[d.strftime("%Y-%m-%d"), r(x.mis, 1), r(x.tarata, 1)] for d, x in vd.iterrows()],
            mesi=[[d.strftime("%Y-%m"), r(x.mis, 0), r(x.stima, 0), r(x.tarata, 0)] for d, x in vm.iterrows() if d >= pd.Timestamp("2025-02-01")],
            # medie di 4 verifiche su periodi non usati (pioggia_spaziotemporale.py e prova della soglia)
            verifiche_split=[dict(metodo="Kriging con orografia x fattore", corr=0.925, rmse=0.382, trovate=84.8, falsi=16.6, bias=0.0),
                             dict(metodo="Spazio-temporale, sfasamento comune", corr=0.919, rmse=0.394, trovate=84.9, falsi=16.6, bias=0.2),
                             dict(metodo="Spazio-temporale, sfasamento per pluviometro", corr=0.917, rmse=0.401, trovate=85.5, falsi=18.3, bias=0.1),
                             dict(metodo="Kriging con soglia 0,3 mm/h", corr=0.924, rmse=0.383, trovate=78.5, falsi=10.9, bias=0.0)],
            prima_dopo=[dict(ind="Correlazione oraria", prima=0.73, dopo=0.93), dict(ind="Errore orario (mm)", prima=0.71, dopo=0.38),
                        dict(ind="Ore piovose trovate (%)", prima=74, dopo=85), dict(ind="Falsi allarmi orari (%)", prima=27, dopo=17),
                        dict(ind="Correlazione su 6 ore", prima=0.94, dopo=0.95), dict(ind="Correlazione giornaliera", prima=0.97, dopo=0.97)],
            sfasamenti=[[r.inizio, r.sfas, r.mm, r.corr] for r in pd.read_csv(f"{R}/dati/sfasamento_orario_misura_per_evento.csv")
                        .rename(columns={"inizio evento": "inizio", "sfasamento": "sfas"}).itertuples() if r.corr >= 0.6])
vt = pd.read_csv(f"{R}/dati/validazione_temperatura.csv")
vp = pd.read_csv(f"{R}/dati/validazione_pioggia.csv")
st = pd.read_csv("/home/claude/outputs/qgis/stazioni_regionali_medie_2001_2020.csv")
orari = {"Salerno genio civile": "P", "Cologna": "P", "Pellezzano": "P", "Cava dei tirreni": "P", "Baronissi": "P",
         "Castiglione del genovesi": "P", "Salerno meteo": "T", "Mercato s.severino": "T", "Tramonti": "T", "Pizzolano": "T"}
stations = [dict(nome=x.nome, lat=r(x.lat, 5), lon=r(x.lon, 5), tipo=orari.get(x.nome, "R"), dist=r(x.dist_punto_km, 1),
                 tmed=r(x.tmedia, 2), pioggia=r(x.pioggia_annua_mm, 0)) for x in st.itertuples()]
out = dict(punto=dict(lat=40.693522, lon=14.760576, quota=245.7, mare_km=1.8),
           annual=annual, clim=clim, daily=daily, valp=valp,
           valt=vt.round(3).to_dict(orient="records"), valg=vp.round(3).to_dict(orient="records"), stations=stations)
json.dump(out, open(f"{R}/assets/dati_dashboard.json", "w"), separators=(",", ":"), allow_nan=False)
print({k: (len(v) if isinstance(v, list) else "ok") for k, v in out.items()})
print(valp["totale"], valp["orario"], valp["giornaliero"])
print(annual)
