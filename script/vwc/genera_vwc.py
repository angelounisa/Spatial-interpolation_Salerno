"""Serie sintetiche di VWC 2016-2026 e parte stocastica (Laio et al. 2001).
1) serie oraria 2016-2026 a 15/30/60 cm forzata dalla pioggia interpolata (parametri tarati su tutto il periodo),
   con banda 5-95% dai residui di verifica incrociata;
2) pioggia stocastica a impulsi di Poisson marcati (lambda, alpha mensili) -> Monte Carlo del modello tarato;
3) pdf analitica stazionaria di Laio et al. (2001) per lo strato superficiale, per stagione;
"""
import sys, json, os
import numpy as np, pandas as pd
sys.argv = [sys.argv[0]] + ['/home/claude/outputs/vwc']
from laio_vwc import *          # dati, ET0, simulate, unpack, metrics, i_start, i_split, i_end

OUT = '/home/claude/outputs/vwc'
vf, va, vb = (np.load(f'{OUT}/{k}.npy') for k in ('vf', 'va', 'vb'))
PF = unpack(vf)
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ 1) serie 2016-2026
first = obs.index[0]
th0 = np.nanmean(O[(H.month <= 2)], axis=0)            # stato iniziale: media osservata gen-feb
SIM = simulate(P, ET0, th0, PF, Z, TH_H, TH_S, 4)
chain = metrics(SIM[i_start:i_end], O[i_start:i_end])   # catena intera, senza ripartire dalle misure
# residui di verifica (A verificato su 2026, B su 2025)
simA = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(va), Z, TH_H, TH_S, 4)
simB = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(vb), Z, TH_H, TH_S, 4)
k = i_split - i_start
res = np.vstack([simA[k:] - O[i_split:i_end], simB[24 * 30:k] - O[i_start + 24 * 30:i_split]])
mon = np.r_[H[i_split:i_end].month, H[i_start + 24 * 30:i_split].month]
wetm = lambda m: np.isin(m, [10, 11, 12, 1, 2, 3, 4])
q = {}
for j in range(3):
    for lab, msk in (('umido', wetm(mon)), ('secco', ~wetm(mon))):
        r = res[msk, j]; r = r[~np.isnan(r)]
        q[(j, lab)] = np.percentile(r, [5, 95])
lo = np.empty_like(SIM); hi = np.empty_like(SIM)
wm = wetm(H.month)
for j in range(3):
    for lab, msk in (('umido', wm), ('secco', ~wm)):
        a, b = q[(j, lab)]
        lo[msk, j] = np.clip(SIM[msk, j] - b, TH_H[j], TH_S[j]); hi[msk, j] = np.clip(SIM[msk, j] - a, TH_H[j], TH_S[j])
cover = {NAMES[j]: float(np.nanmean((O[i_start:i_end, j] >= lo[i_start:i_end, j]) & (O[i_start:i_end, j] <= hi[i_start:i_end, j]))) for j in range(3)}

# ------------------------------------------------------------------ 2) pioggia stocastica (Poisson marcato)
day = (H - pd.Timedelta(minutes=1)).floor('D')
Pd = pd.Series(P, index=H).groupby(day).sum()
THR = 1.0
lam = (Pd >= THR).groupby(Pd.index.month).mean()                 # probabilita' giornaliera ~ lambda [1/giorno]
alp = Pd[Pd >= THR].groupby(Pd[Pd >= THR].index.month).mean()   # altezza media per evento [mm]
# profili orari osservati dei giorni piovosi, per mese
Ph = pd.Series(P, index=H)
prof = {m: [] for m in range(1, 13)}
for dd, tot in Pd[Pd >= THR].items():
    x = Ph[dd + pd.Timedelta(hours=1): dd + pd.Timedelta(hours=24)].to_numpy()
    if len(x) == 24: prof[dd.month].append(x / x.sum())
prof = {m: np.array(v) for m, v in prof.items()}
yrs = list(range(2016, 2026))
E0y = {y: ET0[(H.year == y) & ~((H.month == 2) & (H.day == 29))][:8760] for y in yrs}
E0y = {y: v for y, v in E0y.items() if len(v) == 8760}
doy_month = pd.date_range('2019-01-01 01:00', periods=8760, freq='h')
dm = ((doy_month - pd.Timedelta(minutes=1)).floor('D')).month.to_numpy()[::24]
NY = 300
Ps, Es = [], []
for y in range(NY):
    r = np.zeros(8760)
    for dday in range(365):
        m = dm[dday]
        if rng.random() < lam[m]:
            depth = rng.exponential(alp[m])
            pr = prof[m][rng.integers(len(prof[m]))]
            r[dday * 24:(dday + 1) * 24] = depth * pr
    Ps.append(r); Es.append(E0y[rng.choice(list(E0y))])
Ps, Es = np.concatenate(Ps), np.concatenate(Es)
MC = simulate(Ps, Es, th0, PF, Z, TH_H, TH_S, 4)[8760:]           # primo anno di riscaldamento
mc_month = np.tile(doy_month.month.to_numpy(), NY - 1)

# ------------------------------------------------------------------ 3) pdf analitica di Laio (strato 1)
SEAS = {'inverno (DGF)': [12, 1, 2], 'primavera (MAM)': [3, 4, 5], 'estate (GLA)': [6, 7, 8], 'autunno (SON)': [9, 10, 11]}
def laio_pdf(i, lam_d, alpha, emax_d, ngrid=4000):
    tst, tfc, Ks, b, c = PF[i]
    th = TH_H[i] + np.geomspace(1e-10, TH_S[i] - TH_H[i] - 1e-7, ngrid)   # fitto vicino a th_h (singolarita' integrabile)
    f = np.clip((th - TH_H[i]) / (tst - TH_H[i]), 0, 1)
    L = np.where(th > tfc, Ks * 24 * (np.exp(b * (th - tfc)) - 1) / (np.exp(b * (TH_S[i] - tfc)) - 1), 0)
    rho = (c * emax_d * f + L) / Z[i]                                    # [theta/giorno]
    inv = 1 / rho
    I = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(th))])
    lg = -np.log(rho) - Z[i] * th / alpha + lam_d * I
    p = np.exp(lg - lg.max()); p /= np.trapezoid(p, th)
    return th, p
pdfs = {}
bins = {j: np.linspace(TH_H[j], TH_S[j], 41) for j in range(3)}
E0d = pd.Series(ET0, index=H).groupby(day).sum()
for sn, ms in SEAS.items():
    sel = Pd.index.month.isin(ms)
    lam_s = (Pd[sel] >= THR).mean(); alp_s = Pd[sel][Pd[sel] >= THR].mean(); em = E0d[E0d.index.month.isin(ms)].mean()
    th_a, p_a = laio_pdf(0, lam_s, alp_s, em)
    xg = np.linspace(TH_H[0], TH_S[0], 161); xc = 0.5 * (xg[1:] + xg[:-1])
    cdf = np.concatenate([[0], np.cumsum(0.5 * (p_a[1:] + p_a[:-1]) * np.diff(th_a))])
    pg = np.diff(np.interp(xg, th_a, cdf)) / np.diff(xg)          # densita' media per classe (conserva la massa)
    ent = dict(lam=float(lam_s), alpha=float(alp_s), et0=float(em), media_analitica=float(np.trapezoid(p_a * th_a, th_a)),
               analitica=dict(x=xc.round(4).tolist(), p=pg.round(3).tolist()), strati={})
    for j in range(3):
        hm = lambda x: np.histogram(x[~np.isnan(x)], bins=bins[j], density=True)[0].round(3).tolist()
        ent['strati'][NAMES[j]] = dict(bins=bins[j].round(4).tolist(),
                                       mc=hm(MC[np.isin(mc_month, ms), j]),
                                       serie=hm(SIM[H.month.isin(ms), j]),
                                       misura=hm(O[H.month.isin(ms), j]))
    pdfs[sn] = ent
# Monte Carlo a un solo strato (tasso giornaliero costante) per confermare la pdf analitica
check = {}
for sn, ms in SEAS.items():
    e = pdfs[sn]; nd = 200000
    rr = np.zeros(nd * 24); ev = rng.random(nd) < e['lam']
    rr[np.arange(nd)[ev] * 24] = rng.exponential(e['alpha'], ev.sum())
    one = simulate(rr, np.full(nd * 24, e['et0'] / 24), np.array([0.1, 0.1, 0.1]), PF, Z, TH_H, TH_S, 4)[24 * 365:, 0]
    h = np.histogram(one, bins=bins[0], density=True)[0]
    x = 0.5 * (bins[0][1:] + bins[0][:-1]); pa = np.interp(x, e['analitica']['x'], e['analitica']['p'])
    check[sn] = dict(mc_1strato=h.round(3).tolist(), media_mc=float(np.mean(one)), media_analitica=e['media_analitica'])
    print(sn, 'media MC 1 strato', round(float(np.mean(one)), 4), 'media analitica', round(e['media_analitica'], 4))

# ------------------------------------------------------------------ quantili mensili e indicatori
def mq(X, months):
    return {NAMES[j]: [np.nanpercentile(X[months == m, j], [5, 25, 50, 75, 95]).round(4).tolist() for m in range(1, 13)] for j in range(3)}
quant = dict(mc=mq(MC, mc_month), serie=mq(SIM, H.month.to_numpy()), misura=mq(O, H.month.to_numpy()))
# durata dei periodi sotto soglia (theta15 < 0.04), giorni consecutivi
def spells(x, thr):
    b = x < thr; out = []; c = 0
    for v in b:
        if v: c += 1
        elif c: out.append(c / 24); c = 0
    return np.array(out)
spell = {k: (lambda s_: dict(n_anno=len(s_) / ny, media_g=float(s_.mean()) if len(s_) else 0, max_g=float(s_.max()) if len(s_) else 0))(spells(x, 0.04))
         for k, x, ny in (('mc', MC[:, 0], NY - 1), ('serie 2016-2026', SIM[:, 0], len(H) / 8766), ('misura', np.nan_to_num(O[i_start:i_end, 0], nan=1), (i_end - i_start) / 8766))}
print('periodi secchi theta15<0.04', spell)

# ------------------------------------------------------------------ salvataggi
ser = pd.DataFrame({'data_ora_fine_intervallo_solare': H.strftime('%Y-%m-%d %H:%M'),
                    'pioggia_mm': np.round(P, 2), 'ET0_mm': np.round(ET0, 3)})
for j, nm in enumerate(['15cm', '30cm', '60cm']):
    ser[f'vwc_{nm}'] = SIM[:, j].round(4); ser[f'vwc_{nm}_p05'] = lo[:, j].round(4); ser[f'vwc_{nm}_p95'] = hi[:, j].round(4)
for j, nm in enumerate(['15cm', '30cm', '60cm']):
    ser[f'vwc_{nm}_misurato'] = np.round(O[:, j], 4)
ser.to_csv(f'{OUT}/vwc_sintetico_orario_2016_2026.csv', index=False)
ho = obs[['v15', 'v30', 'v60']].copy(); ho.index.name = 'data_ora_fine_intervallo_solare'
ho.round(4).to_csv(f'{OUT}/vwc_misurato_medie_orarie.csv')
pd.DataFrame({'mese': range(1, 13), 'lambda_eventi_giorno': lam.values.round(3), 'alpha_mm': alp.values.round(2)}).to_csv(f'{OUT}/pioggia_poisson_marcato_mensile.csv', index=False)
mcd = pd.DataFrame({'ora': np.arange(len(MC)), **{f'vwc_{n}': MC[:, j].round(4) for j, n in enumerate(['15cm', '30cm', '60cm'])}})
mcd.iloc[:24 * 365 * 50].to_csv(f'{OUT}/vwc_montecarlo_50anni.csv.gz', index=False, compression='gzip')
rep = json.load(open(f'{OUT}/taratura_laio.json'))
summary = dict(catena=chain, copertura_banda=cover,
               poisson=dict(lam=lam.round(3).tolist(), alpha=alp.round(2).tolist()), periodi_secchi=spell,
               pdf=pdfs, check_analitica=check, quantili=quant, verifica=rep, NY=NY)
json.dump(summary, open(f'{OUT}/sintesi_vwc.json', 'w'))
print('catena', {n: {m: round(x, 3) for m, x in d.items()} for n, d in chain.items()})
print('copertura banda', cover)
