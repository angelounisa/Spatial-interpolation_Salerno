"""Combinazione del modello con ERA5-Land, in verifica incrociata 2025 <-> 2026.
Per ogni profondita': misura = a + b * modello + somma_k c_k * ERA5_k (k = 0-7, 7-28, 28-100 cm), minimi quadrati
sull'anno di taratura (con il modello tarato sullo stesso anno), applicata all'altro anno."""
import sys, json, os
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], '/home/claude/outputs/vwc']
from laio_vwc import *
OUT = '/home/claude/outputs/vwc'
E5 = pd.read_csv('/home/claude/data/era5land_suolo_orario.csv', parse_dates=[0], index_col=0).reindex(H).interpolate(limit=6).to_numpy()


def cv_sims(kind):
    if kind == 'laio':
        va, vb = np.load(f'{OUT}/va.npy'), np.load(f'{OUT}/vb.npy')
        A = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(va), Z, TH_H, TH_S, 4)
        B = simulate(P[i_start:i_end], ET0[i_start:i_end], O[i_start], unpack(vb), Z, TH_H, TH_S, 4)
    else:
        import richards_mol as R
        pre = 'rich2' if os.path.exists(f'{OUT}/rich2_A.npy') else 'rich'
        A = R.simulate(np.load(f'{OUT}/{pre}_A.npy'), i_start, i_end, O[i_start]); B = R.simulate(np.load(f'{OUT}/{pre}_B.npy'), i_start, i_end, O[i_start])
    return A, B


VAR = False
PRED = {}

def evaluate(models):
    """models: dict nome -> (A, B) simulazioni sull'intero periodo (A tarato 2025, B tarato 2026) o None per solo ERA5."""
    k = i_split - i_start; Ob = O[i_start:i_end]; E = E5[i_start:i_end]
    tr = {'2025→2026': (slice(24 * 30, k), slice(k, None), 0), '2026→2025': (slice(k + 24 * 30, None), slice(0, k), 1)}
    res = {}
    for name, sims in models.items():
        res[name] = {}
        for lab, (s_tr, s_te, w) in tr.items():
            pred = np.full((Ob.shape[0], 3), np.nan)
            for j in range(3):
                cols = [E[:, 0], E[:, 1], E[:, 2]]
                if sims is not None: cols = [sims[w][:, j]] + (cols if '+ ERA5-Land' in name else [])
                X = np.column_stack([np.ones(len(Ob))] + cols)
                m = ~np.isnan(Ob[:, j]) & np.all(np.isfinite(X), 1)
                mt = np.zeros(len(Ob), bool); mt[s_tr] = True; mt &= m
                beta = np.linalg.lstsq(X[mt], Ob[mt, j], rcond=None)[0]
                pred[:, j] = X @ beta
                if VAR:                                   # stessa varianza della misura nell'anno di taratura
                    mu = pred[mt, j].mean(); pred[:, j] = Ob[mt, j].mean() + (pred[:, j] - mu) * Ob[mt, j].std() / pred[mt, j].std()
            ob_te = Ob[s_te].copy(); ob_te[~np.isfinite(pred[s_te])] = np.nan     # ore senza ERA5-Land (ultimi giorni) escluse
            mm = metrics(np.nan_to_num(pred[s_te]), ob_te)
            for j, dep in enumerate(NAMES):
                o = ob_te[:, j]; p_ = pred[s_te, j]; ok = ~np.isnan(o); q = o[ok] >= np.quantile(o[ok], 0.95)
                mm[dep]['bias_picchi'] = float(np.mean(p_[ok][q] - o[ok][q]))
            res[name][lab] = mm
            PRED[(name, lab)] = pred
    return res


if __name__ == '__main__':
    kinds = sys.argv[1:] if len(sys.argv) > 1 else ['laio']
    L = cv_sims('laio')
    models = {'Laio': L, 'ERA5-Land riscalato': None, 'Laio + ERA5-Land': L}
    # 'Laio' da solo con riscalatura lineare (stessa regressione, senza ERA5) per confronto equo
    if 'richards' in kinds:
        Rr = cv_sims('richards'); models.update({'Richards': Rr, 'Richards + ERA5-Land': Rr})
    res = evaluate(models)
    globals()['VAR'] = True
    res2 = evaluate({k + ' (varianza conservata)': v for k, v in models.items() if v is not None and k != 'Laio' and k != 'Richards'})
    res.update(res2)
    rows = []
    for nm, d in res.items():
        for lab, mm in d.items():
            for dep, x in mm.items():
                rows.append(dict(modello=nm, verifica=lab, profondita=dep, **{k: round(v, 3) for k, v in x.items()}))
    R_ = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(R_.groupby(['modello', 'profondita'])[['NSE', 'KGE', 'corr', 'RMSE', 'bias', 'bias_picchi']].mean().round(3).to_string())
    R_.to_csv(f'{OUT}/confronto_modelli_cv.csv', index=False)
