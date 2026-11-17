# Spatial-interpolation_Salerno

Temperatura e pioggia **orarie 2016–2026** su un punto sopra Salerno (40.693522 N, 14.760576 E, 246 m),
ricostruite dai sensori del Centro Funzionale Multirischi della Regione Campania con un'interpolazione
che tiene conto dell'orografia (DTM TINITALY 10 m). La pioggia è tarata sulla misura locale (01/2025–07/2026).

**Dashboard:** https://angelounisa.github.io/Spatial-interpolation_Salerno/

## Contenuto
| Percorso | Cosa contiene |
|---|---|
| `index.html`, `assets/` | dashboard (risultati, validazione, metodologia, download) |
| `dati/salerno_punto_orario_2016_2026_tarato.csv` | **serie oraria finale** |
| `dati/stazioni_10min_2016_2026.zip` | dati a 10 minuti delle 10 stazioni dopo i controlli |
| `dati/validazione_*.csv`, `dati/verifica_stazioni.csv` | validazione e controlli |
| `dati/stazioni_regionali_orografia_10m.csv` | 71 stazioni regionali con medie 2001–2020 e variabili orografiche |
| `script/` | script Python per riprodurre l'elaborazione |
| `dati/strati/` | strati orografici GeoTIFF a 10 m (due archivi) e 50 m, EPSG:32632 |

## Metodo in breve
1. Unione e controllo dei dati a 10 minuti; aggregazione oraria (ora = fine intervallo).
2. Variabili orografiche dal DTM (quota, posizione nel rilievo, altezza sul fondovalle, distanza dal mare, esposizione).
3. Medie di lungo periodo dalla rete regionale (2001–2020) spiegate con le variabili orografiche.
4. Ogni ora ogni sensore è riportato al punto con la differenza (temperatura) o il rapporto (pioggia) tra le medie, poi kriging ordinario.
5. Misura locale riportata all'ora solare (il registratore era in ora legale in parte del periodo); pioggia tarata con un fattore unico (1,072); serie finale = misura dove disponibile. Correlazione oraria con la misura: 0,93.

## Limiti
Dati di partenza non validati. La temperatura ha un errore orario stimato di circa 1,6–2 °C perché nessun termometro
orario ha la posizione del punto (mezza costa, vicino al mare). Dettagli nella sezione Metodologia della dashboard.

Dati: Centro Funzionale Multirischi – Protezione Civile Regione Campania; TINITALY 1.1 (INGV, CC BY 4.0).

## Serie sintetiche di VWC (Laio et al. 2001)

`dati/vwc_sintetico_orario_2016_2026.csv`: contenuto d'acqua volumetrico orario a 15, 30 e 60 cm con banda 5–95% e suzione stimata. È generato da un bilancio idrico a tre strati con la funzione di perdita di Laio et al. (2001), tarato sulle misure del sito (suolo nudo, 31/01/2025–16/11/2026) e forzato dalla pioggia interpolata e tarata. La parte stocastica usa piogge a impulsi di Poisson marcati (Monte Carlo di 300 anni) e la pdf analitica di Laio per lo strato superficiale. Script in `script/vwc/`.
