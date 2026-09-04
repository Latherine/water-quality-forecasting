# Water Quality Forecasting — Hanoi

Forecasting the water-quality indicators **NH4, COD and TSS four hours ahead** at three Hanoi
sites (Bay Mau, Cau Nga, Ho Tay), from 5-minute sensor logs recorded during 2024.

The point of this project is **not** to build the most complex model. It is to find out whether
a complex model is justified at all — by measuring every model against a baseline that requires
no training.

## Headline result

**The baseline was kept in 7 of 9 cases.**

| Site | Indicator | Model kept | Skill vs baseline |
|---|---|---|---|
| Bay Mau | nh4, cod, tss | `persistence` | — |
| Cau Nga | nh4, cod | `persistence` | — |
| **Cau Nga** | **tss** | `gradient_boosting` | **+0.194** |
| Ho Tay | nh4, cod | `persistence` | — |
| **Ho Tay** | **tss** | `moving_average` | **+0.173** |

`persistence` is the rule *"in four hours it will be whatever it is now"* — one line of code.
An LSTM beat it in **1 of 9** cases, and even there gradient boosting did better.

## Why this project exists

An earlier version trained an LSTM to predict a composite Water Quality Index and reported
MAE / RMSE / R² with nothing to compare them to. Adding a baseline changed the picture
completely:

| Site | LSTM MAE | Persistence MAE | LSTM R² | Persistence R² |
|---|---|---|---|---|
| Bay Mau | 0.928 | **0.758** | 0.421 | 0.351 |
| Cau Nga | 2.550 | **0.762** | 0.868 | **0.956** |
| Ho Tay | 1.058 | **0.032** | −0.395 | **0.994** |

Persistence won at all three sites — by 33× at Ho Tay.

Three problems in the original approach drove this rebuild:

1. **Circular target.** The WQI is a weighted sum of `cod`, `nh4`, `tss` and `ph`. Predicting it
   from those same columns partly re-learns a formula we wrote by hand. `ph_score` is exactly
   100 for 96.9–99.9% of rows, and at Cau Nga `nh4` alone explains 81% of WQI variance.
2. **Invented data was scored.** Unbounded `interpolate()` turned Ho Tay's 52-day gap into a
   straight line. Roughly half its test set was synthetic, which is what produced R² = −0.395.
3. **No baseline.** Nothing established what "good" meant.

## Method

Four ideas keep the comparison honest:

| Technique | What it prevents |
|---|---|
| An `is_real` mask flagging every measured vs filled hour | scoring on invented data |
| Chronological split (oldest 85% train, newest 15% test) | the model seeing the future |
| One shared set of evaluation rows for all 8 models | a model getting easier data than its rivals |
| Paired bootstrap confidence intervals on the skill score | mistaking luck for skill |

That last one mattered. Two models had positive skill — Bay Mau/COD (+0.070) and Ho Tay/COD
(+0.043) — but their confidence intervals crossed zero under resampling, so they were rejected
as noise rather than adopted.

**Skill score:**

```
skill = 1 − (MAE of the model ÷ MAE of persistence)
```

Positive means better than assuming no change; negative means worse than doing nothing.

## Why the baseline is so hard to beat

The 4-hour autocorrelation of NH4 is **0.89 to 0.99**. The value now and the value four hours
from now are nearly the same number, so there is very little movement left for a model to
predict. Every model scores a high R² on NH4 — but only because the question is easy.

This is a finding about the data, not a modelling failure, and it points straight at the most
promising next experiment: a 24- or 48-hour horizon.

## Repository layout

```
01_eda.ipynb                     Explore the raw files; produces 6 numbered findings
02_data_preprocessing.ipynb      Clean per those findings, aggregate to hours, select features
03_model_training.ipynb          Train 8 models per site and indicator
04_evaluation_comparison.ipynb   Score, compare, bootstrap-check, decide
05_make_forecast.ipynb           Use the chosen models to make a real forecast

forecast.py                      The forecasting module (usable outside Jupyter)
models/                          Fitted machine-learning models
predictions/                     One CSV per site+indicator: actual + every model's forecast
results/
  model_scores.csv               MAE / RMSE / R² / skill for every model
  bootstrap_checks.csv           Which improvements survive resampling
  final_decision.csv             The model to keep for each site and indicator
```

Each notebook hands off through files, so any one of them can be run on its own.

## Making a forecast

```python
import pandas as pd
from forecast import forecast

recent = pd.read_pickle("processed/CAU_NGA_hourly.pkl")["values"].tail(48)
answer = forecast("CAU_NGA", "tss", recent)

print(answer["forecast"], "expected at", answer["valid_for"])
```

Or run `python forecast.py` for all nine forecasts at once.

`forecast.py` reads `results/final_decision.csv`, so it always applies whichever model won for
that site and indicator — nothing is hard-coded. Rerun the comparison and forecasting follows
the new decision automatically.

Only `gradient_boosting` loads a file from `models/`. `persistence` and `moving_average` have no
trained parameters and are recreated in one line each, which is why 8 of the 9 chosen models
need nothing stored at all.

Each result also reports `reading_age_hours` — how old the newest genuine measurement is. These
sensors drop out often, and a forecast built on a six-hour-old reading deserves less trust than
one built on the current hour.

Also included for context:

- `Water_Quality_4h_LSTM_Forecasting.ipynb` — the original LSTM approach
- `Water_Quality_Baseline_Forecasting.ipynb` — the single-notebook rebuild, since split into 01–04

## The models

| Model | Type | Idea |
|---|---|---|
| `train_mean` | baseline | always predict the training average |
| **`persistence`** | **baseline** | **the value in 4 hours equals the value now** |
| `seasonal_naive` | baseline | the same clock hour yesterday |
| `moving_average` | baseline | the mean of the last 4 hours |
| `drift` | baseline | continue the recent trend linearly |
| `ridge` | machine learning | linear model on lagged values |
| `gradient_boosting` | machine learning | tree ensemble on lagged values |
| `lstm` | deep learning | neural network over a 24-hour sequence |

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

Then run the notebooks in order, `01` through `05`.

The raw `.xlsx` sensor files are **not** included in this repository. Place them in the project
root as `BAY MAU.xlsx`, `CAU NGA.xlsx` and `HO TAY.xlsx` to reproduce the pipeline from scratch.
What you can do without the raw data:

| Notebook | Runs from a fresh clone? |
|---|---|
| `01`, `02` | No — needs the `.xlsx` files |
| `03` | No — needs `processed/`, built by `02` |
| `04` | **Yes** — `predictions/` and `results/` are committed |
| `05` | No — needs `processed/` for recent readings, though `models/` is committed |

All notebook outputs are saved in the files, so the full analysis is readable on GitHub without
running anything.

TensorFlow is optional — notebook 03 skips the LSTM automatically if it is not installed, and
every baseline still runs.

## About the data

Three sites, 5-minute logging through 2024. Real coverage is well below 100%:

| | Bay Mau | Cau Nga | Ho Tay |
|---|---|---|---|
| Rows | 89,863 | 76,618 | 61,292 |
| Time coverage | 85% | 91% | **72%** |
| Largest gap | 41 days | 1.3 days | **52 days** |

Eight of the sixteen measurement columns are empty at every site. `no3` is missing at Bay Mau
only; `flow_in` is missing at Cau Nga only — so feature sets are chosen **per site** rather than
globally.

Ho Tay's gaps leave only 169–232 usable test hours, so its results are indicative rather than
conclusive.

## Next steps

1. **Forecast 24–48 hours ahead** instead of 4. Persistence degrades sharply at longer horizons,
   giving a real model room to prove itself. One constant (`HORIZON_HOURS`) per notebook.
2. **Recover Ho Tay's missing months**, or continue reporting its numbers as indicative.
3. **Predict events rather than values** — e.g. *will NH4 exceed the QCVN threshold in the next
   24 hours?* That is operationally useful and cannot be won by copying the last value.
