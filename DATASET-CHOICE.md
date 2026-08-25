# Why this dataset, and what it cannot be used for

The probe runs on the [EV Battery Failure Prediction dataset (200k)](https://www.kaggle.com/datasets/sarveshchhetri/ev-battery-failure-prediction-dataset-200k):
200,000 rows, 70 columns, binary target `battery_failure` at a 9.96% positive rate,
533,554 missing cells, 60 numeric and 10 categorical columns.

It is a good bed for a runnability probe and a bad one for a model comparison. Both
halves of that need stating, because the second half is easy to forget once numbers
start appearing.

## It is synthetic, and says so

From the dataset's own `feature_descriptions.md`:

> This is a synthetic dataset. Every feature is generated from underlying battery-degradation
> physics rather than independent random sampling, so the correlations between features (and
> between features and the target) mirror real-world electric vehicle battery behavior

The generation process is more thoughtful than most synthetic tables. Features come from a
degradation model rather than independent sampling. It is still a table where the label was
computed from the features by a known rule.

That matters more here than it would elsewhere. TabFM is pretrained entirely on synthetic
tables generated from structural causal models. Scoring an SCM-pretrained model on a
physics-simulated causal table is close to a best case for it, and any advantage it shows
would be as much about matching inductive bias as about capability.

## The target is nearly a function of one column

Measured on the held-out split by [`scripts/feature_baseline.py`](scripts/feature_baseline.py),
single-feature ROC AUC:

| feature | AUC alone |
|---|---|
| `capacity_loss_percent` | 0.9589 |
| `aging_score` | 0.9587 |
| `battery_health_percent` | 0.9586 |
| `state_of_health` | 0.9583 |
| `cell_voltage_std` | 0.9534 |
| `voltage_imbalance` | 0.9370 |

Ten features individually clear 0.90 AUC. Sixteen clear 0.80.

The dataset's own documentation names `capacity_loss_percent` as "the central degradation
driver of the whole dataset" and defines `battery_health_percent` as `100 − capacity loss`.
These are the generative variables. The label is downstream of them, so recovering it is
close to reading it off.

## Which leaves no headroom

With 2,000 training rows and a 2,000-row held-out set:

| model | ROC AUC | fit (s) | predict (s) |
|---|---|---|---|
| TabFM | 0.9900 | 0.189 | 16776.951 |
| TabICL | 0.9897 | 0.424 | 23.474 |
| TabDPT | 0.9894 | 0.005 | 1.642 |
| Logistic regression | 0.9881 | 0.024 | 0.002 |
| LightGBM | 0.9870 | 2.008 | 0.010 |
| CatBoost | 0.9849 | 1.882 | 0.003 |
| XGBoost | 0.9807 | 0.847 | 0.007 |

Every model lands between 0.98 and 0.99. TabFM takes it by 0.0019 over a linear model on
median-imputed features, having spent 16,776.951 seconds to the linear model's 0.002.

There is nothing here for a foundation model to win. Any two of these sit inside each
other's noise, and a ranking built on this table would be measuring the resampling seed.

## So what it is for

Exactly one thing: **does the model run on this machine, and what does it cost.**

For that it is close to ideal. The signal is strong enough that a working model must score
well above 0.95, which turns the probe into a correctness check rather than a smoke test.
A library that imports, fits, predicts and returns 0.62 has failed even though it raised no
exception. Silent wrongness is the failure mode that matters with these libraries, and a
saturated dataset catches it.

It is also honest about scale. 200,000 rows with 67 usable features is a realistic size for
an in-context model to choke on, which is the cost question worth measuring.

## What replaced it

Four observed datasets from OpenML, registered in [`scripts/datasets.py`](scripts/datasets.py):
`credit_g`, `heloc`, `bank_marketing` and `kdd_appetency`. All four appear by name in
TabArena's 51-dataset suite, and all four have genuine headroom. Their best single
feature scores between 0.5795 and 0.8209, against 0.9589 here, and none of them has a
single column above 0.90. Results in [FINDINGS.md](FINDINGS.md).

Nothing in this repository's headline claims rests on the EV battery table.

If this work continues, the obvious next step is non-IID splits. BeyondArena covers 142
datasets across IID, temporal and grouped task types, and TabFM has no published results
on any of them. Feasible candidates there include `sberbank_housing_market_forecasting`
(27k rows) and `hotel_booking_demand` (81k) for temporal, and `5g_energy_consumption`
(92k) for grouped.
