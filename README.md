<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12"></a>
  <img src="https://img.shields.io/badge/datasets-4%20real-2a78d6?style=for-the-badge" alt="4 real datasets">
  <img src="https://img.shields.io/badge/API%20keys-none-1baf7a?style=for-the-badge" alt="No API keys">
  <a href="https://satsawat.ai"><img src="https://img.shields.io/badge/author-satsawat.ai-e8a112?style=for-the-badge" alt="Author: satsawat.ai"></a>
</p>

# Tabular foundation models against gradient boosting, on real data

Companion code for the writing at [satsawat.ai](https://satsawat.ai).

Four observed datasets, seven models, identical splits, paired bootstrap with Holm
correction, and prediction cost measured alongside accuracy.

The short version: foundation models win two of the four datasets by about two points of
AUC, and a gradient boosting machine given the training data it would actually have in
production closes the whole gap on one of them and wins the other two outright.

Full results and reasoning in [FINDINGS.md](FINDINGS.md).

## Headline

| dataset | best model | vs strongest classical | survives Holm |
|---|---|---|---|
| `heloc` | TabFM 0.8247 | +0.0230 over logistic regression | yes, p = 0.0192 |
| `bank_marketing` | TabFM 0.9420 | +0.0194 over CatBoost | yes, p = 0.0288 |
| `credit_g` | TabDPT 0.7779 | +0.0244 over XGBoost | no, p = 0.7504 |
| `kdd_appetency` | TabFM 0.7164 | +0.0238 over logistic regression | no, p = 1.0000 |

Then the same LightGBM, trained on every available row instead of the 1,000 the foundation
models could afford:

| dataset | capped at 1,000 rows | uncapped | rows used |
|---|---|---|---|
| `heloc` | 0.7824 | 0.8149 | 9,459 |
| `bank_marketing` | 0.8930 | 0.9421 | 44,211 |
| `kdd_appetency` | 0.6446 | 0.7741 | 47,000 |

LightGBM on all 44,211 rows of `bank_marketing` scores 0.9421. TabFM scored 0.9420 and
took 4,313 seconds per thousand predictions to do it. Against the fastest baseline on that
dataset, measured as the median of 200 calls, that is 11,712,158 times the cost.

## Cost

Seconds per 1,000 predictions, CPU.

| model | credit_g | heloc | bank_marketing | kdd_appetency |
|---|---|---|---|---|
| TabFM | 5,459.3300 | 4,627.9978 | 4,312.7678 | 8,702.0829 |
| TabICL | 5.5714 | 3.4587 | 2.8460 | 38.8713 |
| TabDPT | 0.6331 | 0.4047 | 0.3768 | 0.2267 |
| LightGBM | 0.0135 | 0.0091 | 0.0097 | 0.0065 |
| XGBoost | 0.0082 | 0.0033 | 0.0040 | 0.0037 |
| CatBoost | 0.0033 | 0.0011 | 0.0020 | 0.0038 |
| Logistic regression | 0.0015 | 0.0005 | 0.0005 | 0.0016 |

TabFM's shipped default is a 32-member ensemble, not the single forward pass its
announcement describes. Running both on `credit_g`: 1637.799 seconds against 51.297, for
0.0020 AUC.

## What ran and what did not

| model | version | weights licence | ran |
|---|---|---|---|
| TabFM | 1.0.1 | non-commercial | yes |
| TabICL | 2.1.1 | BSD-3-Clause | yes |
| TabDPT | 1.2.0 | Apache-2.0 | yes |
| TabPFN | 8.4.0 | non-commercial by default | **no** |
| Nori | 0.18.2 | Apache-2.0 | **not applicable** |

TabPFN refuses to download weights without an interactive licence acceptance, so it cannot
be installed unattended and is absent from every result here. Nori 0.18.2 is regression
only: its API defines `Task = Literal["regression", "reg"]` and the package contains no
reference to classification.

Every TabFM cell is now measured. The `kdd_appetency` run took 7 hours 15 minutes for
3,000 predictions.

## Reproduce

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.in
```

```bash
.venv/bin/python scripts/sweep.py --models logreg,xgboost,lightgbm,catboost,tabicl,tabdpt
```

```bash
.venv/bin/python scripts/compare.py && python3 scripts/verify_readme_claims.py
```

The datasets download themselves from OpenML on first use. Adding `--models tabfm` to the
sweep costs about six hours on CPU and roughly 7 GB of disk for the checkpoint.

## Layout

```
scripts/datasets.py          the four real datasets, fetched from OpenML and cached
scripts/prep.py              the shared split, plus the three feature representations
scripts/run_one.py           one model, one dataset, one subprocess
scripts/sweep.py             the real-data sweep and its per-dataset protocol
scripts/compare.py           paired bootstrap against a control, Holm corrected
scripts/context_reference.py what a tree scores without the context cap
scripts/feature_baseline.py  how much of the target one column alone can recover
scripts/time_baselines.py    median-of-200 timing so cost ratios are not artefacts
scripts/smoke_test.py        the synthetic runnability bed, separate from the sweep
scripts/summarise.py         reads the probe JSONs, prints the tables
scripts/verify_readme_claims.py  recomputes every number here and fails on drift
```

Each probe runs in its own process, so an out-of-memory kill is recorded as a result rather
than ending the sweep, and peak memory belongs to one model instead of to whatever ran
before it.

## Protocol notes that change the numbers

**The tree libraries get their categorical features.** Each ran twice, once with native
categorical handling and once on ordinal codes, and the better run stands for that library.
This is not cosmetic: enabling it moved CatBoost on `bank_marketing` from 0.9118 to 0.9227
and roughly halved the gap it is the control for.

**The control is picked after seeing the scores.** That maximises the reported gap in
whichever direction it points, so it is conservative for a win and anti-conservative for a
loss.

**Holm runs within a dataset, not across the four.**

**The trees are untuned.** Fixed sensible settings, no search. A tuned baseline would
narrow every gap reported here.

**One seed, one split, CPU only.** These are not TabArena-grade measurements and are not
meant to be.

## Also here

- [FINDINGS.md](FINDINGS.md) is the write-up.
- [RESEARCH-NOTES.md](RESEARCH-NOTES.md) is the lab journal, including the dead ends.
- [DATASET-CHOICE.md](DATASET-CHOICE.md) explains why the first dataset was thrown out of
  the comparison and kept only as a runnability check.
- [NOTICE.md](NOTICE.md) separates the code licence from the weight licences. The weights
  are not vendored and TabFM's cannot be used commercially.

## Author

Written by [Satsawat Natakarnkitkul](https://satsawat.ai), a data and AI practitioner in
ASEAN. Newsletter: [AI in Practice](https://satsawat.ai/#newsletter).

Companion repositories: [tsfm-bakeoff](https://github.com/netsatsawat/tsfm-bakeoff) asks the
same question of time-series foundation models, and
[agent-failure-lab](https://github.com/netsatsawat/agent-failure-lab) does it for compound
error in agents.
