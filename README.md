<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12"></a>
  <img src="https://img.shields.io/badge/datasets-4%20real-2a78d6?style=for-the-badge" alt="4 real datasets">
  <img src="https://img.shields.io/badge/API%20keys-none-1baf7a?style=for-the-badge" alt="No API keys">
  <a href="https://satsawat.ai"><img src="https://img.shields.io/badge/author-satsawat.ai-e8a112?style=for-the-badge" alt="Author: satsawat.ai"></a>
</p>

# Tabular foundation models against gradient boosting, on real data

<p align="center">
  <img src="assets/scorecard.gif" alt="Four datasets, four ties. The foundation model matches a gradient-boosted tree on accuracy and costs millions of times more to run." width="760">
</p>

Companion code for the writing at [satsawat.ai](https://satsawat.ai).

Every problem gets a large language model thrown at it now. A large language model is the
kind of model behind today's chatbots. Summarise this, classify that,
pull a field out of the other thing, and lately, predict the number in this spreadsheet. A
foundation model is a large model trained once on a huge pile of data, then reused for many
jobs without retraining. On text and images, foundation models earned that reflex. They are
genuinely hard to beat there. The question I could not answer was whether the reflex carries
over to tables, which is where most business prediction lives. Will this customer leave, will
this loan default, is this payment fraud, will this person take the offer. The rows-and-columns
problems a data science team is handed every week. For twenty years the standard tool for
those jobs has been a gradient-boosted tree, a model you train on your own past rows. A new
kind of model, the tabular foundation model, says you can skip the training step and hand it
the table instead. If that holds, it changes how tabular machine learning gets staffed and
shipped.

This repo is that check. Does a tabular foundation model actually replace the tree, or does
the reflex break the moment you look past the leaderboard? A leaderboard is a public ranking
that sorts models by their score. Three of the new models run
against three trees and a plain logistic regression on four public datasets, on identical
rows. Logistic regression is the simplest standard model, a weighted sum of the columns. The
repo asks two concrete questions. Does the new model score higher in a way that is not luck?
And what does each prediction cost?

The short answer, from the runs committed here. When every model is held to the small
number of rows the new models can handle, a foundation model scores highest on all four
datasets. Give the tree all the rows it would have in real life and all four datasets end
in a tie. The best new model also costs millions of times more per prediction, and its
licence bars commercial use.

A script re-derives the numbers in this file from the committed result files.
[`scripts/verify_readme_claims.py`](scripts/verify_readme_claims.py) recomputes every
result-shaped figure, meaning any number written with three or more decimals or with comma
grouping, and it fails if a recomputed figure no longer matches those files. A few plain
two-decimal figures below, such as the peak memory in gigabytes, are read straight from the
probe files rather than recomputed by that script. Check the claims rather than trust them.

## The models and the terms

Seven models ran. Three are tabular foundation models (Google's TabFM, TabICL and TabDPT).
Three are gradient-boosted trees (XGBoost, LightGBM and CatBoost). The seventh is logistic
regression. Below I call the trees and logistic regression the classical models.

- Tabular data is data laid out as rows and columns, like a spreadsheet. Each row is one
  customer or one loan, and each column is one fact about it.
- Training means showing a model your past rows, answers included, so it can learn a rule
  from them. The numbers a model learns are its weights. When you download a model, the
  weights are what you download, and they carry their own licence.
- A gradient-boosted tree is the standard tool for table prediction. It is built from many
  small decision trees, each one fixing the mistakes of the ones before. It trains on your
  data once and then predicts very cheaply.
- A tabular foundation model is a large model that its maker trained once on many tables.
  You do not train it on your data. You hand it your rows and it works out each answer
  from them on the spot, so its cost grows with your data.
- Logistic regression is the simplest sensible model, a weighted sum of the columns. It is
  in the study as the floor that every serious model should beat.
- Context rows are the rows a model is handed as its training data. The study capped every
  model at 700 context rows on German credit and 1,000 on the other three. TabFM's cost
  grows with every context row it reads. At 1,000 rows it already took thousands of seconds
  to score a thousand rows on a CPU (see the cost table below), so the cap stayed there.
- A probe is one model run on one dataset. Each probe is saved as a JSON file under
  `results/probe/`, with a `.npz` file (a compressed NumPy array file) of its predictions
  beside it.
- ROC AUC, written AUC or score below, is the accuracy measure. A score of 0.5 is a coin
  flip and 1.0 is perfect. It measures how well a model ranks the positive rows above the
  rest, where the positives are the rarer of the two outcomes on each dataset. Every
  accuracy number in this file is an AUC.
- The tie test tells whether a gap between two scores is real or noise. Its formal name is
  a paired bootstrap, the name FINDINGS.md and the file list use. The study draws 5,000
  random re-picks of the test rows (some rows repeated, some left out), rescores both
  models on each re-pick, and writes down the gap each time. The 95% interval is the range
  that held 95 out of every 100 of those gaps. Read it like an election poll. A lead inside
  the margin of error is a tie. On some re-picks the gap flipped sign, meaning the other
  model led. The p value is the share of those flips, doubled because the true lead could
  have gone either way. A p under 0.05 counts as a real lead.
- The control is the strongest classical model on a dataset, the one every other model is
  measured against. With six models tested against the control on each dataset, one of them
  can look like a winner by luck alone. The Holm correction tightens each p to account for
  those six tries. A lead that "survives Holm" stays under 0.05 after that tightening.

## How to run

Clone the repo and install with uv, a fast installer for Python packages. uv comes from
PyPI, so `pip install uv` gets it if you do not have it. Keep the folder name `.venv`:
`scripts/sweep.py` and `scripts/smoke_test.py` call `.venv/bin/python` by that exact path.

```bash
git clone https://github.com/netsatsawat/tabular-fm-scorecard.git && cd tabular-fm-scorecard
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

`requirements.txt` pins every package to the exact version the study ran on.
`requirements.in` is the short list of packages I asked for, and `requirements.txt` was
generated from it.

The quick command is `scripts/showdown.py`. It reads the saved predictions for TabFM and for
the LightGBM tree that was trained on every training row, and reruns the tie test between
them. It needs no dataset download and no model weights, so it runs offline.

```bash
.venv/bin/python scripts/showdown.py
```

```
bank_marketing   tree(uncapped) 0.94205  TabFM 0.94205  delta -0.00000  CI [-0.0077, +0.0076]  p=0.9956  indistinguishable
credit_g         tree(uncapped) 0.72259  TabFM 0.76656  delta -0.04397  CI [-0.0870, +0.0000]  p=0.0504  indistinguishable
heloc            tree(uncapped) 0.81495  TabFM 0.82474  delta -0.00980  CI [-0.0235, +0.0039]  p=0.1632  indistinguishable
kdd_appetency    tree(uncapped) 0.77414  TabFM 0.71635  delta +0.05779  CI [-0.0079, +0.1315]  p=0.0868  indistinguishable

wrote results/showdown.json
```

One line per dataset. `tree(uncapped)` is LightGBM trained on every training row the
dataset has. `TabFM` is the foundation model on its 700 or 1,000 context rows. `delta` is
tree minus TabFM, and `CI` is the 95% interval on that gap. Every interval includes zero, so
every line ends `indistinguishable`. The script rewrites `results/showdown.json` with the
same bytes it already holds.

To run one model yourself, give `scripts/run_one.py` five arguments: a model key, a dataset,
the number of context rows, the number of test rows, and an output path. The model keys are
`logreg`, `xgboost`, `lightgbm`, `catboost`, `xgboost_ord`, `lightgbm_ord`, `catboost_ord`,
`tabicl`, `tabdpt`, `tabfm`, `tabfm_single` and `tabfm_ensemble`. The `_ord` keys hand a
tree its category columns as plain integer codes. A `tabpfn` key exists too. That model
cannot run here (see what ran and what did not). The datasets are `credit_g`, `heloc`,
`bank_marketing` and `kdd_appetency`. Make the output folder first. If the folder does not exist, the run does
not crash. It writes an error into the JSON and saves no predictions. Point the output
outside `results/`. The script overwrites whatever path you give it, and `results/` holds
the committed result files you would otherwise destroy. Its first call on a dataset downloads
it from OpenML,
a public library of machine-learning datasets, and caches it under `data/openml/`.

```bash
mkdir -p out
.venv/bin/python scripts/run_one.py logreg credit_g 700 300 out/logreg_credit_g.json
```

```
{"model": "logreg", "dataset": "credit_g", "n_context": 700, "status": "ok"}
```

The JSON file holds the score and the timing. Its `auc` is 0.66328, the same value as the
committed probe for logistic regression on German credit. The `.npz` file beside it holds
the predicted probabilities, which is what the tie test reads.

The full sweep runs the nine fast model keys on all four datasets: logistic regression, each
of the three trees run twice with the two ways of handing it category columns, plus TabICL
and TabDPT.

```bash
.venv/bin/python scripts/sweep.py
```

Every probe that is already committed prints `[cached]` and is skipped, so on a fresh clone
you get 36 lines like this one and nothing re-runs:

```
-> credit_g         logreg     ... ok  auc=0.6633  pred=0.0s  rss=0.19GB  [cached]
```

`auc` is the score, `pred` is the prediction time in seconds, and `rss` is the peak memory
in GB. `--force` re-runs a probe and overwrites the committed result. `--models tabfm` adds
the slow model. TabFM downloads its weights on first use, a multi-gigabyte download, and
`sweep.py` prints a projected run time before TabFM starts. Budget memory as well. The
probe files record the four default TabFM runs peaking between 10.19 and 12.57 GB, and
TabICL at 15.23 GB on the wide telecom table.

For a guided version of all this, [`examples/tutorial.ipynb`](examples/tutorial.ipynb) loads a
dataset, runs the classical models live and reproduces the four ties from the committed
predictions, with every output saved in the notebook.

The check that guards this file:

```bash
.venv/bin/python scripts/verify_readme_claims.py
```

It uses only the standard library, so the system `python3` works too. The last line on a
good run is `every recomputed claim matches the artifacts`.

One known trap. Do not re-run `scripts/compare.py` on a checkout unless you mean to redo the
tables. Here is what goes wrong, in order:

1. The `results/` folder already holds an extra one-pass TabFM file on German credit
   (`results/probe/credit_g__tabfm_single__c700.json`, explained under the three things the
   leaderboard hides).
2. A re-run of `compare.py` would count seven models on that dataset, not the six it first
   saw.
3. That extra model changes the corrected p values and rewrites
   `results/paired_comparison.json`.
4. The verifier would then fail.

To restore the committed file, run `git checkout -- results/paired_comparison.json`. Both
notebooks end by suggesting `compare.py` followed by the verifier. Run `scripts/showdown.py`
instead.

## What it found

![The four datasets, shown as lead against margin of error: every lead sits inside its own margin, so each is a tie.](assets/accuracy.png)

`scripts/datasets.py` fetches the four datasets from OpenML. The rarer of the two outcomes
is coded as the positive, and that script applies the rule to all four datasets. On `heloc`
the rarer outcome is the loan labelled Good, so the positives there are the good loans rather
than the defaults.

| dataset | what it is | context rows | test rows | positives in test |
|---|---|---|---|---|
| `credit_g` | German credit risk | 700 | 300 | 90 |
| `heloc` | Home equity credit lines (loans against a house), real loan-underwriting records from FICO | 1,000 | 1,000 | 478 |
| `bank_marketing` | Portuguese bank telemarketing outcomes | 1,000 | 1,000 | 117 |
| `kdd_appetency` | Orange telecom customer records (appetency is the target column each model predicts, a yes or no label on each customer) | 1,000 | 3,000 | 53 |

Every model got the same 700 or 1,000 context rows and the same test rows. For each dataset
the table shows the top scorer, its lead over the control, and whether that lead passed the
tie test after Holm correction.

| dataset | best model | vs strongest classical | survives Holm |
|---|---|---|---|
| `heloc` | TabFM 0.8247 | +0.0230 over logistic regression | yes, p = 0.0192 |
| `bank_marketing` | TabFM 0.9420 | +0.0194 over CatBoost | yes, p = 0.0288 |
| `credit_g` | TabDPT 0.7779 | +0.0244 over XGBoost | no, p = 0.7504 |
| `kdd_appetency` | TabFM 0.7164 | +0.0238 over logistic regression | no, p = 1.0000 |

Read the `heloc` row like this. TabFM's AUC was 0.8247, which is 0.0230 above logistic
regression at 0.8017. The corrected p is 0.0192. Anything under 0.05 counts as a real lead.
On `credit_g` and `kdd_appetency` the p values of 0.7504 and 1.0000 mean the lead could
easily be chance. TabICL also beat the control on the same two datasets: 0.8217 on `heloc`
(+0.0199, p = 0.0460) and 0.9360 on `bank_marketing` (+0.0133, p = 0.0288). The tally is
4 wins that clear Holm correction across 2 datasets. Nothing survives on the other two.

The foundation models win that table. But the row cap came from TabFM's cost, not from the
problem. A tree in production has no such cap. So I gave the tree the data it would actually
have in production: I trained the same LightGBM on every training row each dataset has and
scored it on the identical test rows (`scripts/context_reference.py`). German credit has
only 700 training rows in total. The cap changed nothing there, so that dataset is left out
of this table.

| dataset | LightGBM at the cap | LightGBM on all rows | rows | gain |
|---|---|---|---|---|
| `heloc` | 0.7824 | 0.8149 | 9,459 | +0.0326 |
| `bank_marketing` | 0.8930 | 0.9421 | 44,211 | +0.0491 |
| `kdd_appetency` | 0.6446 | 0.7741 | 47,000 | +0.1295 |

The tie test between that uncapped tree and TabFM is what `scripts/showdown.py` reruns.
Scores here carry five decimals. The TabFM column is the same set of runs as the first
table, rounded differently, so 0.8247 there and 0.82474 here are one measurement. The TabFM
column is labelled 1,000 rows, but German credit has only 700 training rows in total, so
both TabFM and the tree used 700 there.

| dataset | TabFM (1,000 rows) | tree (all rows) | tree rows | 95% CI on the gap | p |
|---|---|---|---|---|---|
| German credit | 0.76656 | 0.72259 | 700 | [-0.0870, +0.0000] | 0.0504 |
| FICO home equity | 0.82474 | 0.81495 | 9,459 | [-0.0235, +0.0039] | 0.1632 |
| Bank marketing | 0.94205 | 0.94205 | 44,211 | [-0.0077, +0.0076] | 0.9956 |
| Telecom appetency | 0.71635 | 0.77414 | 47,000 | [-0.0079, +0.1315] | 0.0868 |

Four datasets, four ties. Every interval includes zero. On bank marketing the two scores are
identical to the limit of floating point, and the 0.9421 in the gain table is this same
0.94205 rounded to four places.

Drawing level cost TabFM 4,313 seconds per thousand predictions on bank marketing (the cost
table below, rounded). Against logistic regression, the fastest classical model on that
dataset, that is 11,712,158 times the cost. Here is how that ratio is built. The 0.0005 in
the cost table is rounded too coarsely to divide by. The real median timing is 0.000368
seconds per thousand predictions, stored to more digits than shown here. Dividing 4,312.7678
by that stored figure gives 11,712,158. A single logistic-regression call runs well under a
millisecond. That is too short to time once and trust. So `scripts/time_baselines.py` times
the call 200 times and stores the median in `results/baseline_timing.json`. Every cost ratio
in this repo uses that stored figure.

## What it costs

![Scoring one million customers: about half a second with classical tooling, about fifty days with the foundation model.](assets/cost.png)

Seconds per 1,000 predictions on a CPU, one timed call per probe. For each tree the cell is
the run the comparison used, whichever of its two forms (told which columns are categories,
or given them as integer codes) scored higher on that dataset.

| model | credit_g | heloc | bank_marketing | kdd_appetency |
|---|---|---|---|---|
| TabFM | 5,459.3300 | 4,627.9978 | 4,312.7678 | 8,702.0829 |
| TabICL | 5.5714 | 3.4587 | 2.8460 | 38.8713 |
| TabDPT | 0.6331 | 0.4047 | 0.3768 | 0.2267 |
| LightGBM | 0.0135 | 0.0091 | 0.0097 | 0.0065 |
| XGBoost | 0.0082 | 0.0033 | 0.0040 | 0.0037 |
| CatBoost | 0.0033 | 0.0011 | 0.0020 | 0.0038 |
| Logistic regression | 0.0015 | 0.0005 | 0.0005 | 0.0016 |

Scoring one million bank-marketing customers takes about fifty days at TabFM's rate and
about half a second at logistic regression's. TabICL sits in the seconds, TabDPT under a
second, and the trees in thousandths of a second.

The gap is built in. A tree learns from your data once and then predicts cheaply for as long
as you like. A foundation model never learns anything and re-reads your whole table on every
call, so your data becomes the running cost instead of the training input.

## Three things the leaderboard hides

![Three things the leaderboard leaves out: a non-commercial licence, an unstable category, and a default that quietly runs 32 models.](assets/catches.png)

The model that tops the leaderboard, TabFM, ships its weights under a non-commercial
licence. The strongest result here is the one a business is least able to use, and no
leaderboard has a column for that. Check the licence before your data scientists benchmark,
not after. The details are in [NOTICE.md](NOTICE.md).

"Tabular foundation model" is not really one category, at least on this data. TabFM scored
0.7164 on the telecom data. TabICL scored 0.5867. Two tools sold under the same label landed
0.13 AUC apart on the same rows. The tree runs that stood for each library were
closer to each other than that: CatBoost, the best of them, scored 0.6699 and LightGBM, the
worst, scored 0.6446.

TabFM's shipped default is a 32-member ensemble, not the single forward pass its announcement
describes. A forward pass is one run through the model. An ensemble runs the model several
times with different settings and averages the answers. On `credit_g` the default
took 1637.799 seconds to predict the 300 test rows and scored 0.76656. One pass took 51.297
seconds and scored 0.76455. The default therefore costs about 32 times as much for a gain of
0.0020 AUC, which is well inside the noise on 300 rows.

## The verdict

![Time, cost, quality: a draw on quality, a win on development time, a loss on serving cost.](assets/verdict.png)

Any delivery lead knows the frame: time, cost, quality. On quality the foundation model
draws once the comparison is fair. On serving cost, the cost of producing predictions once
the model is in use, it loses by orders of magnitude. Development time is where I think it
wins. There is nothing to train or retrain. Nothing in this repo measures that, though, so
take it as my judgement rather than a finding. Treat today's tabular foundation models as a
prototyping accelerator, not a production system.

## How it works

Nothing here trains a foundation model. For TabFM, TabICL and TabDPT the training step (the
`fit` call) learns no weights. It stores and pre-processes your rows. The prediction step
(`predict_proba`) reads them again and works out an answer for each test row fresh on every
call. Cost therefore grows with the size of your training data. For the same reason the
head-to-head has to cap everyone at the same small training set, and the uncapped table
above is the comparison a business would face.

Each model on each dataset runs in its own process (`scripts/run_one.py`) and writes one
probe JSON under `results/probe/`, with its predicted probabilities beside it as `.npz`. A
crash or out-of-memory kill is recorded as a result instead of ending the sweep. Peak memory
belongs to one model rather than to whatever ran before it.

Every model on a dataset sees the same split, meaning the same division of rows into
context rows and test rows. The split is fixed by seed 1234, the starting number for the
random picker, so it comes out the same on every run. The rarer of the two outcomes is
always coded as 1, the positive. That keeps AUC meaning the same thing on every dataset.

A few protocol choices move the numbers, and all of them are stated in the open.

Each tree library ran twice. Once it was told which columns hold categories (labels such as
"married" or "radio/tv"), and once it was given those columns as plain integer codes. The
better run stands for that library. Which form a tree gets is not cosmetic: telling
CatBoost which columns were categories on `bank_marketing` raised CatBoost from 0.9118 to
0.9227 and cut TabFM's lead over CatBoost from +0.0303 to +0.0194, about a third.

The control is picked after seeing the scores. Picking the strongest classical model shrinks
any foundation-model lead to its smallest and stretches any foundation-model loss to its
largest. The choice is cautious when the foundation model wins and harsh when it loses. So
TabICL's and TabDPT's losses to logistic regression on the telecom data are worst cases.
Those losses were 0.1059 and 0.0965 AUC, recorded in `results/paired_comparison.json`.

Holm runs within a dataset, not across the four. The tie test says how sure we can be about
this one division of the data into context and test rows. It says nothing about how the
result would change if the rows were divided differently.

The trees are untuned. They run on fixed, sensible settings with no search over those
settings. Tuning the trees would only shrink the foundation model's leads, so the ties
reported here are, if anything, generous to the foundation model.

```
scripts/datasets.py          the four real datasets, fetched from OpenML and cached
scripts/prep.py              the shared split, plus the three ways columns are handed to a model
scripts/run_one.py           one model, one dataset, one subprocess
scripts/sweep.py             the real-data sweep and its per-dataset protocol
scripts/compare.py           paired bootstrap against a control, Holm corrected
scripts/context_reference.py what a tree scores without the context cap
scripts/showdown.py          the tie test between TabFM and the uncapped tree
scripts/feature_baseline.py  how much of the target (the thing predicted) one column alone recovers
scripts/time_baselines.py    median-of-200 timing so cost ratios are not artefacts
scripts/smoke_test.py        a quick check on fake data that every model runs at all, kept apart from the real sweep
scripts/summarise.py         reads the probe JSONs, prints the tables
scripts/verify_readme_claims.py  recomputes every measured figure here and fails on drift
```

## What ran and what did not

| model | version | weights licence | ran |
|---|---|---|---|
| TabFM | 1.0.1 | non-commercial | yes |
| TabICL | 2.1.1 | BSD-3-Clause | yes |
| TabDPT | 1.2.0 | Apache-2.0 | yes |
| TabPFN | 8.4.0 | non-commercial by default | **no** |
| Nori | 0.18.2 | Apache-2.0 | **not applicable** |

TabPFN refuses to download weights without an interactive licence acceptance, so it cannot
be installed unattended and is absent from every result here. Nori 0.18.2 documents only
regression, predicting a number, in its public API, which defines
`Task = Literal["regression", "reg"]`. The four datasets here are classification, predicting
a yes or no label, so I did not run it on them.

Every TabFM cell in the cost table is a measured run. The `kdd_appetency` probe, at
8,702.0829 seconds per thousand for 3,000 predictions, is the most expensive single run in
the repo.

## Limits

This is a personal experiment, shared for education and reproducibility. It is not
production advice and not a formal benchmark.

The study uses one seed, one split and a CPU. Four datasets is enough to show that the
answer changes from dataset to dataset, and not enough to rank these models in general. The
trees are untuned, as noted above. These are not TabArena-grade measurements and are not
meant to be. TabArena is a published leaderboard suite of 51 tabular datasets. Google ships
its TabArena result files alongside TabFM. All four datasets here appear in that suite
by name, though not necessarily in the same version.

`data/` is not committed, so the first run of anything that loads a dataset fetches it from
OpenML over the network. Re-running `scripts/compare.py` changes the committed comparison,
as described under "How to run".

The field is moving fast, so read the conclusion with a date on it. Tabular foundation
models are a real new capability, and what holds them back today is cost, not accuracy.
TabDPT points the way. It is retrieval-based, meaning it pulls in similar past rows to guide
each prediction, it is Apache-2.0, it predicts in under a second per thousand rows in the
cost table, and it had the top score on German credit. A
future model may well close the gap and start to replace tree pipelines for everyday
tabular work. That is not where things stand now. Today, once a gradient boosting model is
given the full dataset, it matches these models on accuracy and runs for a tiny fraction of
the cost. Treat the numbers as one careful data point and check them against your own data
before acting on them.

## Also here

- [examples/foundation-model-live.ipynb](examples/foundation-model-live.ipynb) runs TabDPT
  and TabICL live on `heloc`. It pins the libraries to one thread, which is what keeps torch
  and FAISS from crashing the notebook kernel on macOS. It needs a first-run weight
  download.
- [FINDINGS.md](FINDINGS.md) is the full write-up.
- [RESEARCH-NOTES.md](RESEARCH-NOTES.md) is the lab journal, including the dead ends.
- [DATASET-CHOICE.md](DATASET-CHOICE.md) explains why a fifth dataset, a synthetic one tried
  before these four, was dropped from the comparison and kept only as a check that the code
  runs.
- [NOTICE.md](NOTICE.md) separates the code licence from the weight licences. The weights are
  not vendored and TabFM's cannot be used commercially.

## Author

Written by [Satsawat Natakarnkitkul](https://satsawat.ai), a data and AI practitioner in
ASEAN. Newsletter: [AI in Practice](https://satsawat.ai/#newsletter).

Companion repositories: [tsfm-bakeoff](https://github.com/netsatsawat/tsfm-bakeoff) asks the
same question of time-series foundation models, and
[agent-failure-lab](https://github.com/netsatsawat/agent-failure-lab) does it for compound
error in agents.
