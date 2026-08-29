# Research notes

The lab journal. The dead ends and corrections are half the point, so I write them down
here instead of quietly fixing them.

## 1. The model field

Five tabular foundation models were considered. Four can do classification.

**Nori 0.18.2 cannot.** `synthefy_nori/api.py` defines `Task = Literal["regression",
"reg"]` and the package contains no reference to classification anywhere. Excluded on a
fact about the library, not a scoping decision.

**TabPFN 8.4.0 will not install unattended.** It raises `TabPFNLicenseError` rather than
downloading weights:

> TabPFN requires a one-time license acceptance to download model weights for local
> inference, but no interactive terminal is available.

Clearing it needs a browser session at `ux.priorlabs.ai`, an accepted licence, and a
`TABPFN_TOKEN` environment variable. That is worth stating, because TabPFN is the
model most often described as the open alternative, and it is the one that cannot go into
CI without a human first.

**TabFM, TabICL and TabDPT install and run** with no interaction.

## 2. One environment, not three

The time-series equivalent of this study needed three isolated virtualenvs, because
numpy 2.x and 1.26 are binary incompatible and no resolver flag fixes a C-ABI break.

The tabular field does not have that problem. `tabfm[pytorch]==1.0.1`, `tabpfn==8.4.0`,
`tabicl==2.1.1`, `tabdpt==1.2.0`, `synthefy-nori==0.18.2` and the three GBDTs resolve into
a single Python 3.12 environment, 72 packages. Only AutoGluon would need its own, and it
is not used here.

Two macOS details cost time and are worth recording. TabFM's `requirements.txt` pins
`torch==2.12.1+cpu`, and that `+cpu` local version has no macOS wheel at all: it is
published for Linux and Windows only. The pins live in the lockfile rather than in
`pyproject.toml`, whose dependencies float, so installing the package normally sidesteps
the problem entirely. Separately, `autogluon.tabular[xgboost]` depends on `xgboost-cpu`,
which publishes no macOS wheels either.

## 3. The pandas 3 worry that did not materialise

The resolver lands on pandas 3.0.5, numpy 2.5.2 and scikit-learn 1.9.0, while TabFM's own
lockfile was built against pandas 2.2.3, numpy 2.2.0 and scikit-learn 1.6.0. TabFM does
its own column-type detection, and pandas 3 is exactly where a dtype check can start
returning nothing without saying so.

TabFM 1.0.1 handles it. From `classifier_and_regressor.py`:

> Accept object dtype and the pandas string dtype (incl. the pyarrow-backed default in
> pandas>=3). Otherwise date-as-text columns load as 'string', fail this object-only gate,
> and silently fall through to categorical.

The probe still asserts on accuracy rather than on the absence of an exception, because
that was the right design whether or not this particular trap was live.

## 4. The first dataset was the wrong dataset

The probe started on a Kaggle EV battery failure set, 200,000 rows and 70 columns. It is
synthetic by its author's own description, and its target is close to a function of one
column: `capacity_loss_percent` alone scores 0.9589 ROC AUC, and the dataset's
documentation calls that column "the central degradation driver of the whole dataset".

Every model landed between 0.985 and 0.990, with logistic regression winning. The spread
across eight models was 0.005 AUC. Nothing there can rank anything.

It was kept as the runnability bed, because a saturated target is useful for exactly one
purpose: any working model must score above 0.95, so a library that fits, predicts and
returns 0.62 has failed even though it raised no exception. Details in
[DATASET-CHOICE.md](DATASET-CHOICE.md).

The real comparison moved to four observed datasets from OpenML. All four appear by name
in TabArena's 51-dataset suite, checked against the dataset column of the result parquets
Google ships with TabFM. The OpenML version pinned here is not necessarily the one TabArena
runs, so the overlap is in the underlying data and not in the task definition. An earlier
draft claimed "the exact versions TabArena uses", which was never established.

## 5. Two claims from the source, confirmed by running it

Both were read out of TabFM's code before the first run and both held.

**There is no 100-row context limit.** The README FAQ says the estimators default to
"500 features and 100 context rows". They do not. A fitted estimator reports
`max_num_rows: null`, and the constructor default is `None` in all three places the
parameter is defined. That same FAQ paragraph also names `inference_batch_size`, which
does not exist anywhere in the codebase. Anyone who read it and set `max_num_rows=100` to
match handicapped the model to a hundredth of the context it would otherwise have used.

**"Default" TabFM is a 32-member ensemble.** The fitted estimator reports
`n_estimators: 32`. Google's public description of a single forward pass is not what the
constructor builds, and any latency figure quoted as single-pass understates the default
configuration by a factor of 32.

## 6. Timer resolution nearly produced a fake number

A single GBDT prediction call on 2,000 rows takes between 1 and 6 milliseconds, which is
close enough to the timer's resolution that a ratio against a foundation model would have
been an artefact of measurement rather than a fact about the models.
[`scripts/time_baselines.py`](scripts/time_baselines.py) takes the median of 200 repeats
after a warm-up call. Every published ratio comes from that artifact. The single call
recorded inside each probe is reported as a figure but never divided by.

## 7. Why the control is chosen the way it is

Every model is scored against the strongest classical model on its dataset, chosen after
seeing the test scores.

The first version of this note said that was conservative, full stop. It is only half
true, and the part it leaves out matters. Argmax selection maximises the reported gap in whichever
direction it points: conservative when a foundation model wins, anti-conservative when it
loses. So the negative result on `kdd_appetency` is an upper bound on the deficit rather
than a fair estimate of it, and the write-up now says so.

It is left in deliberately. A foundation model that beats the best tree anyone would have
tried is making a claim worth reporting. One that only beats a weak baseline is not.

## 8. The control was crippled, and nobody noticed for a day

An adversarial review of this repository found that no tree library was ever told which
columns were categorical. No `cat_features`, no `categorical_feature`, no
`enable_categorical` anywhere in the harness. Every categorical column reached the trees as
an arbitrary integer code, which a tree then treats as an ordered magnitude.

That is not a hyperparameter choice, which is what "the trees are untuned" would cover. It
is a representation choice, and it was switched off for the model acting as the control for
the strongest result in the study. CatBoost's ordered target statistics exist for exactly
the `bank_marketing` shape: nine low-cardinality categorical columns, no missing values.

Fixing it moved CatBoost on `bank_marketing` from 0.9118 to 0.9227 and cut TabFM's margin
there from +0.0303 to +0.0194. The finding survived. It nearly halved.

Two things went wrong on the way to fixing it, both worth recording.

LightGBM began erroring with "categorical_feature is not a number, if you want to use a
column name, please add the prefix name:". The constructor argument wants indices or
prefixed strings. LightGBM reads the pandas category dtype on its own, so the dtype is the
interface and the argument was removed.

XGBoost with `enable_categorical=True` scored 0.4502 on `kdd_appetency`, below chance. That
dataset has eleven categorical columns above 100 levels and one at 836, against 1,000
training rows, so most levels appear once and the split memorises noise. Rather than tune
per dataset, every tree library now runs twice, once with native categorical handling and
once on ordinal codes, and the better run stands for the library. The representation that
won is recorded per dataset in `results/paired_comparison.json`.

Both variants of one library are the same model given two representations, not two
hypotheses, so the loser is dropped before the Holm family is counted. Leaving both in
inflated the family from six to ten and cost TabICL its `heloc` win to a correction it
should never have paid.

## 9. The context cap was measured, not assumed

The write-up claimed the 1,000-row cap was "a constraint on the foundation models
specifically". It is not. `load_split` returns one training frame and every model receives
it, so the cap trained the trees on exactly the rows it gave the foundation models as
context. The evidence quoted for the claim was itself a tree.

Worse, the number used to support it, a LightGBM score of 0.8170 on the full telecom data,
came from an ad hoc shell command and was never committed. It sat in a table beside
measured figures in the same typeface.

[`scripts/context_reference.py`](scripts/context_reference.py) now measures it properly:
the same LightGBM, on every training row outside the held-out split, scored on the
identical test rows. The result is stronger than the claim it replaced. On
`bank_marketing` the uncapped tree reaches 0.9421 against TabFM's 0.9420, and on
`kdd_appetency` it reaches 0.7741 while every foundation model sits below 0.60.

## 10. The verifier could only print "ok"

The same review found that the claim checker was decorative. Its per-model loop skipped any
figure it could not find and then asserted `True`, so a wrong number produced no check at
all. Its number scan matched `\b0\.\d{4}\b`, which ignores `0.00052` and `5,459.3300`,
so the entire cost table sat outside the only check that could fail. Worst of all, it
whitelisted every pairwise difference between any two AUCs in the artifacts. That covered
roughly 6% of the four-decimal space, and a drift from +0.0303 to +0.0300 sailed through.

The rewrite checks both directions. Whole rows are rebuilt from the artifacts and must
appear in the prose, and every result-shaped number in the prose must exist in an artifact.
It caught two of my own errors within a minute: a truncated 4,312 where the artifact says
4,312.7678, and a win count spelled as a word.
