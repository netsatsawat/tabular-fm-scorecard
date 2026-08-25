# What four real datasets say about tabular foundation models

Seven models, four observed datasets, identical splits, paired bootstrap with Holm
correction. Every measured number below is recomputed from the committed artifacts by
[`scripts/verify_readme_claims.py`](scripts/verify_readme_claims.py), which fails if any of
them drift. Figures attributed to TabArena or to Google are external citations and are
marked as such where they appear.

## The setup

| dataset | what it is | context | test rows | positives | features |
|---|---|---|---|---|---|
| `credit_g` | German credit risk | 700 | 300 | 90 | 20 |
| `heloc` | FICO home equity lines | 1000 | 1000 | 478 | 23 |
| `bank_marketing` | Portuguese bank telemarketing | 1000 | 1000 | 117 | 16 |
| `kdd_appetency` | Orange telecom, 64% missing | 1000 | 3000 | 53 | 212 |

All four come from OpenML, and all four appear by name in TabArena's 51-dataset suite,
checked against the result parquets Google ships with TabFM. The OpenML versions pinned
here are not necessarily the ones TabArena runs, so the overlap is in the underlying data
rather than in the task definition.

Every model on a dataset sees the same rows, and the split is a pure function of the seed,
so the bootstrap below is genuinely paired.

The three tree libraries each ran twice, once with native categorical handling and once on
ordinal codes, and the stronger run stands for that library. This matters more than it
sounds. Giving CatBoost its categorical features on `bank_marketing` moved it from 0.9118 to
0.9227, which cut the headline gap it is the control for almost in half.

The control on each dataset is the strongest classical model after that. Picking it by
looking at the scores maximises the reported gap in whichever direction it points. That is
conservative when a foundation model wins and the opposite when it loses, so the negative
result on `kdd_appetency` should be read as an upper bound on the deficit, not a fair
estimate of it.

Holm correction is applied within each dataset, across the models compared against that
dataset's control. It is not applied across the four datasets.

## Foundation models win two of the four, by about two points

| dataset | model | AUC | vs control | 95% CI | p after Holm |
|---|---|---|---|---|---|
| `heloc` | TabFM | 0.8247 | +0.0230 over logistic regression | [+0.0079, +0.0387] | 0.0192 |
| `heloc` | TabICL | 0.8217 | +0.0199 | [+0.0047, +0.0357] | 0.0460 |
| `bank_marketing` | TabFM | 0.9420 | +0.0194 over CatBoost | [+0.0045, +0.0355] | 0.0288 |
| `bank_marketing` | TabICL | 0.9360 | +0.0133 | [+0.0037, +0.0240] | 0.0288 |

That is 4 confirmed wins across 2 datasets. Nothing survives on `credit_g` or
`kdd_appetency`.

Two points of AUC is worth money on a credit book. It is also a long way from the picture
the leaderboards paint.

## The comparison is capped at a size the foundation models chose

Every model above was trained on 700 or 1,000 rows. That cap exists because TabFM's cost
grows with context multiplied by query, and larger runs were not affordable on a CPU. A
tree has no such constraint in production.

So the same LightGBM was run again on every training row that is not in the held-out split,
scored on the identical test rows ([`scripts/context_reference.py`](scripts/context_reference.py)):

| dataset | capped | rows | uncapped | rows | gain |
|---|---|---|---|---|---|
| `credit_g` | 0.7226 | 700 | 0.7226 | 700 | +0.0000 |
| `heloc` | 0.7824 | 1000 | 0.8149 | 9459 | +0.0326 |
| `bank_marketing` | 0.8930 | 1000 | 0.9421 | 44211 | +0.0491 |
| `kdd_appetency` | 0.6446 | 1000 | 0.7741 | 47000 | +0.1295 |

Read the `bank_marketing` row against the table above it. LightGBM on all 44,211 rows
scores 0.9421. TabFM, on 1,000 context rows, scored 0.9420.

The gradient boosting machine catches the foundation model exactly, using data it already
has, at under a hundredth of a second per thousand predictions.

On `kdd_appetency` the uncapped tree reaches 0.7741 and every foundation model sits below
0.60. On `credit_g` nothing changes, because the dataset only has 1,000 rows and the cap
was never binding.

This does not make the wins above fake. They are real at 1,000 rows, and there are settings
where 1,000 rows is all you have. It does mean the win is a statement about a regime, and
the regime was set by the constraint the foundation model brings, not by the problem.

## The largest margin is the one that cannot be confirmed

On `credit_g`, TabDPT scores 0.7779 against XGBoost's 0.7535. That gap of +0.0244 is larger
than any confirmed win in this study. It does not survive: the Holm-adjusted p is 0.7504
and the interval runs from -0.0121 to +0.0635.

The reason is the test set. `credit_g` holds 1,000 rows in total, so a held-out third
leaves 300 rows carrying 90 positives. Too few to separate models two points apart.

That is worth dwelling on. The dataset most often reached for to argue that foundation
models win on small data is too small to demonstrate it. A benchmark that reported the
point estimate and stopped would have called this a win.

## On the telecom data, the foundation models disagree with each other

`kdd_appetency` is Orange's customer records. It has 212 features, 64% of its cells are
missing, and 1.78% of customers convert.

| model | AUC |
|---|---|
| TabFM | 0.7164 |
| Logistic regression | 0.6926 |
| CatBoost | 0.6699 |
| XGBoost | 0.6693 |
| LightGBM | 0.6446 |
| TabDPT | 0.5961 |
| TabICL | 0.5867 |

TabFM has the highest score and does not separate from anything: +0.0238 over logistic
regression, with an interval from -0.0478 to +0.1008 and an adjusted p of 1.0000. With 53
positives in the test split, almost nothing here is distinguishable. What does survive is
at the bottom, where TabICL's deficit of -0.1059 clears Holm at p = 0.0408.

The interesting part is the spread inside the foundation models. TabFM at 0.7164 and
TabICL at 0.5867 are 0.13 AUC apart on the same rows, which is larger than the distance
between the best and worst tree. Treating "tabular foundation models" as one category
does not survive contact with this dataset.

TabFM took 7 hours 15 minutes to produce those 3,000 predictions, or 8,702 seconds per
thousand. The uncapped LightGBM reaches 0.7741 on the same test rows, above TabFM, using
47,000 training rows and 0.0065 seconds per thousand.

Why the two in-context models that lose, lose, follows from how they work. They do not
train. Your rows go in as context and the answer is worked out from them fresh on every
call, which means that at a 1.78% positive rate in a 1,000-row context the model is
reasoning from roughly 18 conversions and nothing else.

A tree sees the same 18. It gets to keep what it learned from them, and it can be handed
47,000 rows instead.

Rare events are not an edge case in business data. Fraud, churn, default and conversion are
all rare by construction, and they are most of what a commercial data science team is asked
to predict.

## What it costs

Prediction time in seconds per 1,000 rows, from a single timed call inside each probe. For
each tree library this is the run the comparison used, which is whichever representation
scored higher on that dataset.

| model | credit_g | heloc | bank_marketing | kdd_appetency |
|---|---|---|---|---|
| TabFM | 5,459.3300 | 4,627.9978 | 4,312.7678 | 8,702.0829 |
| TabICL | 5.5714 | 3.4587 | 2.8460 | 38.8713 |
| TabDPT | 0.6331 | 0.4047 | 0.3768 | 0.2267 |
| LightGBM | 0.0135 | 0.0091 | 0.0097 | 0.0065 |
| XGBoost | 0.0082 | 0.0033 | 0.0040 | 0.0037 |
| CatBoost | 0.0033 | 0.0011 | 0.0020 | 0.0038 |
| Logistic regression | 0.0015 | 0.0005 | 0.0005 | 0.0016 |

A single call on a tree takes single-digit milliseconds, which is close enough to the
timer's resolution that a ratio built from one call would be an artefact.
[`scripts/time_baselines.py`](scripts/time_baselines.py) takes the median of 200 repeats
after a warm-up, and every ratio quoted here comes from that artifact rather than from the
table above.

Against the fastest baseline on each dataset, measured that way, TabFM costs 5,628,988
times as much on `credit_g`, 11,744,995 on `heloc`, 11,712,158 on `bank_marketing` and
11,681,589 on `kdd_appetency`.

### What the 32-member default buys

Google describes TabFM as returning predictions in a single forward pass. The constructor
defaults to `n_estimators=32`, so the shipped default is a 32-member ensemble. Running it
both ways on `credit_g`:

| configuration | ROC AUC | predict (s) |
|---|---|---|
| `n_estimators=32` (the default) | 0.76656 | 1637.799 |
| `n_estimators=1` (what the description says) | 0.76455 | 51.297 |

The default costs 31.9 times as much and gains 0.0020 AUC. On a 300-row test split that
difference is far inside the noise, and no significance is claimed for it.

So a latency figure quoted as a single forward pass understates the shipped configuration
by a factor of 32, and the thing that factor buys is not measurable here.

Peak memory follows the same shape. TabFM held between 10.19 and 12.57 GB, TabICL reached
15.23 GB on the wide telecom table, and no tree exceeded 0.67 GB.

TabDPT deserves its own line. It is Apache-2.0, it predicts in under a second per thousand
rows, it never exceeded 1.01 GB, and it took `credit_g` outright. TabICL is also
commercially licensed under BSD-3-Clause and also ran on CPU throughout, so TabDPT is not
the only deployable option here, but it is the cheap one.

## So should you use one

If your problem is small, dense, reasonably balanced, and you genuinely cannot get more
than about a thousand labelled rows, a foundation model is a real improvement. `heloc` and
`bank_marketing` show that, and the improvement survives correction.

If you can get more rows, get more rows. The uncapped tree closed the entire gap on
`bank_marketing` and opened a large one on `kdd_appetency`. More data beat a better model
in three of the four cases here, and it costs nothing at prediction time.

If your events are rare or your table is wide and sparse, the trees win now and the
simplest model is competitive. Test before believing otherwise.

If you serve predictions at volume, accuracy stops being the question. Nothing here
justifies four thousand seconds per thousand rows.

One constraint sits above all of it. TabFM's weights are non-commercial, so the two wins it
took cannot be carried into a commercial deployment at all. The strongest model in this
study is the one you are least able to use.

## What would change the answer

**One seed and one split.** Every result is a single partition. The paired bootstrap
quantifies uncertainty within that split, not across resplits. TabArena's published
protocol runs up to thirty splits per dataset.

**Untuned trees.** Each library gets fixed sensible settings and its better of two
representations. Neither gets a hyperparameter search. TabArena's protocol gives each
family one default plus 200 sampled configurations and then ensembles them, and their
tuned trees beat their untuned ones by a wide margin. A tuned baseline would narrow every
gap reported here.

**A small Holm family.** Six models compared against a control on each dataset. A larger
field would tighten nothing and loosen the adjusted p-values.

**CPU only.** TabArena reports TabFM at 6.985 seconds per 1,000 rows on a GPU. The figures
here are roughly a thousand times that. The ordering of the cost column would not change,
because the trees would still be orders of magnitude cheaper, but the magnitude would.

**Four datasets.** Enough to show the answer varies by dataset. Not enough to say which
kind of dataset predicts which answer.
