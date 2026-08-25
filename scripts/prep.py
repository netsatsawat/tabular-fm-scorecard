"""Shared data preparation. Every model must see the identical split.

Imported by run_one.py inside whatever process is running a model, so it stays
light: pandas, numpy and the dataset registry only, no model libraries.

The split is a pure function of (dataset, seed, n_context, n_test). Two runs with
the same arguments produce byte-identical frames, which is what makes the comparison
paired rather than a set of unrelated experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# The original synthetic probe set, kept so its results stay reproducible. It is not
# in the real-data registry on purpose — see DATASET-CHOICE.md.
EV_CSV = REPO / "data" / "ev_battery_failure_dataset.csv"
EV_TARGET = "battery_failure"
EV_DROP = ["vehicle_id", "battery_serial"]


def _load_frame(dataset):
    if dataset == "ev_battery":
        df = pd.read_csv(EV_CSV).drop(columns=EV_DROP)
        return df.drop(columns=[EV_TARGET]), df[EV_TARGET].to_numpy()

    from datasets import fetch

    return fetch(dataset)


def load_split(dataset="ev_battery", seed=1234, n_context=2000, n_test=2000):
    """Returns (X_train, y_train, X_test, y_test), stratified on the target.

    n_context is the number of rows handed to the model as training data. For the
    in-context models this IS the context window, which is why it is swept rather
    than fixed. Both sizes are clipped to what the dataset can supply, and the
    actual sizes are recoverable from the returned arrays.
    """
    X, y = _load_frame(dataset)
    y = np.asarray(y).astype("int64")

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rate = len(pos) / len(y)

    # Never ask for more of either class than exists, and never leave a split with
    # a single class in it.
    budget = len(y)
    n_test = min(n_test, budget // 2)
    n_context = min(n_context, budget - n_test)

    def take(n, used):
        n_pos = int(np.clip(round(n * rate), 1, len(pos) - 1))
        avail_pos = np.setdiff1d(pos, used)
        avail_neg = np.setdiff1d(neg, used)
        n_pos = min(n_pos, len(avail_pos))
        n_neg = min(n - n_pos, len(avail_neg))
        p = rng.choice(avail_pos, size=n_pos, replace=False)
        q = rng.choice(avail_neg, size=n_neg, replace=False)
        idx = np.concatenate([p, q])
        rng.shuffle(idx)
        return idx

    test_idx = take(n_test, np.array([], int))
    train_idx = take(n_context, test_idx)

    assert not set(train_idx) & set(test_idx), "train and test overlap"
    assert len(np.unique(y[test_idx])) == 2, "test split is single-class"
    assert len(np.unique(y[train_idx])) == 2, "train split is single-class"

    return (
        X.iloc[train_idx].reset_index(drop=True), y[train_idx],
        X.iloc[test_idx].reset_index(drop=True), y[test_idx],
    )


def categorical_columns(X):
    """Column names pandas does not consider numeric."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def encode_for_gbdt(X_train, X_test):
    """Ordinal-encode non-numeric columns, mapping unseen test values to NaN.

    For models that need a purely numeric matrix and cannot be told which columns
    are categorical: the linear baseline, and TabDPT, whose estimator takes arrays.
    Prefer as_categorical() for the tree libraries, which all support categorical
    features natively and are handicapped without it.
    """
    tr, te = X_train.copy(), X_test.copy()
    for c in categorical_columns(tr):
        cats = pd.Index(tr[c].astype("string").dropna().unique())
        mapping = {v: i for i, v in enumerate(cats)}
        tr[c] = tr[c].astype("string").map(mapping).astype("float64")
        te[c] = te[c].astype("string").map(mapping).astype("float64")
    return tr, te


def as_categorical(X_train, X_test):
    """Give the tree libraries real categorical columns instead of fake numbers.

    Returns (train, test, categorical_column_names). Categories are taken from the
    training split only. A value seen only in test becomes NaN under the shared
    category set, which is the same information an unseen level carries, but the
    column keeps its categorical dtype so each library applies its own handling
    rather than treating an arbitrary integer code as an ordered magnitude.

    Missing values in categorical columns become an explicit "__missing__" level.
    CatBoost rejects NaN in a declared categorical feature, and an explicit level
    is more honest than an imputed one anyway.
    """
    tr, te = X_train.copy(), X_test.copy()
    cat_cols = categorical_columns(tr)
    for c in cat_cols:
        a = tr[c].astype("string").fillna("__missing__")
        b = te[c].astype("string").fillna("__missing__")
        levels = pd.Index(sorted(a.unique()))
        tr[c] = pd.Categorical(a, categories=levels)
        te[c] = pd.Categorical(b, categories=levels)
    return tr, te, cat_cols
