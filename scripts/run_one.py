"""Run exactly one (model, context size) probe and write a JSON result.

One model per process on purpose. An out-of-memory kill, a segfault or a native
crash takes down this process and nothing else, and peak RSS measured here belongs
to this model alone rather than to whatever ran before it.

Usage:
    python scripts/run_one.py <model_key> <dataset> <n_context> <n_test> <out.json>

Predicted probabilities are written alongside the JSON as a .npz, because a paired
comparison between two models needs their predictions on identical rows, not their
summary scores.
"""

import json
import platform
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from prep import as_categorical, encode_for_gbdt, load_split

SEED = 1234


def peak_rss_gb():
    """Peak resident set size for this process, in GB.

    ru_maxrss is bytes on macOS and kilobytes on Linux.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024**3 if sys.platform == "darwin" else raw / 1024**2


# --------------------------------------------------------------------------- #
# Adapters. Each returns (probabilities_for_class_1, extra_metadata).
# Foundation models get the raw frame; tree models get the encoded one.
# --------------------------------------------------------------------------- #

def run_tabfm(Xtr, ytr, Xte, preset="default"):
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as backend

    model = backend.load(model_type="classification")
    ctor = TabFMClassifier.ensemble if preset == "ensemble" else TabFMClassifier
    kw = {}
    if preset == "single":
        # Google describes TabFM as one forward pass. The constructor defaults to
        # n_estimators=32, so 'default' is already a 32-member ensemble. Running
        # with one member is what the marketing describes, and the gap between the
        # two is the cost the marketing omits.
        kw["n_estimators"] = 1
    clf = ctor(model=model, random_state=SEED, **kw)

    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = np.asarray(clf.predict_proba(Xte))
    t_pred = time.perf_counter() - t0

    meta = {
        "n_estimators": getattr(clf, "n_estimators", None),
        "max_num_rows": getattr(clf, "max_num_rows", None),
        "max_num_features": getattr(clf, "max_num_features", None),
        "classes_": [str(c) for c in getattr(clf, "classes_", [])],
    }
    return proba[:, 1], t_fit, t_pred, meta


def run_tabpfn(Xtr, ytr, Xte):
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(random_state=SEED)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = np.asarray(clf.predict_proba(Xte))
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {"device": str(getattr(clf, "device_", "?"))}


def run_tabicl(Xtr, ytr, Xte):
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(random_state=SEED)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = np.asarray(clf.predict_proba(Xte))
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


def run_tabdpt(Xtr, ytr, Xte):
    from tabdpt import TabDPTClassifier

    # TabDPT takes numeric arrays, so it uses the encoded frame like the trees do.
    clf = TabDPTClassifier()
    t0 = time.perf_counter()
    clf.fit(Xtr.to_numpy(dtype="float64"), ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = np.asarray(clf.predict_proba(Xte.to_numpy(dtype="float64")))
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


def run_xgboost(Xtr, ytr, Xte, cat_cols=()):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
        colsample_bytree=0.9, eval_metric="auc", random_state=SEED, n_jobs=-1,
        enable_categorical=True, tree_method="hist",
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


def run_lightgbm(Xtr, ytr, Xte, cat_cols=()):
    from lightgbm import LGBMClassifier

    # LightGBM picks up pandas category dtype on its own. The constructor argument
    # wants column indices or "name:"-prefixed strings and rejects plain names, so
    # the dtype is the interface here.
    clf = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
        colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


def run_catboost(Xtr, ytr, Xte, cat_cols=()):
    import pandas as pd
    from catboost import CatBoostClassifier

    # CatBoost wants categorical features as strings and rejects NaN in them. It
    # handles a level it never saw in training on its own, so unlike XGBoost and
    # LightGBM the test frame is not forced onto the training category set.
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("string").fillna("__missing__").astype(str)
        Xte[c] = Xte[c].astype("string").fillna("__missing__").astype(str)

    clf = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, random_seed=SEED,
        verbose=0, allow_writing_files=False,
        cat_features=list(cat_cols) or None,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


def run_xgboost_ord(Xtr, ytr, Xte):
    """XGBoost on ordinal codes, with categorical handling switched off."""
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
        colsample_bytree=0.9, eval_metric="auc", random_state=SEED, n_jobs=-1,
        tree_method="hist",
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


def run_logreg(Xtr, ytr, Xte):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    return proba[:, 1], t_fit, t_pred, {}


# Which representation each model receives. Recorded in every probe so the choice
# is visible in the artifact rather than only in this file.
#   raw       - the untouched DataFrame. The in-context models do their own encoding.
#   category  - pandas category dtype plus the column names, for the tree libraries.
#   numeric   - ordinal codes as float64, for models whose estimator takes an array.
RAW_FRAME = {"tabfm", "tabfm_single", "tabfm_ensemble", "tabpfn", "tabicl"}
CAT_FRAME = {"xgboost", "lightgbm", "catboost"}
# Each tree library also runs on plain ordinal codes. Native categorical handling
# is not always the better choice: on a column with 836 levels and 1,000 training
# rows it memorises noise. Running both and letting the control be whichever wins
# is how the baseline stays strong without per-dataset tuning.
ORD_FRAME = {"xgboost_ord", "lightgbm_ord", "catboost_ord"}

RUNNERS = {
    "tabfm": lambda a, b, c: run_tabfm(a, b, c, "default"),
    "tabfm_single": lambda a, b, c: run_tabfm(a, b, c, "single"),
    "tabfm_ensemble": lambda a, b, c: run_tabfm(a, b, c, "ensemble"),
    "tabpfn": run_tabpfn,
    "tabicl": run_tabicl,
    "tabdpt": run_tabdpt,
    "xgboost": run_xgboost,
    "lightgbm": run_lightgbm,
    "catboost": run_catboost,
    "xgboost_ord": lambda a, b, c: run_xgboost_ord(a, b, c),
    "lightgbm_ord": lambda a, b, c: run_lightgbm(a, b, c, ()),
    "catboost_ord": lambda a, b, c: run_catboost(a, b, c, ()),
    "logreg": run_logreg,
}


def frame_kind(key):
    if key in RAW_FRAME:
        return "raw"
    if key in CAT_FRAME:
        return "category"
    return "numeric"


CLASSICAL = {"logreg"} | CAT_FRAME | ORD_FRAME


def main():
    key, dataset = sys.argv[1], sys.argv[2]
    n_context, n_test = int(sys.argv[3]), int(sys.argv[4])
    out_path = Path(sys.argv[5])
    # Create the output folder up front. The .npz is written inside the try block below,
    # so without this a fresh output folder makes that write throw and the run is recorded
    # as status="error" even though the model ran fine.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "model": key,
        "dataset": dataset,
        "n_context_requested": n_context,
        "n_test_requested": n_test,
        "seed": SEED,
        # Record the compute class, not the host. The OS version and architecture
        # identify a specific machine and do not belong in a committed artifact; "cpu"
        # is the load-bearing fact for the cost numbers.
        "device": "cpu",
        "python": platform.python_version(),
        "status": "unknown",
    }

    try:
        Xtr, ytr, Xte, yte = load_split(dataset, SEED, n_context, n_test)
        record["n_context"] = int(len(ytr))
        record["n_test"] = int(len(yte))
        record["positive_rate_train"] = round(float(ytr.mean()), 4)
        record["positive_rate_test"] = round(float(yte.mean()), 4)
        record["n_positives_test"] = int(yte.sum())
        record["n_features"] = int(Xtr.shape[1])

        kind = frame_kind(key)
        record["frame"] = kind
        cat_cols = []
        if kind == "category":
            Xtr, Xte, cat_cols = as_categorical(Xtr, Xte)
        elif kind == "numeric":
            Xtr, Xte = encode_for_gbdt(Xtr, Xte)
        record["n_categorical_columns"] = len(cat_cols)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn = RUNNERS[key]
            p, t_fit, t_pred, meta = (
                fn(Xtr, ytr, Xte, cat_cols) if kind == "category" else fn(Xtr, ytr, Xte)
            )
            record["warnings"] = sorted({str(w.message)[:200] for w in caught})[:12]

        p = np.asarray(p, dtype="float64")
        np.savez_compressed(
            out_path.with_suffix(".npz"), proba=p, y_true=yte.astype("int64"),
        )

        record.update(
            auc=round(float(roc_auc_score(yte, p)), 5),
            fit_seconds=round(t_fit, 3),
            predict_seconds=round(t_pred, 3),
            predict_seconds_per_1k=round(t_pred / len(yte) * 1000, 4),
            peak_rss_gb=round(peak_rss_gb(), 2),
            proba_min=round(float(p.min()), 6),
            proba_max=round(float(p.max()), 6),
            proba_mean=round(float(p.mean()), 6),
            n_unique_proba=int(np.unique(np.round(p, 8)).size),
            meta=meta,
            status="ok",
        )
    except Exception as exc:
        record.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc)[:600],
            traceback=traceback.format_exc()[-1500:],
            peak_rss_gb=round(peak_rss_gb(), 2),
        )

    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record.get(k) for k in ("model", "dataset", "n_context", "status")}))


if __name__ == "__main__":
    main()
