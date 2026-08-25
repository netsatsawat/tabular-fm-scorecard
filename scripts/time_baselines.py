"""Measure GBDT prediction time precisely enough to divide by.

A single predict call on 2,000 rows takes 1-6 ms, which is close enough to the
timer's resolution that the ratio against a foundation model would be an artefact.
This takes the median of 200 repeats after a warm-up call, and writes the result
so the README's 'x times slower' figures have a runnable path behind them.
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np

from prep import encode_for_gbdt, load_split

REPO = Path(__file__).resolve().parent.parent
SEED = 1234
REPEATS = 200


def build():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, eval_metric="auc", random_state=SEED, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
            colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6, random_seed=SEED,
            verbose=0, allow_writing_files=False,
        ),
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
    }


PROTOCOL = {
    "ev_battery": (2000, 2000),
    "credit_g": (700, 300),
    "heloc": (1000, 1000),
    "bank_marketing": (1000, 1000),
    "kdd_appetency": (1000, 3000),
}


def main():
    out = {"repeats": REPEATS, "seed": SEED, "datasets": {}}
    for dataset, (n_context, n_test_req) in PROTOCOL.items():
        Xtr, ytr, Xte, _ = load_split(dataset, SEED, n_context, n_test_req)
        Xtr, Xte = encode_for_gbdt(Xtr, Xte)
        n_test = len(Xte)
        block = {"n_context": int(len(ytr)), "n_test": int(n_test), "models": {}}
        print(f"\n{dataset}  ({len(ytr)} context, {n_test} test)")
        for name, model in build().items():
            model.fit(Xtr, ytr)
            model.predict_proba(Xte)                   # warm-up, not timed
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                model.predict_proba(Xte)
                times.append(time.perf_counter() - t0)
            med = float(np.median(times))
            block["models"][name] = {
                "median_seconds_per_call": round(med, 8),
                "seconds_per_1k": round(med / n_test * 1000, 8),
                "p05_seconds": round(float(np.percentile(times, 5)), 8),
                "p95_seconds": round(float(np.percentile(times, 95)), 8),
            }
            print(f"  {name:10s} {med * 1000:8.3f} ms/call   "
                  f"{med / n_test * 1000:.6f} s/1K")
        out["datasets"][dataset] = block

    path = REPO / "results" / "baseline_timing.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
