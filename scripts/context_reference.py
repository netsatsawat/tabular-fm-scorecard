"""What a tree scores when it is not held to the foundation models' context budget.

The sweep caps every model at 700 or 1,000 training rows, because that is what TabFM
can afford on CPU. A tree has no such limit in production, so the comparison at the
capped size understates what the classical option would actually deliver.

This measures that gap on the same held-out rows the sweep uses, so the two numbers
are directly comparable: same test split, same seed, only the training size differs.

Writes results/context_reference.json.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

from prep import as_categorical, load_split
from sweep import PROTOCOL

REPO = Path(__file__).resolve().parent.parent
SEED = 1234


def tree(cat_cols):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
        colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1,
    )


def main():
    out = {"seed": SEED, "model": "lightgbm", "datasets": {}}
    for dataset, (n_context, n_test) in PROTOCOL.items():
        # The capped run, exactly as the sweep does it.
        Xc, yc, Xte, yte = load_split(dataset, SEED, n_context, n_test)

        # The uncapped run: every row that is not in the held-out split. Asking for
        # a context larger than the dataset is clipped by load_split, so this is
        # "as much training data as exists".
        Xf, yf, Xte2, yte2 = load_split(dataset, SEED, 10**9, n_test)
        assert np.array_equal(yte, yte2), f"{dataset}: test split moved"

        row = {"n_test": int(len(yte))}
        for label, (Xtr, ytr) in {"capped": (Xc, yc), "full": (Xf, yf)}.items():
            tr, te, cats = as_categorical(Xtr, Xte)
            m = tree(cats)
            m.fit(tr, ytr)
            auc = float(roc_auc_score(yte, m.predict_proba(te)[:, 1]))
            row[label] = {"n_train": int(len(ytr)), "auc": round(auc, 5)}
        row["gain"] = round(row["full"]["auc"] - row["capped"]["auc"], 5)
        out["datasets"][dataset] = row
        print(f"{dataset:16s} capped n={row['capped']['n_train']:>6} "
              f"auc={row['capped']['auc']:.4f}   full n={row['full']['n_train']:>6} "
              f"auc={row['full']['auc']:.4f}   gain {row['gain']:+.4f}")

    path = REPO / "results" / "context_reference.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
