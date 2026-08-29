"""The comparison the benchmark protocol forbids, and the one a business asks for.

Every model in the sweep trains on the same 700 or 1,000 rows, because that is what
TabFM can afford. That is the right rule for a controlled comparison and the wrong
question for anyone deciding what to deploy, where the tree would simply be given
every row the company already has.

This scores the uncapped tree against TabFM on the identical held-out rows, with the
same paired bootstrap the rest of the study uses. A difference of a ten-thousandth
is not a result until you can show the interval around it.

Writes results/showdown.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "results" / "probe"

B_BOOT = 5000
SEED = 1234


def paired(y, a, b):
    """Bootstrap the AUC difference between two prediction vectors on shared rows."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.integers(0, n, size=(B_BOOT, n))
    idx = idx[np.array([len(np.unique(y[r])) == 2 for r in idx])]

    diff = np.array([
        roc_auc_score(y[r], a[r]) - roc_auc_score(y[r], b[r]) for r in idx
    ])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    share = float(np.mean(diff <= 0))
    return {
        "delta": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 6),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "p": round(min(1.0, 2 * min(share, 1 - share)), 4),
        "draws": int(len(idx)),
    }


def main():
    out = {"seed": SEED, "bootstrap": B_BOOT, "datasets": {}}
    for full in sorted(PROBE.glob("*__lightgbm_full__*.npz")):
        dataset = full.name.split("__")[0]
        fm_path = PROBE / f"{dataset}__tabfm__c1000.npz"
        if not fm_path.exists():
            fm_path = next(PROBE.glob(f"{dataset}__tabfm__*.npz"), None)
        if fm_path is None:
            continue

        tree, fm = np.load(full), np.load(fm_path)
        if not np.array_equal(tree["y_true"], fm["y_true"]):
            raise SystemExit(f"{dataset}: the two runs used different test rows")

        y = tree["y_true"]
        auc_tree = float(roc_auc_score(y, tree["proba"]))
        auc_fm = float(roc_auc_score(y, fm["proba"]))
        row = paired(y, tree["proba"], fm["proba"])
        row.update(
            auc_tree_uncapped=round(auc_tree, 5),
            auc_tabfm_capped=round(auc_fm, 5),
            # Kept at full precision because rounding to four places is exactly how a
            # dead heat turns into a phantom 0.0001 gap.
            raw_difference=float(auc_tree - auc_fm),
            n_test=int(len(y)),
            n_train_tree=int(full.name.split("__c")[1].split(".")[0]),
        )
        out["datasets"][dataset] = row

        verdict = "indistinguishable" if row["p"] >= 0.05 else "separated"
        print(f"{dataset:16s} tree(uncapped) {auc_tree:.5f}  TabFM {auc_fm:.5f}  "
              f"delta {row['delta']:+.5f}  CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
              f"p={row['p']:.4f}  {verdict}")

    path = REPO / "results" / "showdown.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
