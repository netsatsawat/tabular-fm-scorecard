"""How much of the target one column can recover, per dataset.

A dataset whose best single feature already scores 0.96 cannot rank models: there is
no headroom left for one to win by. This measures that ceiling so the claim in
DATASET-CHOICE.md has an artifact behind it rather than a shell command someone ran
once.

Numeric columns only, scored directly as a ranking. A categorical column has no
natural order, so an ordinal code would measure the encoder rather than the column.

Writes results/feature_baseline.json.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prep import _load_frame, load_split
from sweep import PROTOCOL

REPO = Path(__file__).resolve().parent.parent
SEED = 1234

# The synthetic bed is scored at the size its own probe used, the real datasets at
# the sizes the sweep uses, so every number here matches a split that exists.
JOBS = {"ev_battery": (2000, 2000), **PROTOCOL}


def main():
    out = {"seed": SEED, "datasets": {}}
    for dataset, (n_context, n_test) in JOBS.items():
        _, _, Xte, yte = load_split(dataset, SEED, n_context, n_test)
        # Shape of the whole table, not just the split, because the prose quotes it.
        Xfull, yfull = _load_frame(dataset)

        scores = []
        for col in Xte.columns:
            if not pd.api.types.is_numeric_dtype(Xte[col]):
                continue
            v = Xte[col].to_numpy(dtype="float64")
            keep = ~np.isnan(v)
            if keep.sum() < 100 or len(np.unique(yte[keep])) < 2:
                continue
            auc = roc_auc_score(yte[keep], v[keep])
            # A column that ranks backwards is just as informative as one that ranks
            # forwards, so the direction is folded out.
            scores.append((col, round(float(max(auc, 1 - auc)), 5)))

        scores.sort(key=lambda kv: -kv[1])
        out["datasets"][dataset] = {
            "n_rows_total": int(len(yfull)),
            "n_columns_total": int(Xfull.shape[1]),
            "missing_cells_total": int(Xfull.isna().to_numpy().sum()),
            "n_test": int(len(yte)),
            "n_numeric_scored": len(scores),
            "best": scores[0][1] if scores else None,
            "above_0_90": sum(1 for _, a in scores if a > 0.90),
            "above_0_80": sum(1 for _, a in scores if a > 0.80),
            "top": [{"feature": c, "auc": a} for c, a in scores[:8]],
        }
        best = out["datasets"][dataset]["best"]
        print(f"{dataset:16s} best single feature {best:.4f}   "
              f"{out['datasets'][dataset]['above_0_90']} above 0.90   "
              f"{out['datasets'][dataset]['above_0_80']} above 0.80")

    path = REPO / "results" / "feature_baseline.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
