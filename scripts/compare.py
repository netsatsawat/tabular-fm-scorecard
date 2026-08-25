"""Paired bootstrap comparison of every model against a control, per dataset.

Reads the saved probability vectors, not the summary scores. Two models scored on
identical rows can be compared far more sensitively than two independent confidence
intervals suggest, because the row-level noise is shared and cancels.

One resample-index matrix is drawn per dataset and reused across every model, which
is what makes the comparison paired rather than a set of unrelated intervals. That
detail is borrowed from HR-Analytics/code/statistical_rigour.ipynb.

Holm correction is applied across the models compared on each dataset, because the
question 'does anything beat the control' is a family of tests and reporting the
best raw p-value would overstate it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "results" / "probe"

B_BOOT = 5000
SEED = 1234

CLASSICAL = {"logreg", "xgboost", "lightgbm", "catboost"}

# Each tree library ran twice, once with native categorical handling and once on
# ordinal codes. That is one model given two representations, not two hypotheses,
# so the better of the pair stands for the library and the loser is dropped before
# the Holm family is counted. Testing the same library twice would inflate m and
# make the correction punish the foundation models for a baseline design choice.
# On heloc, which has no categorical columns, the two runs are identical.
PAIRED_VARIANTS = {"xgboost": "xgboost_ord", "lightgbm": "lightgbm_ord",
                   "catboost": "catboost_ord"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    p = np.asarray(pvals, dtype="float64")
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m, dtype="float64")
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def load(dataset):
    """Returns {model: proba}, y_true, and the split metadata they share."""
    out, y_ref, meta = {}, None, None
    for js in sorted(PROBE.glob(f"{dataset}__*.json")):
        rec = json.loads(js.read_text())
        if rec.get("status") != "ok":
            continue
        npz = js.with_suffix(".npz")
        if not npz.exists():
            continue
        d = np.load(npz)
        if y_ref is None:
            y_ref, meta = d["y_true"], rec
        elif not np.array_equal(y_ref, d["y_true"]):
            raise SystemExit(
                f"{dataset}: {rec['model']} was scored on a different test split. "
                "Paired comparison is invalid; re-run the sweep with --force."
            )
        out[rec["model"]] = d["proba"]
    return out, y_ref, meta


def collapse_variants(proba, aucs):
    """Keep the stronger representation of each tree library, note which won."""
    chosen = {}
    for base, alt in PAIRED_VARIANTS.items():
        if base in aucs and alt in aucs:
            winner = base if aucs[base] >= aucs[alt] else alt
            chosen[base] = "categorical" if winner == base else "ordinal"
            if winner != base:
                proba[base] = proba[alt]
                aucs[base] = aucs[alt]
            proba.pop(alt, None)
            aucs.pop(alt, None)
        elif alt in aucs and base not in aucs:
            proba[base] = proba.pop(alt)
            aucs[base] = aucs.pop(alt)
            chosen[base] = "ordinal"
    return chosen


def compare(dataset, control=None):
    proba, y, meta = load(dataset)
    if len(proba) < 2:
        print(f"{dataset}: fewer than two models scored, skipping")
        return None

    aucs = {m: roc_auc_score(y, p) for m, p in proba.items()}
    representation = collapse_variants(proba, aucs)
    if control is None:
        # The strongest non-foundation model, over both representations each tree
        # library was given. Beating a weak control proves nothing, and picking the
        # argmax after seeing the scores maximises the reported gap in whichever
        # direction it points: conservative for a foundation-model win, and
        # anti-conservative for a loss. Both halves are stated in FINDINGS.md.
        classical = {m: a for m, a in aucs.items() if m in CLASSICAL}
        control = max(classical, key=classical.get)

    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.integers(0, n, size=(B_BOOT, n))

    # A resample with one class missing has no defined AUC. Drop those draws once,
    # for every model, so the paired structure survives.
    keep = np.array([len(np.unique(y[row])) == 2 for row in idx])
    idx = idx[keep]

    boot = {}
    for m, p in proba.items():
        boot[m] = np.array([roc_auc_score(y[r], p[r]) for r in idx])

    rows = []
    for m in proba:
        if m == control:
            continue
        diff = boot[m] - boot[control]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        # Two-sided bootstrap p-value: how often the difference crosses zero.
        share = float(np.mean(diff <= 0))
        p_raw = min(1.0, 2 * min(share, 1 - share))
        # A bootstrap cannot resolve below 2/B. Reporting p = 0 would claim a
        # precision the method does not have, so the floor is recorded alongside.
        p_floor = 2.0 / len(idx)
        rows.append({
            "model": m, "auc": round(float(aucs[m]), 4),
            "delta": round(float(aucs[m] - aucs[control]), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "p_raw": round(p_raw, 5),
            "p_below": round(p_floor, 5) if p_raw == 0.0 else None,
        })

    for row, adj in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = round(float(adj), 5)
        row["beats_control"] = bool(row["delta"] > 0 and adj < 0.05)

    rows.sort(key=lambda r: -r["auc"])
    return {
        "dataset": dataset, "control": control,
        "control_auc": round(float(aucs[control]), 4),
        "n_test": int(n), "n_positives": int(y.sum()),
        "n_context": meta.get("n_context"), "bootstrap_draws": int(len(idx)),
        "holm_family_size": len(rows),
        "tree_representation": representation,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="credit_g,heloc,bank_marketing,kdd_appetency")
    args = ap.parse_args()

    all_out = []
    for dataset in args.datasets.split(","):
        res = compare(dataset)
        if res is None:
            continue
        all_out.append(res)
        print(f"\n### {dataset} — control: {res['control']} "
              f"(AUC {res['control_auc']}), {res['n_test']} test rows, "
              f"{res['n_positives']} positives\n")
        print("| model | AUC | delta vs control | 95% CI | p (Holm) | beats control |")
        print("|---|---|---|---|---|---|")
        for r in res["rows"]:
            print(f"| {r['model']} | {r['auc']:.4f} | {r['delta']:+.4f} | "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_holm']:.4f} | "
                  f"{'yes' if r['beats_control'] else 'no'} |")

    path = REPO / "results" / "paired_comparison.json"
    path.write_text(json.dumps(all_out, indent=2))
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
