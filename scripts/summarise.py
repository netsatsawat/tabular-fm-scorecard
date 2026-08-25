"""Turn the probe JSONs into the tables the README quotes.

Reads results/probe/*.json only. Runs no model and imports nothing from the model
libraries, so it works in any interpreter and cannot accidentally re-run a probe.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "results" / "probe"

FAMILY = {
    "logreg": "linear", "xgboost": "GBDT", "lightgbm": "GBDT", "catboost": "GBDT",
    "tabfm": "foundation", "tabfm_single": "foundation", "tabfm_ensemble": "foundation",
    "tabpfn": "foundation", "tabicl": "foundation", "tabdpt": "foundation",
}

LABEL = {
    "logreg": "Logistic regression", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "tabfm": "TabFM", "tabfm_single": "TabFM (1 estimator)",
    "tabfm_ensemble": "TabFM (ensemble preset)", "tabpfn": "TabPFN",
    "tabicl": "TabICL", "tabdpt": "TabDPT",
}

ORDER = ["credit_g", "heloc", "bank_marketing", "kdd_appetency"]


def load():
    out = []
    for p in sorted(PROBE.glob("*.json")):
        rec = json.loads(p.read_text())
        if "dataset" in rec:
            out.append(rec)
    return out


def per_dataset(recs):
    for ds in ORDER:
        rows = [r for r in recs if r["dataset"] == ds]
        if not rows:
            continue
        ok = [r for r in rows if r["status"] == "ok"]
        ok.sort(key=lambda r: -r["auc"])
        head = ok[0] if ok else rows[0]
        print(f"\n### {ds} — {head.get('n_context','?')} context rows, "
              f"{head.get('n_test','?')} test rows, "
              f"{head.get('n_positives_test','?')} positives, "
              f"{head.get('n_features','?')} features\n")
        print("| model | family | ROC AUC | fit (s) | predict (s) | predict s/1K | peak RSS (GB) |")
        print("|---|---|---|---|---|---|---|")
        for r in ok:
            print(f"| {LABEL.get(r['model'], r['model'])} | {FAMILY.get(r['model'],'?')} "
                  f"| {r['auc']:.4f} | {r['fit_seconds']:.3f} | {r['predict_seconds']:.3f} "
                  f"| {r['predict_seconds_per_1k']:.4f} | {r['peak_rss_gb']} |")
        for r in rows:
            if r["status"] != "ok":
                print(f"| {LABEL.get(r['model'], r['model'])} | {FAMILY.get(r['model'],'?')} "
                      f"| did not run ({r.get('error_type') or r['status']}) | — | — | — | — |")


def cost_table(recs):
    """Predict cost per 1,000 rows, model by dataset. The business column."""
    ok = [r for r in recs if r["status"] == "ok"]
    models = sorted({r["model"] for r in ok}, key=lambda m: (FAMILY.get(m, "z"), m))
    seen = [d for d in ORDER if any(r["dataset"] == d for r in ok)]
    print("\n### Prediction cost, seconds per 1,000 rows\n")
    print("| model | " + " | ".join(seen) + " |")
    print("|---" * (len(seen) + 1) + "|")
    for m in models:
        cells = []
        for d in seen:
            hit = [r for r in ok if r["model"] == m and r["dataset"] == d]
            cells.append(f"{hit[0]['predict_seconds_per_1k']:,.4f}" if hit else "—")
        print(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-only", action="store_true")
    args = ap.parse_args()

    recs = load()
    if not recs:
        print("no probes found")
        return
    if not args.cost_only:
        per_dataset(recs)
    cost_table(recs)

    missing = [r for r in recs if r["status"] != "ok"]
    print(f"\n{len(recs) - len(missing)} of {len(recs)} probes ran.")
    for r in missing:
        print(f"  {r['dataset']}/{r['model']}: {r.get('error_type') or r['status']}")


if __name__ == "__main__":
    main()
