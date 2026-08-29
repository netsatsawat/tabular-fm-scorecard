"""Run the real-data sweep: every model on every dataset, identical splits.

Split sizes are fixed per dataset rather than globally, for two reasons. The test
split has to carry enough positives for ROC AUC to mean anything: a 1.78% positive
rate needs thousands of rows to get past a hundred events. And TabFM's cost grows
with context multiplied by query, so a size that is free for a tree is hours for it.

The sizes below are the compromise, and every model on a given dataset sees exactly
the same rows, so the comparison stays paired.

Usage:
    python scripts/sweep.py --models lightgbm,xgboost --datasets all
    python scripts/sweep.py --models tabfm --datasets credit_g,heloc
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "results" / "probe"
PYTHON = REPO / ".venv" / "bin" / "python"

# dataset -> (n_context, n_test). Ordered cheapest-first for TabFM.
PROTOCOL = {
    "credit_g": (700, 300),
    "heloc": (1000, 1000),
    "bank_marketing": (1000, 1000),
    "kdd_appetency": (1000, 3000),
}

FAST = ["logreg", "xgboost", "lightgbm", "catboost", "xgboost_ord",
        "lightgbm_ord", "catboost_ord", "tabdpt", "tabicl"]
SLOW = ["tabfm"]

# Measured on this machine: 16,776.95 s for 2,000 context by 2,000 query rows.
# Used only to print an expectation before a long run, never as a result.
TABFM_S_PER_CELL = 16776.951 / (2000 * 2000)


def probe_path(dataset, model, n_context):
    return PROBE / f"{dataset}__{model}__c{n_context}.json"


def run(model, dataset, force=False):
    n_context, n_test = PROTOCOL[dataset]
    out = probe_path(dataset, model, n_context)
    if out.exists() and not force:
        return json.loads(out.read_text()), True

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(PYTHON), str(REPO / "scripts" / "run_one.py"), model, dataset,
         str(n_context), str(n_test), str(out)],
        capture_output=True, text=True,
    )
    wall = time.perf_counter() - t0

    if out.exists():
        rec = json.loads(out.read_text())
        rec["wall_seconds"] = round(wall, 1)
        out.write_text(json.dumps(rec, indent=2))
        return rec, False

    rec = {
        "model": model, "dataset": dataset, "n_context": n_context,
        "status": "died", "returncode": proc.returncode,
        "wall_seconds": round(wall, 1), "stderr_tail": proc.stderr[-1200:],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    return rec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(FAST))
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    names = list(PROTOCOL) if args.datasets == "all" else args.datasets.split(",")

    if any(m.startswith("tabfm") for m in models):
        total = sum(TABFM_S_PER_CELL * PROTOCOL[d][0] * PROTOCOL[d][1] for d in names)
        print(f"TabFM projected total: {total / 3600:.1f} h across {len(names)} datasets")
        for d in names:
            c, t = PROTOCOL[d]
            print(f"    {d:16s} {c} x {t} -> {TABFM_S_PER_CELL * c * t / 60:.0f} min")
        print()

    for dataset in names:
        for model in models:
            print(f"-> {dataset:16s} {model:10s} ... ", end="", flush=True)
            rec, cached = run(model, dataset, args.force)
            tag = "cached" if cached else f"{rec.get('wall_seconds', '?')}s"
            if rec["status"] == "ok":
                print(f"ok  auc={rec['auc']:.4f}  pred={rec['predict_seconds']}s  "
                      f"rss={rec['peak_rss_gb']}GB  [{tag}]")
            else:
                print(f"{rec['status'].upper()}  "
                      f"{rec.get('error_type') or rec.get('returncode')}  [{tag}]")


if __name__ == "__main__":
    main()
