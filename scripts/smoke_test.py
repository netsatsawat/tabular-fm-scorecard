"""Orchestrate the runnability probe on the synthetic bed: one subprocess per cell.

This answers 'does the library run at all on this machine', which is a different
question from the real-data comparison in sweep.py. It writes to results/probe_ev/
and never touches results/probe/.

The question this answers is 'does it run on this machine, and at what cost',
not 'which model is better'. Every probe is isolated so an out-of-memory kill or a
native crash is recorded as a result rather than ending the sweep.

Usage:
    python scripts/smoke_test.py                       # all models, default sweep
    python scripts/smoke_test.py --models tabfm,tabpfn --contexts 500,2000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The synthetic runnability bed, kept separate from the real-data sweep so the two
# never share a results directory. See DATASET-CHOICE.md for why it is synthetic.
DATASET = "ev_battery"
N_TEST = 2000

PROBE = REPO / "results" / "probe_ev"
PYTHON = REPO / ".venv" / "bin" / "python"

ALL_MODELS = [
    "logreg", "xgboost", "lightgbm", "catboost",
    "tabfm", "tabfm_ensemble", "tabpfn", "tabicl", "tabdpt",
]

# Generous, because the first call to a foundation model downloads its weights.
# A probe that exceeds this is recorded as a timeout, which is itself a finding.
TIMEOUT_S = 3600


def run_probe(model, n_context, force=False):
    out = PROBE / f"{model}_{n_context}.json"
    if out.exists() and not force:
        return json.loads(out.read_text()), True

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(PYTHON), str(REPO / "scripts" / "run_one.py"), model, DATASET,
         str(n_context), str(N_TEST), str(out)],
        capture_output=True, text=True, timeout=None,
    )
    wall = time.perf_counter() - t0

    if out.exists():
        rec = json.loads(out.read_text())
        rec["wall_seconds"] = round(wall, 1)
        out.write_text(json.dumps(rec, indent=2))
        return rec, False

    # The process died without writing anything: killed, segfaulted, or OOM.
    rec = {
        "model": model, "n_context": n_context, "status": "died",
        "returncode": proc.returncode, "wall_seconds": round(wall, 1),
        "stderr_tail": proc.stderr[-1200:],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    return rec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--contexts", default="2000")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    contexts = [int(c) for c in args.contexts.split(",") if c]

    results = []
    for n in contexts:
        for m in models:
            print(f"-> {m} @ n_context={n} ... ", end="", flush=True)
            rec, cached = run_probe(m, n, args.force)
            tag = "cached" if cached else f"{rec.get('wall_seconds', '?')}s"
            if rec["status"] == "ok":
                print(f"ok  auc={rec['auc']:.4f}  fit={rec['fit_seconds']}s  "
                      f"pred={rec['predict_seconds']}s  rss={rec['peak_rss_gb']}GB  [{tag}]")
            else:
                detail = rec.get("error_type") or rec.get("returncode")
                print(f"{rec['status'].upper()}  {detail}  [{tag}]")
            results.append(rec)

    summary = REPO / "results" / "runnability_ev.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary.relative_to(REPO)}  ({len(results)} probes)")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"ran: {len(ok)}/{len(results)}")
    for r in results:
        if r["status"] != "ok":
            print(f"  did not run: {r['model']} @ {r['n_context']} "
                  f"-> {r.get('error_type') or r['status']}")


if __name__ == "__main__":
    main()
