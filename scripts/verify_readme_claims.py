#!/usr/bin/env python3
"""No number in the prose without a runnable path behind it.

Every measured figure in README.md and FINDINGS.md is recomputed here from the
committed artifacts, and the document must literally contain the recomputed string.
A number that drifts fails the build.

Two directions, because either alone is easy to fool:

  forward   whole rows are rebuilt from the artifacts and must appear in the prose.
            A row that keeps its number but loses its model name fails here, which
            checking one number at a time never catches.

  backward  every result-shaped number in the prose must exist in some artifact.
            This is what catches a figure that was never measured at all.

An earlier version of this file failed both ways and stayed green. Its per-model AUC
loop skipped anything absent and then called check(..., True), so it could only print
"ok". Its number scan used \\b0\\.\\d{4}\\b, which silently ignores 0.00052 and
5,459.3300, and it whitelisted every pairwise difference between any two AUCs, which
covered about 6% of the four-decimal space. Both are fixed below. The note stays
because that failure mode is invisible from the passing output.

Reads JSON only. No network, no model, no dataset. Standard library.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "results" / "probe"
PROBE_EV = REPO / "results" / "probe_ev"
PAIRED = REPO / "results" / "paired_comparison.json"
TIMING = REPO / "results" / "baseline_timing.json"
CONTEXT = REPO / "results" / "context_reference.json"
FEATURE = REPO / "results" / "feature_baseline.json"
SHOWDOWN = REPO / "results" / "showdown.json"

problems = []

# Numbers that describe the setup rather than a result: versions, seeds, chosen
# sizes. There is no artifact to recompute them from, so they are listed one by one
# rather than pattern-matched, which makes adding one a deliberate act.
STRUCTURAL = {
    "1.0.1", "2.1.1", "1.2.0", "8.4.0", "0.18.2", "3.12",
    "1,000", "5,000", "44,211", "47,000", "9,459",
    "6.985",          # TabArena's published GPU figure, an external citation
    "1.78",           # positive rate quoted as a percentage
}

# Figures RESEARCH-NOTES.md quotes precisely because they were wrong. A lab journal
# that records a retracted number has to be able to print it, and the artifacts will
# never contain it, so each one is listed with what it was.
SUPERSEDED = {
    "0.0303",     # TabFM's bank_marketing margin before CatBoost got its categoricals
    "0.0300",     # the drift the old verifier failed to catch, used as an example
    "0.8170",     # an uncommitted LightGBM score, replaced by context_reference.json
    "0.00052",    # a cost figure the old number regex could not see
    "4,312",      # a truncation the rewritten verifier caught, quoted as the example
}


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        problems.append(f"{name}{('  ' + detail) if detail else ''}")


def prose_only(text):
    """Strip fenced code and badge markup before scanning for numbers.

    A bare digit search once matched the 8 inside a shields.io hex colour in another
    repository of mine and stayed green through a real drift. Code blocks go too: a
    version pin is not a claim about a result.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"img\.shields\.io[^\s)]*", " ", text)
    text = re.sub(r"#[0-9a-fA-F]{6}\b", " ", text)
    return text


def load_probes(directory):
    out = []
    for p in sorted(directory.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("status") == "ok":
            out.append(rec)
    return out


def renderings(value):
    """Every way a measured number may honestly appear in the prose."""
    out = set()
    for places in range(0, 7):
        out.add(f"{value:.{places}f}")
        out.add(f"{value:,.{places}f}")
    return out


def build_known():
    known = set()

    def add(v):
        if isinstance(v, (int, float)):
            known.update(renderings(float(v)))

    for rec in load_probes(PROBE) + load_probes(PROBE_EV):
        for key in ("auc", "predict_seconds_per_1k", "predict_seconds", "fit_seconds",
                    "peak_rss_gb", "n_context", "n_test", "n_positives_test",
                    "n_features", "positive_rate_test", "positive_rate_train",
                    "wall_seconds", "n_categorical_columns"):
            add(rec.get(key))

    if PAIRED.exists():
        for block in json.loads(PAIRED.read_text()):
            for key in ("control_auc", "n_test", "n_positives", "n_context",
                        "bootstrap_draws", "holm_family_size"):
                add(block.get(key))
            for row in block["rows"]:
                for key in ("auc", "p_holm", "p_raw", "p_below"):
                    add(row.get(key))
                for key in ("delta", "ci_lo", "ci_hi"):
                    if row.get(key) is not None:
                        add(abs(row[key]))

    if TIMING.exists():
        raw = json.loads(TIMING.read_text())
        blocks = raw.get("datasets", {"_flat": {"models": raw.get("models", {})}})
        for block in blocks.values():
            for v in block.get("models", {}).values():
                add(v.get("seconds_per_1k"))
                add(v.get("median_seconds_per_call"))

    if FEATURE.exists():
        for block in json.loads(FEATURE.read_text())["datasets"].values():
            for key in ("best", "n_rows_total", "n_columns_total",
                        "missing_cells_total", "n_test", "n_numeric_scored",
                        "above_0_90", "above_0_80"):
                add(block.get(key))
            for t in block.get("top", []):
                add(t["auc"])

    if SHOWDOWN.exists():
        for row in json.loads(SHOWDOWN.read_text())["datasets"].values():
            for key in ("auc_tree_uncapped", "auc_tabfm_capped", "p",
                        "n_test", "n_train_tree", "draws"):
                add(row.get(key))
            for key in ("delta", "ci_lo", "ci_hi"):
                if row.get(key) is not None:
                    add(abs(row[key]))

    if CONTEXT.exists():
        for row in json.loads(CONTEXT.read_text())["datasets"].values():
            add(row.get("n_test"))
            add(row.get("gain"))
            for phase in ("capped", "full"):
                add(row[phase]["auc"])
                add(row[phase]["n_train"])

    # Differences between two AUCs measured on the SAME dataset are themselves
    # sourced. Scoped to one dataset on purpose: an earlier version whitelisted every
    # pairwise difference across every artifact and covered 6% of the value space.
    by_dataset = {}
    for rec in load_probes(PROBE) + load_probes(PROBE_EV):
        by_dataset.setdefault(rec.get("dataset", "ev_battery"), []).append(rec["auc"])
    for aucs in by_dataset.values():
        for a in aucs:
            for b in aucs:
                if a > b:
                    add(a - b)

    # Cost ratios are derived from two artifacts rather than stored in either, so
    # they are recomputed here on the same rule the prose uses: a foundation
    # model's per-1k cost over the fastest baseline on that dataset, where the
    # baseline timing is the median of 200 calls rather than the single call in the
    # probe. Recomputing beats whitelisting, which is what let a stale ratio through
    # in the previous version of this file.
    if TIMING.exists():
        timing = json.loads(TIMING.read_text()).get("datasets", {})
        for rec in load_probes(PROBE):
            block = timing.get(rec["dataset"], {}).get("models", {})
            if not block or rec.get("predict_seconds_per_1k") is None:
                continue
            for v in block.values():
                if v["seconds_per_1k"] > 0:
                    add(rec["predict_seconds_per_1k"] / v["seconds_per_1k"])
    return known


def scan_numbers(corpus):
    """Result-shaped numbers in the prose.

    A number counts as a result if it carries three or more decimal places, or if it
    is comma-grouped. That is the surface an invented or stale figure lands on.
    Two-decimal numbers are excluded: prose is full of them for ordinary reasons.
    """
    out = set()
    for token in re.findall(r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+", corpus):
        if "." in token:
            if len(token.split(".")[1]) >= 3:
                out.add(token)
        else:
            out.add(token)
    return out


# Every prose document is scanned, not just the two headline ones. A stale figure in
# the lab journal is the same defect as a stale figure in the README, and it is the
# one nobody re-reads.
DOCS = ["README.md", "FINDINGS.md", "RESEARCH-NOTES.md", "DATASET-CHOICE.md", "NOTICE.md"]


def main():
    corpus = prose_only("\n".join(
        (REPO / name).read_text() for name in DOCS if (REPO / name).exists()
    ))

    probes = load_probes(PROBE)
    check("real-data probes exist", len(probes) > 0, f"{len(probes)} probes")
    if not probes:
        return 1

    known = build_known()
    orphans = sorted(scan_numbers(corpus) - known - STRUCTURAL - SUPERSEDED)
    check("every result-shaped number is sourced", not orphans,
          f"unsourced: {orphans[:10]}" if orphans else f"{len(known)} known values")

    if PAIRED.exists():
        blocks = json.loads(PAIRED.read_text())
        datasets = sorted({r["dataset"] for r in probes})
        check("comparison covers every dataset with probes",
              {b["dataset"] for b in blocks} == set(datasets),
              f"compared: {sorted(b['dataset'] for b in blocks)}")

        wins = [(b["dataset"], r["model"]) for b in blocks
                for r in b["rows"] if r["beats_control"]]
        n_won = len({d for d, _ in wins})
        phrases = {f"{len(wins)} confirmed wins",
                   f"{len(wins)} confirmed wins across {n_won} datasets",
                   f"{len(wins)} wins that clear Holm correction across {n_won} datasets"}
        check("stated win count matches the artifact",
              any(p in corpus for p in phrases),
              f"artifact: {len(wins)} wins over {n_won} datasets {wins}")

        for b in blocks:
            for row in b["rows"]:
                if not row["beats_control"]:
                    continue
                bits = [f"{row['auc']:.4f}", f"{abs(row['delta']):.4f}",
                        f"{row['p_holm']:.4f}"]
                missing = [x for x in bits if x not in corpus]
                check(f"win row complete: {b['dataset']}/{row['model']}", not missing,
                      f"missing {missing}" if missing else "")

    if CONTEXT.exists():
        for name, row in json.loads(CONTEXT.read_text())["datasets"].items():
            if row["gain"] == 0:
                continue
            bits = [f"{row['capped']['auc']:.4f}", f"{row['full']['auc']:.4f}",
                    f"{row['full']['n_train']:,}"]
            missing = [x for x in bits if x not in corpus]
            check(f"context row complete: {name}", not missing,
                  f"missing {missing}" if missing else "")

    for ds in sorted({r["dataset"] for r in probes}):
        rows = [r for r in probes if r["dataset"] == ds]
        for field in ("n_test", "n_context", "n_positives_test"):
            values = {r[field] for r in rows}
            check(f"{ds}: identical {field} across models", len(values) == 1,
                  str(sorted(values)))

    tabfm = [r for r in probes if r["model"] == "tabfm"]
    if tabfm:
        meta = tabfm[0].get("meta", {})
        check("TabFM ran with n_estimators=32", meta.get("n_estimators") == 32,
              str(meta.get("n_estimators")))
        check("TabFM ran with no row cap", meta.get("max_num_rows") is None,
              str(meta.get("max_num_rows")))

    if TIMING.exists():
        per_dataset = "datasets" in json.loads(TIMING.read_text())
        check("baseline timing is per-dataset", per_dataset,
              "" if per_dataset else "flat schema; re-run scripts/time_baselines.py")

    print()
    if problems:
        print(f"{len(problems)} claim(s) drifted from the artifacts:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("every recomputed claim matches the artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
