"""Registry of real datasets, fetched from OpenML and cached to data/.

Every entry is observed data — no synthetic tables. The EV battery set the first
probe used is deliberately absent: it is synthetic and saturated, and its role was
to prove the harness works, not to rank anything (see DATASET-CHOICE.md).

All four appear by name in TabArena's 51-dataset suite, checked against the dataset
column of the result parquets Google ships in google-research/tabfm. The OpenML
version pinned here is not necessarily the version TabArena runs, so the overlap is
in the underlying data rather than in the exact task definition. The `tabarena` flag
records membership in the suite, not version identity.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "openml"


@dataclass(frozen=True)
class Spec:
    key: str
    openml_id: int
    target: str
    domain: str
    note: str
    tabarena: bool


SPECS = [
    Spec(
        key="credit_g", openml_id=31, target="class", domain="credit risk",
        note="German credit, 1,000 rows. The small-data regime where in-context "
             "models are claimed to win. Hard target: published AUC sits near 0.79.",
        tabarena=True,
    ),
    Spec(
        key="heloc", openml_id=46932, target="RiskPerformance", domain="credit risk",
        note="FICO home equity lines, ~10.5k rows. Real underwriting data with a "
             "genuine noise floor around 0.80 AUC.",
        tabarena=True,
    ),
    Spec(
        key="bank_marketing", openml_id=1461, target="Class", domain="marketing",
        note="Portuguese bank telemarketing, 45k rows. Real campaign outcomes, "
             "imbalanced at roughly 12% positive.",
        tabarena=True,
    ),
    Spec(
        key="kdd_appetency", openml_id=46939, target="appetency", domain="telecom",
        note="Orange telecom customer records, 50k rows and 213 columns. Severely "
             "imbalanced and heavily missing. The high-dimensional stress case.",
        tabarena=True,
    ),
]

BY_KEY = {s.key: s for s in SPECS}


def fetch(key):
    """Returns (X, y) as a DataFrame and a 0/1 integer array, cached on disk."""
    spec = BY_KEY[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.parquet"

    if not path.exists():
        import openml

        ds = openml.datasets.get_dataset(
            spec.openml_id, download_data=True, download_qualities=False,
            download_features_meta_data=True,
        )
        X, y, _, _ = ds.get_data(target=spec.target)
        frame = X.copy()
        frame["__target__"] = y.to_numpy() if hasattr(y, "to_numpy") else y
        frame.to_parquet(path)

    frame = pd.read_parquet(path)
    y_raw = frame["__target__"]
    X = frame.drop(columns=["__target__"])

    # Targets arrive as strings, categories, or already-encoded integers depending
    # on the upload. Whichever it is, the rarer class becomes 1 so that ROC AUC and
    # the positive rate mean the same thing across every dataset in the registry.
    codes = pd.Categorical(y_raw.astype("string")).codes
    counts = pd.Series(codes).value_counts()
    minority = counts.idxmin()
    y = (codes == minority).astype("int64")

    return X, y
