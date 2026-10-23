
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from features import FEATURE_NAMES, extract_features

DATA_PATH = ROOT / "data" / "raw" / "url_features_extracted1.csv"
REPORT_PATH = ROOT / "reports" / "split_comparison.csv"
SEED = 42


def hostname(url):
    try:
        value = str(url).strip()
        if "://" not in value:
            value = "http://" + value
        return (urlparse(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def approximate_domain(host):
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in {
        "co.uk", "org.uk", "ac.uk", "com.au",
        "net.au", "org.au", "co.in", "firm.in",
        "net.in", "co.jp", "co.nz", "com.br",
    }:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["URL", "ClassLabel"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]
    labels = pd.to_numeric(df["ClassLabel"], errors="coerce")
    df = df[labels.isin([0, 1])].copy()
    df["ClassLabel"] = labels.loc[df.index].astype(int)

    # Remove duplicate URLs.
    df["normalized"] = (
        df["URL"].str.lower()
        .str.replace(r"^https?://", "", regex=True)
        .str.rstrip("/")
    )
    df = df.drop_duplicates("normalized").copy()

    # Dataset: 0 = phishing, 1 = legitimate.
    y = (df["ClassLabel"] == 0).astype(int)

    df["host"] = df["URL"].map(hostname)
    df["domain"] = df["host"].map(approximate_domain)
    valid = df["host"].ne("") & df["domain"].ne("")
    df = df.loc[valid].copy()
    y = y.loc[df.index]

    print("Extracting features...")
    rows = []
    good_indices = []

    for idx, url in df["URL"].items():
        try:
            rows.append(extract_features(url))
            good_indices.append(idx)
        except (ValueError, TypeError):
            pass

    df = df.loc[good_indices].copy()
    y = y.loc[good_indices]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES, index=good_indices)

    print("Usable URLs:", len(X))
    print("Unique approximate domains:", df["domain"].nunique())

    def evaluate(name, train_idx, test_idx):
        model = RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx]

        result = {
            "split": name,
            "accuracy": accuracy_score(actual, pred),
            "precision": precision_score(actual, pred, zero_division=0),
            "phishing_recall": recall_score(actual, pred, zero_division=0),
            "f1": f1_score(actual, pred, zero_division=0),
        }

        print(f"\n{name}")
        for key, value in result.items():
            if key != "split":
                print(f"{key}: {value:.4f}")
        return result

    # A: Random URL split — reference comparison.
    all_indices = list(range(len(X)))
    train_a, test_a = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    results = [
        evaluate("Random URL split", train_a, test_a)
    ]

    # B: Grouped split — keeps approximate domains separate.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )
    train_b, test_b = next(
        splitter.split(X, y, groups=df["domain"])
    )

    train_domains = set(df["domain"].iloc[train_b])
    test_domains = set(df["domain"].iloc[test_b])
    print("\nOverlapping approximate domains:",
          len(train_domains & test_domains))

    results.append(
        evaluate("Domain-grouped split", train_b, test_b)
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORT_PATH, index=False)
    print("\nComparison saved to:", REPORT_PATH)
    print("\nExperiment finished. Production model unchanged.")


if __name__ == "__main__":
    main()
