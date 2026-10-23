
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from features import FEATURE_NAMES, extract_features

DATA = ROOT / "data" / "raw" / "url_features_extracted1.csv"
OUTPUT = ROOT / "reports" / "misclassified_urls.csv"


def get_domain(url):
    try:
        value = str(url).strip()
        if "://" not in value:
            value = "http://" + value
        host = (urlparse(value).hostname or "").lower()
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except ValueError:
        return ""


def main():
    df = pd.read_csv(DATA).dropna(
        subset=["URL", "ClassLabel"]
    ).copy()

    labels = pd.to_numeric(df["ClassLabel"], errors="coerce")
    df = df[labels.isin([0, 1])].copy()
    df["ClassLabel"] = labels.loc[df.index].astype(int)
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]

    df["normalized"] = (
        df["URL"].str.lower()
        .str.replace(r"^https?://", "", regex=True)
        .str.rstrip("/")
    )
    df = df.drop_duplicates("normalized").copy()

    # 1 = phishing, 0 = legitimate.
    y = (df["ClassLabel"] == 0).astype(int)
    df["domain"] = df["URL"].map(get_domain)

    rows, indices = [], []
    for idx, url in df["URL"].items():
        try:
            rows.append(extract_features(url))
            indices.append(idx)
        except (ValueError, TypeError):
            continue

    df = df.loc[indices].copy()
    y = y.loc[indices]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES, index=indices)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=df["domain"])
    )

    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    predicted = model.predict(X.iloc[test_idx])
    actual = y.iloc[test_idx]

    errors = df.iloc[test_idx].copy()
    errors["actual"] = actual.to_numpy()
    errors["predicted"] = predicted
    errors["actual_label"] = errors["actual"].map({
        0: "Legitimate",
        1: "Phishing",
    })
    errors["predicted_label"] = errors["predicted"].map({
        0: "Legitimate",
        1: "Phishing",
    })

    errors = errors[errors["actual"] != errors["predicted"]].copy()
    errors["error_type"] = errors.apply(
        lambda row: (
            "False positive"
            if row["actual"] == 0
            else "False negative"
        ),
        axis=1,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    errors[[
        "URL", "actual_label", "predicted_label", "error_type"
    ]].to_csv(OUTPUT, index=False)

    print("Misclassified URLs:", len(errors))
    print("\nErrors by type:")
    print(errors["error_type"].value_counts().to_string())
    print("\nExamples:")
    print(errors[[
        "URL", "actual_label", "predicted_label", "error_type"
    ]].head(15).to_string(index=False))
    print("\nSaved report:", OUTPUT)
    print("Production model unchanged.")


if __name__ == "__main__":
    main()
